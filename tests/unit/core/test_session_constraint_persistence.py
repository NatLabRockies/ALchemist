"""Linear input constraints must survive save_session -> load_session.

They did not. ``save_session`` wrote a ``search_space`` block of
``{'variables', 'derived_variables'}`` with no ``constraints`` key at all, so
the constraints were never dropped by a bug -- they were never written, and
``load_session`` had nothing to restore. A user registered constraints, saved,
reloaded, and every later DoE and acquisition call sampled the excluded region
with no warning.

The blast radius was wider than the desktop Save/Load menu: ``session_store``
persists every REST session through the same ``save_session`` and rehydrates it
through ``load_session`` (``_save_to_disk`` / ``_load_from_disk``), so an API
server restart, ``POST /sessions/{id}/save`` and every recovery backup lost them
too.

These tests assert the restored constraints are *live* -- they steer a design
and a feasibility verdict on the reloaded session -- not merely echoed back by
``get_constraints()``. A restore that rebuilt nothing usable would pass a
list comparison.
"""

import json

import pytest

from alchemist_core import OptimizationSession


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _mixed_session():
    """Five variables spanning every type, so no test rides on one of them.

    ``c1`` (categorical) and ``ctx`` (context) are present but unreferenced:
    ``add_constraint`` refuses a non-numeric variable, so a saved constraint
    can never name one, and their presence proves the round trip is not
    disturbed by variables the constraints do not touch.
    """
    s = OptimizationSession()
    s.add_variable("x1", "real", bounds=(0.0, 10.0))
    s.add_variable("x2", "integer", bounds=(0, 8))
    s.add_variable("x3", "discrete", allowed_values=[0.0, 2.0, 4.0, 6.0])
    s.add_variable("c1", "categorical", categories=["A", "B"])
    s.add_variable("ctx", "context")
    return s


def _register_mixed_constraints(s):
    """Four constraints: both types, both signs, explicit and auto names."""
    # All-positive inequality.
    s.add_input_constraint("inequality", {"x1": 1.0, "x2": 2.0}, rhs=12.0,
                           name="budget")
    # Equality carrying a negative coefficient.
    s.add_input_constraint("equality", {"x1": 1.0, "x3": -1.0}, rhs=0.0,
                           name="balance")
    # All-negative inequality -- a lower bound written as -x1 - x2 <= -3.
    s.add_input_constraint("inequality", {"x1": -1.0, "x2": -1.0}, rhs=-3.0,
                           name="floor")
    # Auto-named, so the round trip is tested on a generated name too.
    s.add_input_constraint("inequality", {"x3": 1.0}, rhs=6.0)


def _round_trip(session, tmp_path, filename="session.json"):
    path = tmp_path / filename
    session.save_session(str(path))
    return OptimizationSession.load_session(str(path), retrain_on_load=False), path


# ---------------------------------------------------------------------------
# The writer
# ---------------------------------------------------------------------------

class TestSaveSessionWritesConstraints:
    def test_search_space_block_carries_a_constraints_key(self, tmp_path):
        """The key that was missing. Same name and place as save_to_json's."""
        s = _mixed_session()
        _register_mixed_constraints(s)
        path = tmp_path / "session.json"
        s.save_session(str(path))

        data = json.loads(path.read_text())
        assert "constraints" in data["search_space"], (
            f"search_space block is {sorted(data['search_space'])}; the "
            f"constraints were never written"
        )
        assert len(data["search_space"]["constraints"]) == 4

    def test_written_constraints_are_field_for_field_what_was_registered(self, tmp_path):
        s = _mixed_session()
        _register_mixed_constraints(s)
        path = tmp_path / "session.json"
        s.save_session(str(path))

        written = json.loads(path.read_text())["search_space"]["constraints"]
        assert written == [
            {"type": "inequality", "coefficients": {"x1": 1.0, "x2": 2.0},
             "rhs": 12.0, "name": "budget"},
            {"type": "equality", "coefficients": {"x1": 1.0, "x3": -1.0},
             "rhs": 0.0, "name": "balance"},
            {"type": "inequality", "coefficients": {"x1": -1.0, "x2": -1.0},
             "rhs": -3.0, "name": "floor"},
            {"type": "inequality", "coefficients": {"x3": 1.0},
             "rhs": 6.0, "name": "constraint_0"},
        ]

    def test_zero_constraints_writes_an_empty_list_not_junk(self, tmp_path):
        """A constraint-free session stays clean, and the key is still there.

        The key is written unconditionally on purpose: its *presence* is what
        separates a file saved before constraint persistence existed (absent,
        constraint set unknown) from one saved after it with genuinely none
        (present, empty).
        """
        s = _mixed_session()
        path = tmp_path / "session.json"
        s.save_session(str(path))

        space = json.loads(path.read_text())["search_space"]
        assert space["constraints"] == []
        assert [v["name"] for v in space["variables"]] == [
            "x1", "x2", "x3", "c1", "ctx"
        ]

    def test_saving_does_not_disturb_the_live_constraint_list(self, tmp_path):
        """Writing is a read of session state, not a mutation of it."""
        s = _mixed_session()
        _register_mixed_constraints(s)
        before = s.search_space.get_constraints()
        s.save_session(str(tmp_path / "session.json"))
        assert s.search_space.get_constraints() == before


# ---------------------------------------------------------------------------
# The round trip
# ---------------------------------------------------------------------------

class TestConstraintsSurviveTheRoundTrip:
    def test_every_field_of_every_constraint_survives(self, tmp_path):
        s = _mixed_session()
        _register_mixed_constraints(s)
        before = s.search_space.get_constraints()

        loaded, _ = _round_trip(s, tmp_path)

        assert loaded.search_space.get_constraints() == before

    def test_names_survive_exactly(self, tmp_path):
        """Names are the delete identity for ``DELETE /constraints/{name}``.

        A round trip that restored constraints but regenerated their names
        would silently break deletion on every reloaded session, so this is
        asserted on its own rather than only inside the whole-dict comparison.
        """
        s = _mixed_session()
        _register_mixed_constraints(s)

        loaded, _ = _round_trip(s, tmp_path)

        assert [c["name"] for c in loaded.search_space.get_constraints()] == [
            "budget", "balance", "floor", "constraint_0"
        ]

    def test_both_types_and_both_signs_survive_together(self, tmp_path):
        s = _mixed_session()
        _register_mixed_constraints(s)

        loaded, _ = _round_trip(s, tmp_path)
        restored = {c["name"]: c for c in loaded.search_space.get_constraints()}

        assert restored["budget"]["type"] == "inequality"
        assert restored["balance"]["type"] == "equality"
        assert restored["balance"]["coefficients"]["x3"] == -1.0
        assert restored["floor"]["coefficients"] == {"x1": -1.0, "x2": -1.0}
        assert restored["floor"]["rhs"] == -3.0

    def test_a_restored_name_is_the_delete_identity(self, tmp_path):
        """Exactly what ``DELETE /constraints/{name}`` does, on a reloaded space."""
        s = _mixed_session()
        _register_mixed_constraints(s)

        loaded, _ = _round_trip(s, tmp_path)
        existing = loaded.search_space.constraints
        match = [c for c in existing if c["name"] == "balance"]
        assert len(match) == 1
        existing.remove(match[0])

        assert [c["name"] for c in loaded.search_space.get_constraints()] == [
            "budget", "floor", "constraint_0"
        ]

    def test_auto_naming_resumes_past_the_restored_names(self, tmp_path):
        """The generator derives the next index from the names present.

        If the restore lost the names, the next auto name would resynchronize
        to ``constraint_0`` and collide with a constraint already held.
        """
        s = _mixed_session()
        _register_mixed_constraints(s)

        loaded, _ = _round_trip(s, tmp_path)
        loaded.add_input_constraint("inequality", {"x2": 1.0}, rhs=5.0)

        names = [c["name"] for c in loaded.search_space.get_constraints()]
        assert names[-1] == "constraint_1"
        assert len(set(names)) == len(names), names

    def test_instance_method_load_restores_constraints_too(self, tmp_path):
        """``load_session`` has two entry points; both must carry them."""
        s = _mixed_session()
        _register_mixed_constraints(s)
        path = tmp_path / "session.json"
        s.save_session(str(path))

        target = OptimizationSession()
        target.load_session(str(path), retrain_on_load=False)

        assert [c["name"] for c in target.search_space.get_constraints()] == [
            "budget", "balance", "floor", "constraint_0"
        ]

    def test_round_trip_is_a_fixed_point(self, tmp_path):
        """save -> load -> save writes the identical constraint block."""
        s = _mixed_session()
        _register_mixed_constraints(s)
        first = tmp_path / "a.json"
        s.save_session(str(first))

        loaded = OptimizationSession.load_session(str(first), retrain_on_load=False)
        second = tmp_path / "b.json"
        loaded.save_session(str(second))

        assert (json.loads(second.read_text())["search_space"]["constraints"]
                == json.loads(first.read_text())["search_space"]["constraints"])

    def test_zero_constraints_round_trips_as_zero(self, tmp_path):
        s = _mixed_session()
        loaded, _ = _round_trip(s, tmp_path)
        assert loaded.search_space.get_constraints() == []
        assert len(loaded.search_space.variables) == 5


# ---------------------------------------------------------------------------
# Restored constraints must be LIVE, not merely present
# ---------------------------------------------------------------------------

class TestRestoredConstraintsAreFunctionallyLive:
    """Feasibility is computed by hand here, not read back from the object.

    ``filter_feasible`` sums only the constraint terms whose column is present
    in the frame it is handed and skips a constraint entirely only when *none*
    of its columns are present, so it can return a confidently wrong verdict
    for a half-restored constraint rather than raising. Every expectation below
    is therefore derived from literal arithmetic on the same numbers.
    """

    def test_inequality_rejects_a_point_it_excludes(self, tmp_path):
        s = OptimizationSession()
        s.add_variable("x1", "real", bounds=(0.0, 10.0))
        s.add_variable("x2", "integer", bounds=(0, 8))
        s.add_input_constraint("inequality", {"x1": 1.0, "x2": 2.0}, rhs=12.0,
                               name="budget")

        loaded, _ = _round_trip(s, tmp_path)

        # 1.0*3.0 + 2.0*2 = 7.0 <= 12.0  -> feasible
        inside = {"x1": 3.0, "x2": 2}
        assert 1.0 * 3.0 + 2.0 * 2 == 7.0
        # 1.0*9.0 + 2.0*7 = 23.0 > 12.0  -> infeasible
        outside = {"x1": 9.0, "x2": 7}
        assert 1.0 * 9.0 + 2.0 * 7 == 23.0

        assert loaded.search_space.is_feasible(inside) is True
        assert loaded.search_space.is_feasible(outside) is False

    def test_equality_with_a_negative_coefficient_is_live(self, tmp_path):
        s = OptimizationSession()
        s.add_variable("x1", "real", bounds=(0.0, 10.0))
        s.add_variable("x3", "discrete", allowed_values=[0.0, 2.0, 4.0, 6.0])
        s.add_input_constraint("equality", {"x1": 1.0, "x3": -1.0}, rhs=0.0,
                               name="balance")

        loaded, _ = _round_trip(s, tmp_path)

        # 1.0*4.0 + (-1.0)*4.0 = 0.0 == 0.0 -> feasible
        assert 1.0 * 4.0 + (-1.0) * 4.0 == 0.0
        assert loaded.search_space.is_feasible({"x1": 4.0, "x3": 4.0}) is True
        # 1.0*4.0 + (-1.0)*6.0 = -2.0 != 0.0 -> infeasible
        assert 1.0 * 4.0 + (-1.0) * 6.0 == -2.0
        assert loaded.search_space.is_feasible({"x1": 4.0, "x3": 6.0}) is False

    def test_equality_with_a_nonzero_rhs_and_unequal_magnitudes(self, tmp_path):
        """The sibling above balances to zero, where rhs and -rhs agree.

        A restore that flipped the sign of every rhs would slide past it. This
        one carries rhs = 5.0 and coefficients of different magnitude, so both
        the sign and the value have to arrive intact.
        """
        s = OptimizationSession()
        s.add_variable("x1", "real", bounds=(0.0, 10.0))
        s.add_variable("x2", "integer", bounds=(0, 8))
        s.add_input_constraint("equality", {"x1": 2.0, "x2": -1.0}, rhs=5.0,
                               name="offset")

        loaded, _ = _round_trip(s, tmp_path)

        # 2.0*4.5 + (-1.0)*4 = 5.0 == 5.0 -> feasible
        assert 2.0 * 4.5 + (-1.0) * 4 == 5.0
        assert loaded.search_space.is_feasible({"x1": 4.5, "x2": 4}) is True
        # 2.0*4.5 + (-1.0)*4 = 5.0 != -5.0, so a flipped rhs would reject it
        # 2.0*0.5 + (-1.0)*6 = -5.0 != 5.0 -> infeasible
        assert 2.0 * 0.5 + (-1.0) * 6 == -5.0
        assert loaded.search_space.is_feasible({"x1": 0.5, "x2": 6}) is False

    def test_all_negative_inequality_is_live_as_a_lower_bound(self, tmp_path):
        s = OptimizationSession()
        s.add_variable("x1", "real", bounds=(0.0, 10.0))
        s.add_variable("x2", "integer", bounds=(0, 8))
        s.add_input_constraint("inequality", {"x1": -1.0, "x2": -1.0}, rhs=-3.0,
                               name="floor")

        loaded, _ = _round_trip(s, tmp_path)

        # -1.0*5.0 + -1.0*2 = -7.0 <= -3.0 -> feasible (x1+x2 = 7 >= 3)
        assert -1.0 * 5.0 + -1.0 * 2 == -7.0
        assert loaded.search_space.is_feasible({"x1": 5.0, "x2": 2}) is True
        # -1.0*0.5 + -1.0*1 = -1.5 > -3.0 -> infeasible (x1+x2 = 1.5 < 3)
        assert -1.0 * 0.5 + -1.0 * 1 == -1.5
        assert loaded.search_space.is_feasible({"x1": 0.5, "x2": 1}) is False

    def test_all_restored_constraints_bind_at_once(self, tmp_path):
        """A point each single constraint allows but the set together rejects."""
        s = _mixed_session()
        _register_mixed_constraints(s)

        loaded, _ = _round_trip(s, tmp_path)
        space = loaded.search_space

        # budget:  1.0*4.0 + 2.0*2 =  8.0 <= 12.0  ok
        # balance: 1.0*4.0 - 1.0*4.0 = 0.0 == 0.0  ok
        # floor:  -1.0*4.0 - 1.0*2  = -6.0 <= -3.0 ok
        # constraint_0: 1.0*4.0 = 4.0 <= 6.0       ok
        ok = {"x1": 4.0, "x2": 2, "x3": 4.0, "c1": "A", "ctx": 1.0}
        assert space.is_feasible(ok) is True

        # Only 'balance' is violated: 1.0*4.0 - 1.0*6.0 = -2.0 != 0.0
        only_balance_bad = {"x1": 4.0, "x2": 2, "x3": 6.0, "c1": "A", "ctx": 1.0}
        assert 1.0 * 4.0 + (-1.0) * 6.0 == -2.0
        assert space.is_feasible(only_balance_bad) is False

        # Only 'floor' is violated: -1.0*0.0 - 1.0*0 = 0.0 > -3.0
        only_floor_bad = {"x1": 0.0, "x2": 0, "x3": 0.0, "c1": "A", "ctx": 1.0}
        assert -1.0 * 0.0 + -1.0 * 0 == 0.0
        assert space.is_feasible(only_floor_bad) is False

    def test_botorch_conversion_sees_the_restored_constraints(self, tmp_path):
        """The acquisition path reads them through this, not filter_feasible."""
        s = _mixed_session()
        _register_mixed_constraints(s)

        loaded, _ = _round_trip(s, tmp_path)
        names = loaded.search_space.get_dimension_names()
        ineq, eq = loaded.search_space.to_botorch_constraints(names)

        assert ineq is not None and len(ineq) == 3
        assert eq is not None and len(eq) == 1
        # BoTorch equality keeps ALchemist's sign convention as-is.
        _idx, coeffs, rhs = eq[0]
        assert sorted(float(c) for c in coeffs) == [-1.0, 1.0]
        assert rhs == 0.0

    def test_a_design_from_the_reloaded_session_honors_the_constraint(self, tmp_path):
        """The user-visible symptom: DoE on a reloaded session.

        Every returned point is checked with literal arithmetic. The companion
        assertion below proves the check discriminates: the same seed on an
        unconstrained session produces points this bound would reject, so a
        dropped constraint fails this test rather than sliding through.
        """
        s = OptimizationSession()
        s.add_variable("x1", "real", bounds=(0.0, 10.0))
        s.add_variable("x2", "real", bounds=(0.0, 10.0))
        s.add_input_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=6.0,
                               name="cap")

        loaded, _ = _round_trip(s, tmp_path)
        design = loaded.generate_initial_design(n_points=8, method="lhs",
                                                random_seed=1234)

        assert len(design) == 8
        for point in design:
            lhs = 1.0 * float(point["x1"]) + 1.0 * float(point["x2"])
            assert lhs <= 6.0 + 1e-3, (point, lhs)

        # Discrimination: without the constraint, this seed leaves the region.
        free = OptimizationSession()
        free.add_variable("x1", "real", bounds=(0.0, 10.0))
        free.add_variable("x2", "real", bounds=(0.0, 10.0))
        unconstrained = free.generate_initial_design(n_points=8, method="lhs",
                                                     random_seed=1234)
        assert any(
            1.0 * float(p["x1"]) + 1.0 * float(p["x2"]) > 6.0 + 1e-3
            for p in unconstrained
        ), "seed chosen badly: the unconstrained design is already feasible"


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestPreExistingFilesStillLoad:
    """Session files at 1.1.0 with no ``constraints`` key exist on disk.

    Every session saved before this fix is exactly that -- the writer never
    emitted the key -- so refusing them, or failing on them, is not on the
    table. They restore an empty constraint list, which is what they have
    always done.
    """

    def _file_without_the_key(self, tmp_path):
        s = _mixed_session()
        path = tmp_path / "legacy.json"
        s.save_session(str(path))
        data = json.loads(path.read_text())
        del data["search_space"]["constraints"]
        assert "constraints" not in data["search_space"]
        assert data["version"] == "1.1.0"
        path.write_text(json.dumps(data, indent=2))
        return path

    def test_a_1_1_0_file_with_no_constraints_key_loads(self, tmp_path):
        path = self._file_without_the_key(tmp_path)

        loaded = OptimizationSession.load_session(str(path), retrain_on_load=False)

        assert loaded.search_space.get_constraints() == []
        assert [v["name"] for v in loaded.search_space.variables] == [
            "x1", "x2", "x3", "c1", "ctx"
        ]

    def test_a_constraint_less_file_leaves_every_point_feasible(self, tmp_path):
        path = self._file_without_the_key(tmp_path)
        loaded = OptimizationSession.load_session(str(path), retrain_on_load=False)

        assert loaded.search_space.is_feasible(
            {"x1": 10.0, "x2": 8, "x3": 6.0, "c1": "B", "ctx": 0.0}
        ) is True

    def test_constraints_can_be_registered_on_a_reloaded_legacy_file(self, tmp_path):
        path = self._file_without_the_key(tmp_path)
        loaded = OptimizationSession.load_session(str(path), retrain_on_load=False)

        loaded.add_input_constraint("inequality", {"x1": 1.0}, rhs=4.0)

        assert [c["name"] for c in loaded.search_space.get_constraints()] == [
            "constraint_0"
        ]

    def test_the_version_string_is_unchanged(self, tmp_path):
        """No bump: the presence of the key is the discriminator, not this.

        Nothing branches on the version beyond a ``startswith('1.')`` warning,
        both directions already load without consulting it, and a second
        marker for the same fact is a second thing that can drift from it.
        """
        s = _mixed_session()
        _register_mixed_constraints(s)
        path = tmp_path / "session.json"
        s.save_session(str(path))

        assert json.loads(path.read_text())["version"] == "1.1.0"

    def test_a_constraint_naming_a_removed_variable_still_loads(self, tmp_path):
        """Why the loader assigns rather than re-registering.

        ``remove_variable`` documents that it leaves referencing constraints
        alone, and ``DELETE /variables/{name}`` calls it, so a session can hold
        -- and therefore save -- a constraint naming a variable that is gone.
        ``add_constraint`` rejects exactly that, so a re-registering loader
        would turn a reachable session into an unopenable file.
        """
        s = _mixed_session()
        s.add_input_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=9.0,
                               name="dangling")
        s.search_space.remove_variable("x2")
        assert "x2" in s.search_space.get_constraints()[0]["coefficients"]

        loaded, _ = _round_trip(s, tmp_path)

        assert loaded.search_space.get_constraints()[0]["name"] == "dangling"
        assert [v["name"] for v in loaded.search_space.variables] == [
            "x1", "x3", "c1", "ctx"
        ]

    def test_re_registering_that_dangling_constraint_would_have_failed(self):
        """The premise of the test above, executed rather than asserted."""
        s = _mixed_session()
        s.search_space.remove_variable("x2")
        with pytest.raises(ValueError, match="not found in search space"):
            s.add_input_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=9.0)


class TestAHandEditedConstraintIsRefusedAtTheLoadBoundary:
    """The ``constraints`` key is new, so upload is a new way to install one.

    ``load_session`` assigned the file's list across with no shape check at
    all, which made ``POST /sessions/upload`` the only constraint-bearing path
    on the branch with no validation -- ``/variables/load`` validates the
    identical structure completely. The escalation was the repair route: a
    single entry with no ``name`` key made ``DELETE /constraints/{name}``
    raise ``KeyError`` while building its match list over *every* registered
    constraint, so the session was a dead end through the API.

    Shape and re-registration are separable. These pin the shape half; the
    class above pins that the reference half is still not re-run.
    """

    def _saved(self, tmp_path, constraints, filename="edited.json"):
        s = _mixed_session()
        path = tmp_path / filename
        s.save_session(str(path))
        data = json.loads(path.read_text())
        data["search_space"]["constraints"] = constraints
        path.write_text(json.dumps(data))
        return path

    @pytest.mark.parametrize("entry,fragment", [
        ({"type": "equality", "coefficients": {"x3": -1.0}, "rhs": None,
          "name": "c_null"}, "rhs must be a finite number"),
        ({"type": "inequality", "coefficients": {"x1": 1.0}, "rhs": 1e999,
          "name": "c_inf"}, "rhs must be finite"),
        ({"type": "at_most", "coefficients": {"x2": -2.0}, "rhs": -4.0,
          "name": "c_type"}, "constraint_type must be one of"),
        ({"type": "inequality", "coefficients": {"x1": None}, "rhs": 7.0,
          "name": "c_coeff"}, "coefficient for 'x1' must be a finite number"),
        ({"type": "equality", "coefficients": "x1", "rhs": 0.5,
          "name": "c_str"}, "must be a mapping"),
    ])
    def test_the_file_no_longer_loads(self, tmp_path, entry, fragment):
        path = self._saved(tmp_path, [entry])
        with pytest.raises(ValueError) as exc:
            OptimizationSession.load_session(str(path), retrain_on_load=False)
        detail = str(exc.value)
        assert fragment in detail, detail
        assert entry["name"] in detail

    def test_a_nameless_entry_loads_addressably_rather_than_being_refused(self, tmp_path):
        """The entry that broke the repair route. ``name`` is optional on every
        other entry path, so it is filled, not rejected -- and the filled name
        is what ``DELETE /constraints/{name}`` needs to exist at all.
        """
        path = self._saved(tmp_path, [
            {"type": "inequality", "coefficients": {"x1": 1.0, "x2": 2.0},
             "rhs": 11.0},
            {"type": "equality", "coefficients": {"x3": -0.5}, "rhs": -1.0,
             "name": "explicit"},
        ])
        loaded = OptimizationSession.load_session(str(path), retrain_on_load=False)
        names = [c["name"] for c in loaded.search_space.get_constraints()]
        assert names == ["constraint_0", "explicit"]
        # Every entry answers to c["name"] -- the expression that used to raise
        # KeyError across the whole list.
        assert all("name" in c for c in loaded.search_space.constraints)

    def test_the_surviving_constraints_are_still_live(self, tmp_path):
        """A filled name must not be all that changed: the constraint has to
        still bind. An install that merely echoed would pass a name check.
        """
        import pandas as pd
        path = self._saved(tmp_path, [
            {"type": "inequality", "coefficients": {"x1": 1.0, "x2": 1.0},
             "rhs": 5.0},
        ])
        loaded = OptimizationSession.load_session(str(path), retrain_on_load=False)
        mask = loaded.search_space.filter_feasible(pd.DataFrame([
            {"x1": 1.0, "x2": 1, "x3": 0.0, "c1": "A", "ctx": 0.0},
            {"x1": 9.0, "x2": 8, "x3": 0.0, "c1": "A", "ctx": 0.0},
        ]))
        assert list(mask) == [True, False]

    def test_one_bad_entry_refuses_the_file_rather_than_half_loading_it(self, tmp_path):
        path = self._saved(tmp_path, [
            {"type": "inequality", "coefficients": {"x1": 1.0}, "rhs": 3.0,
             "name": "good"},
            {"type": "equality", "coefficients": {"x3": -1.0}, "rhs": None,
             "name": "bad"},
        ])
        with pytest.raises(ValueError) as exc:
            OptimizationSession.load_session(str(path), retrain_on_load=False)
        assert "constraints[1]" in str(exc.value)
        assert "bad" in str(exc.value)
