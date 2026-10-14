"""D1/D2 wiring: a refused sync stops every action that was about to follow it.

The desktop window cannot be built headless, so each method under test is
called unbound on a stand-in ``self`` carrying only what the method touches.
That exercises the real method bodies (the guard, the dialog, the early
returns) but not Tk itself.
"""
import types
from unittest import mock

import pandas as pd
import pytest

pytest.importorskip("customtkinter")

from alchemist_core import OptimizationSession
from alchemist_core.data.experiment_manager import ExperimentManager
from alchemist_core.data.search_space import SearchSpace
from ui import gpr_panel, ui as ui_mod
from ui import variables_setup
from ui.ui import ALchemistApp


def _web_session():
    """What a web-app / work-queue session looks like once desktop loads it."""
    s = OptimizationSession()
    s.add_variable("T", "real", bounds=(100.0, 300.0))
    s.add_variable("feed", "real", bounds=(0.0, 1.0), role="observed")
    s.experiment_manager.df = pd.DataFrame({
        "T": [150.0, 200.0], "feed": [0.2, 0.3], "Output": [0.1, 0.2],
        "ProvenanceId": ["a", "b"]})
    return s


def _app(session, exp_df):
    """Stand-in for ALchemistApp, as desktop load leaves it (manager IS the
    session's search space, exp_df is a copy of the session's table)."""
    app = types.SimpleNamespace()
    app.session = session
    app.search_space_manager = session.search_space
    app.experiment_manager = session.experiment_manager
    app.exp_df = exp_df
    for name in ("_check_sync_safe", "_sync_data_to_session"):
        setattr(app, name, types.MethodType(getattr(ALchemistApp, name), app))
    return app


@pytest.fixture
def dialog():
    with mock.patch.object(ui_mod.tk, "messagebox", create=True) as mb:
        yield mb


def test_refused_sync_returns_false_shows_one_dialog_and_changes_nothing(dialog):
    s = _web_session()
    before_space = s.search_space
    before_df = s.experiment_manager.df.copy()
    app = _app(s, s.experiment_manager.df.copy())

    assert app._sync_data_to_session() is False

    assert dialog.showerror.call_count == 1
    title, body = dialog.showerror.call_args.args
    assert title == "Can't sync this session"
    assert "feed" in body and "web app" in body and "ProvenanceId" in body
    assert s.search_space is before_space
    assert [v["name"] for v in s.search_space.variables] == ["T", "feed"]
    pd.testing.assert_frame_equal(s.experiment_manager.df, before_df)


def test_desktop_native_session_syncs(dialog):
    """A session built entirely in the desktop app, with every column the
    desktop writes, is never refused."""
    s = OptimizationSession()
    manager = SearchSpace()
    manager.from_dict([
        {"name": "T", "type": "Real", "min": 100.0, "max": 300.0},
        {"name": "cat", "type": "Categorical", "values": ["a", "b"]},
    ])
    app = types.SimpleNamespace(
        session=s, search_space_manager=manager,
        experiment_manager=ExperimentManager())
    app.experiment_manager.target_columns = ["Output"]
    app.exp_df = pd.DataFrame({
        "T": [150.0, 200.0], "cat": ["a", "b"], "Output": [0.1, 0.2],
        "Noise": [1e-6, 1e-6], "Iteration": [0, 1],
        "Reason": ["Initial Design", "Manual"]})
    for name in ("_check_sync_safe", "_sync_data_to_session"):
        setattr(app, name, types.MethodType(getattr(ALchemistApp, name), app))

    assert app._sync_data_to_session() is True
    dialog.showerror.assert_not_called()
    assert len(s.experiment_manager.df) == 2
    assert [v["name"] for v in s.search_space.variables] == ["T", "cat"]


def _refusing_app():
    app = types.SimpleNamespace()
    app._sync_data_to_session = lambda: False
    app.session = mock.MagicMock()
    app.current_session_file = "/tmp/x.json"
    return app


def test_save_session_cmd_saves_nothing_when_refused():
    app = _refusing_app()
    ALchemistApp.save_session_cmd(app)
    app.session.save_session.assert_not_called()


def test_save_session_as_opens_no_file_dialog_when_refused():
    app = _refusing_app()
    with mock.patch.object(ui_mod.filedialog, "asksaveasfilename") as ask:
        ALchemistApp.save_session_as(app)
    ask.assert_not_called()
    app.session.save_session.assert_not_called()


def test_audit_log_writes_nothing_when_refused():
    app = _refusing_app()
    assert ALchemistApp.log_optimization_to_audit(app) is False
    app.session.lock_data.assert_not_called()


def test_quit_does_not_close_on_a_refused_save():
    app = _refusing_app()
    app.session.audit_log.entries = [object()]
    app.destroy = mock.Mock()
    with mock.patch.object(ui_mod.tk, "messagebox", create=True) as mb:
        mb.askyesnocancel.return_value = True
        ALchemistApp._quit(app)
    app.destroy.assert_not_called()
    app.session.save_session.assert_not_called()


def test_training_does_not_start_when_refused():
    panel = types.SimpleNamespace(main_app=types.SimpleNamespace(
        _sync_data_to_session=lambda: False))
    with mock.patch.object(gpr_panel.ctk, "CTkToplevel") as window, \
            mock.patch.object(gpr_panel.threading, "Thread") as thread:
        gpr_panel.GaussianProcessPanel.train_model_threaded(panel)
    window.assert_not_called()
    thread.assert_not_called()


def test_variable_dialog_save_leaves_session_alone_when_refused():
    master = types.SimpleNamespace(
        _check_sync_safe=lambda: False, session=OptimizationSession())
    before = master.session.search_space
    dialog_self = types.SimpleNamespace(
        master=master, destroy=mock.Mock(),
        get_data=lambda: [{"name": "T", "type": "Real", "min": 0.0, "max": 1.0}])
    variables_setup.SpaceSetupWindow.save_and_close(dialog_self)
    assert not hasattr(master, "variable_space_data")
    assert master.session.search_space is before
    dialog_self.destroy.assert_not_called()


# --- non-refusal outcomes of the sync -------------------------------------

def _native_app(session, exp_df):
    """A desktop-built session: manager separate from the session's space."""
    manager = SearchSpace()
    manager.from_dict([
        {"name": "T", "type": "Real", "min": 100.0, "max": 300.0},
        {"name": "P", "type": "Real", "min": 1.0, "max": 10.0},
    ])
    app = types.SimpleNamespace(
        session=session, search_space_manager=manager,
        experiment_manager=ExperimentManager(), exp_df=exp_df)
    app.experiment_manager.target_columns = ["Output"]
    for name in ("_check_sync_safe", "_sync_data_to_session", "save_session_as"):
        setattr(app, name, types.MethodType(getattr(ALchemistApp, name), app))
    return app


def _template_table():
    """generate_template / a design CSV: inputs filled, every Output empty."""
    return pd.DataFrame({"T": [110.0, 220.0], "P": [2.0, 4.0], "Output": [None, None],
                         "Iteration": [0, 0], "Reason": ["Initial Design"] * 2})


def test_all_unmeasured_table_is_not_a_refusal(dialog):
    s = OptimizationSession()
    app = _native_app(s, _template_table())
    assert app._sync_data_to_session() is True
    dialog.showerror.assert_not_called()
    assert [v["name"] for v in s.search_space.variables] == ["T", "P"]


def test_save_as_reaches_the_file_dialog_with_an_all_unmeasured_table(dialog, tmp_path):
    s = OptimizationSession()
    app = _native_app(s, _template_table())
    target = str(tmp_path / "template.json")
    with mock.patch.object(ui_mod.filedialog, "asksaveasfilename", return_value=target) as ask:
        assert app.save_session_as() is True
    ask.assert_called_once()
    assert app.current_session_file == target
    dialog.showerror.assert_not_called()


def test_unexpected_sync_failure_tells_the_user_and_returns_false(dialog):
    s = OptimizationSession()
    app = _native_app(s, _template_table())
    with mock.patch.object(s, "add_variable", side_effect=RuntimeError("boom")):
        assert app._sync_data_to_session() is False
    assert dialog.showerror.call_count == 1
    title, body = dialog.showerror.call_args.args
    assert "boom" in body and "nothing was saved" in body.lower()


def test_save_as_returns_false_on_cancel(dialog):
    s = OptimizationSession()
    app = _native_app(s, _template_table())
    with mock.patch.object(ui_mod.filedialog, "asksaveasfilename", return_value=""):
        assert app.save_session_as() is False


def _quit_app(save_as_result):
    app = types.SimpleNamespace()
    app.session = mock.MagicMock()
    app.session.audit_log.entries = [object()]
    app.current_session_file = None
    app.maybe_prompt_create_session = lambda: True
    app.save_session_as = mock.Mock(return_value=save_as_result)
    app.destroy = mock.Mock()
    return app


@pytest.mark.parametrize("saved,closes", [(True, True), (False, False)])
def test_quit_without_a_file_closes_only_if_save_as_saved(saved, closes):
    app = _quit_app(saved)
    with mock.patch.object(ui_mod.tk, "messagebox", create=True) as mb:
        mb.askyesnocancel.return_value = True
        ALchemistApp._quit(app)
    app.save_session_as.assert_called_once()
    assert app.destroy.called is closes
