"""Unit tests for DerivedFeatureTransform."""
import pytest
import torch

from alchemist_core.models.transforms import DerivedFeatureTransform


def test_transform_appends_derived_columns():
    """transform() appends K derived columns to an N-dim input tensor."""
    derived_vars = [("sum_xy", lambda row: row["x"] + row["y"], ["x", "y"])]
    transform = DerivedFeatureTransform(derived_vars, base_var_names=["x", "y"])

    X = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    X_out = transform.transform(X)

    assert X_out.shape == (2, 3), f"Expected (2, 3), got {X_out.shape}"
    assert torch.allclose(X_out[:, 2], torch.tensor([3.0, 7.0], dtype=torch.float64))


def test_transform_preserves_base_columns():
    """Base columns are unchanged after transform."""
    derived_vars = [("prod", lambda row: row["a"] * row["b"], ["a", "b"])]
    transform = DerivedFeatureTransform(derived_vars, base_var_names=["a", "b"])

    X = torch.tensor([[2.0, 5.0]], dtype=torch.float64)
    X_out = transform.transform(X)

    assert torch.allclose(X_out[:, :2], X)
    assert torch.allclose(X_out[:, 2], torch.tensor([10.0], dtype=torch.float64))


def test_transform_multiple_derived_vars():
    """Multiple derived variables are appended in order."""
    derived_vars = [
        ("sum_xy", lambda row: row["x"] + row["y"], ["x", "y"]),
        ("diff_xy", lambda row: row["x"] - row["y"], ["x", "y"]),
    ]
    transform = DerivedFeatureTransform(derived_vars, base_var_names=["x", "y"])

    X = torch.tensor([[3.0, 1.0]], dtype=torch.float64)
    X_out = transform.transform(X)

    assert X_out.shape == (1, 4)
    assert torch.allclose(X_out[:, 2], torch.tensor([4.0], dtype=torch.float64))   # sum
    assert torch.allclose(X_out[:, 3], torch.tensor([2.0], dtype=torch.float64))   # diff


def test_untransform_strips_derived_columns():
    """untransform() returns only the first N (base) columns."""
    derived_vars = [("z", lambda row: row["x"] ** 2, ["x"])]
    transform = DerivedFeatureTransform(derived_vars, base_var_names=["x"])

    X_aug = torch.tensor([[2.0, 4.0], [3.0, 9.0]], dtype=torch.float64)
    X_base = transform.untransform(X_aug)

    assert X_base.shape == (2, 1)
    assert torch.allclose(X_base, torch.tensor([[2.0], [3.0]], dtype=torch.float64))


def test_transform_handles_batch_dimensions():
    """transform() works with 3D batch tensors (q-batch shape from BoTorch)."""
    derived_vars = [("s", lambda row: row["x"] + row["y"], ["x", "y"])]
    transform = DerivedFeatureTransform(derived_vars, base_var_names=["x", "y"])

    # Shape: (batch=2, q=3, d=2) — typical BoTorch q-batch
    X = torch.ones(2, 3, 2, dtype=torch.float64)
    X_out = transform.transform(X)

    assert X_out.shape == (2, 3, 3)


def test_is_trained_is_true():
    """DerivedFeatureTransform is always considered trained (no fitting needed)."""
    transform = DerivedFeatureTransform([], base_var_names=["x"])
    assert transform.is_trained is True


def test_empty_derived_vars_is_identity():
    """With no derived variables, transform() is an identity."""
    transform = DerivedFeatureTransform([], base_var_names=["x", "y"])
    X = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    assert torch.allclose(transform.transform(X), X)


def test_transform_raises_on_column_count_mismatch():
    """transform() raises a clear error when X columns don't match base_var_names."""
    derived_vars = [("z", lambda row: row["x"] ** 2, ["x"])]
    transform = DerivedFeatureTransform(derived_vars, base_var_names=["x"])

    X_wrong = torch.tensor([[1.0, 2.0]], dtype=torch.float64)  # 2 cols but 1 expected
    with pytest.raises(ValueError, match="base_var_names"):
        transform.transform(X_wrong)
