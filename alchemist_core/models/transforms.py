"""Custom BoTorch input transforms for ALchemist."""

import torch
import pandas as pd
from typing import List, Tuple, Callable
from botorch.models.transforms.input import InputTransform
from torch.nn import Module


class DerivedFeatureTransform(InputTransform, Module):
    """
    BoTorch InputTransform that appends derived/computed features to the input tensor.

    Derived features are deterministic functions of base input variables, applied
    row-wise using caller-provided callables. Chain this *before* Normalize in
    _create_transforms() so Normalize sees the full N+K dimensional input.

    transform_on_train, transform_on_eval, and transform_on_fantasize are all
    True so the augmentation is applied consistently at every evaluation point.

    Args:
        derived_vars: List of (name, func, input_cols) tuples.
                      func signature: ``func(row: dict) -> float``
        base_var_names: Ordered list of base variable names used to build the
                        DataFrame column mapping before calling each func.
    """

    transform_on_train: bool = True
    transform_on_eval: bool = True
    transform_on_fantasize: bool = True
    is_one_to_many: bool = False

    def __init__(
        self,
        derived_vars: List[Tuple[str, Callable, List[str]]],
        base_var_names: List[str],
    ):
        Module.__init__(self)
        self.derived_vars = derived_vars
        self.base_var_names = list(base_var_names)
        self._is_trained = True  # Deterministic — no fitting required

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Append derived feature columns to X.

        Args:
            X: Input tensor of shape (..., n_base_vars).

        Returns:
            Augmented tensor of shape (..., n_base_vars + n_derived).
        """
        if not self.derived_vars:
            return X

        original_shape = X.shape
        X_2d = X.reshape(-1, X.shape[-1])
        df = pd.DataFrame(
            X_2d.detach().cpu().numpy(),
            columns=self.base_var_names,
        )

        new_cols = []
        for _name, func, _input_cols in self.derived_vars:
            col_vals = df.apply(lambda row: func(row.to_dict()), axis=1)
            new_cols.append(
                torch.tensor(col_vals.values, dtype=X.dtype, device=X.device).unsqueeze(-1)
            )

        X_aug = torch.cat([X_2d] + new_cols, dim=-1)
        new_shape = original_shape[:-1] + (X_aug.shape[-1],)
        return X_aug.reshape(new_shape)

    def untransform(self, X: torch.Tensor) -> torch.Tensor:
        """Strip derived feature columns, returning only base variable columns."""
        return X[..., : len(self.base_var_names)]
