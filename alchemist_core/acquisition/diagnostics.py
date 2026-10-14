"""How much the model knows at a suggested point.

The knowledge ratio is sqrt(posterior latent variance / prior latent variance)
at a point, noise excluded, both in the model's internal (transformed) output
scale. 0 means the data pins the function there; 1 means the model is at its
prior. A held or observed value far from every training row drives it to 1,
which is how a suggestion that only looks confident is caught.
"""
from typing import List

import numpy as np
import torch

KNOWLEDGE_WARNING_THRESHOLD = 0.9


def knowledge_ratio(model, X_encoded: np.ndarray) -> List[float]:
    """Ratio per row of ``X_encoded`` (encoded, model column order). Max over outputs."""
    gp = model.model
    members = list(gp.models) if hasattr(gp, "models") else [gp]
    X = torch.as_tensor(np.asarray(X_encoded, dtype=float), dtype=torch.double)
    per_output = []
    for m in members:
        was_training = m.training
        m.eval()
        with torch.no_grad():
            Xt = m.transform_inputs(X)
            posterior_var = m(Xt).variance
            prior_var = m.covar_module(Xt, diag=True)
            per_output.append((posterior_var / prior_var).clamp(min=0.0).sqrt())
        if was_training:
            m.train()
    return torch.stack(per_output).max(dim=0).values.reshape(-1).tolist()
