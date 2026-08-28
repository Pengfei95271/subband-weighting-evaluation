"""
FBCNet as a baseline (Mane et al., 2021).

Why this baseline and not another: FBCNet is filter bank -> spatial convolution
-> variance layer, which is the same processing chain the manuscript's method
operates on. It is also small (a few thousand parameters), so it occupies the
same lightweight, interpretable niche the manuscript claims. Omitting it invites
the obvious question.

The architecture, following the paper:
  input   (n_trials, 1, n_channels, n_times, n_bands)  -- reshaped internally
  1. spatial convolution, m filters per band, depthwise over channels
  2. variance layer: log-variance over temporal strides
  3. linear classifier

Requires torch. If torch is absent the wrapper raises on fit(), so the rest of
the pipeline still imports.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted


def _require_torch():
    try:
        import torch
        return torch
    except ImportError as e:
        raise ImportError(
            "FBCNet needs PyTorch.  pip install torch --index-url "
            "https://download.pytorch.org/whl/cpu") from e


# --------------------------------------------------------------------------
# layers
# --------------------------------------------------------------------------

def _build_modules():
    """Defined inside a function so the module imports without torch."""
    torch = _require_torch()
    import torch.nn as nn

    class VarLayer(nn.Module):
        """Log-variance over the temporal dimension, in `n_strides` windows."""

        def __init__(self, n_strides):
            super().__init__()
            self.n_strides = n_strides

        def forward(self, x):
            # x: (B, C, 1, T)
            b, c, _, t = x.shape
            w = t // self.n_strides
            x = x[..., : w * self.n_strides]
            x = x.reshape(b, c, 1, self.n_strides, w)
            v = x.var(dim=-1, unbiased=True).clamp_min(1e-8)
            return torch.log(v).reshape(b, -1)

    class LinearWithConstraint(nn.Linear):
        """Max-norm constrained linear layer, as in the reference model."""

        def __init__(self, *a, max_norm=0.5, **kw):
            self.max_norm = max_norm
            super().__init__(*a, **kw)

        def forward(self, x):
            with torch.no_grad():
                self.weight.data = torch.renorm(
                    self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
            return super().forward(x)

    class Conv2dWithConstraint(nn.Conv2d):
        def __init__(self, *a, max_norm=2.0, **kw):
            self.max_norm = max_norm
            super().__init__(*a, **kw)

        def forward(self, x):
            with torch.no_grad():
                self.weight.data = torch.renorm(
                    self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
            return super().forward(x)

    class FBCNetModule(nn.Module):
        def __init__(self, n_channels, n_bands, n_classes, m=32, n_strides=4):
            super().__init__()
            self.n_bands = n_bands
            self.m = m
            self.n_strides = n_strides
            # depthwise spatial conv: one group per band, m filters each
            self.scb = nn.Sequential(
                Conv2dWithConstraint(n_bands, m * n_bands, (n_channels, 1),
                                     groups=n_bands, padding=0, max_norm=2.0),
                nn.BatchNorm2d(m * n_bands),
                nn.SiLU(),
            )
            self.var = VarLayer(n_strides)
            self.head = nn.Sequential(
                nn.BatchNorm1d(m * n_bands * n_strides),
                nn.Dropout(0.5),
                LinearWithConstraint(m * n_bands * n_strides, n_classes,
                                     max_norm=0.5),
            )

        def forward(self, x):
            # x: (B, n_bands, n_channels, T)
            x = self.scb(x)            # (B, m*n_bands, 1, T)
            x = self.var(x)            # (B, m*n_bands*n_strides)
            return self.head(x)

    return FBCNetModule


# --------------------------------------------------------------------------
# sklearn wrapper
# --------------------------------------------------------------------------

class FBCNet(BaseEstimator, ClassifierMixin):
    """FBCNet with a scikit-learn interface.

    Expects the same 4-D filter-bank input as the rest of the pipeline:
    (n_trials, n_channels, n_times, n_bands). Transposed internally.

    Parameters follow the reference implementation. `m=32` and `n_strides=4`
    are the published defaults; keep them fixed across subjects rather than
    tuning per subject, so the comparison stays honest.

    Early stopping needs room. On IV-2a a fold leaves roughly 184 training
    trials after the validation split, and patience=20 stops the run before
    convergence: holdout accuracy on S2 was 0.655 at patience=20 against 0.707
    at patience=50, with no further change at 100. Defaults are therefore
    max_epochs=300 and patience=50, chosen once on one subject and then held
    fixed for every subject and fold. Under-training a baseline is a way of
    winning an argument that does not survive review.
    """

    def __init__(self, m=32, n_strides=4, lr=1e-3, batch_size=32,
                 max_epochs=300, patience=50, weight_decay=0.0,
                 device="cpu", random_state=0, verbose=False):
        self.m = m
        self.n_strides = n_strides
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.weight_decay = weight_decay
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _to_nchw(X):
        """(n_trials, n_ch, n_times, n_bands) -> (n_trials, n_bands, n_ch, n_times)"""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 4:
            raise ValueError("FBCNet expects 4-D filter-bank input, got %s"
                             % (X.shape,))
        return np.transpose(X, (0, 3, 1, 2))

    def _normalise(self, X, fit=False):
        """Per-band z-score, statistics from the training set only."""
        if fit:
            self.mu_ = X.mean(axis=(0, 2, 3), keepdims=True)
            self.sd_ = X.std(axis=(0, 2, 3), keepdims=True) + 1e-8
        return (X - self.mu_) / self.sd_

    # -- api ---------------------------------------------------------------

    def fit(self, X, y):
        torch = _require_torch()
        import torch.nn as nn

        Module = _build_modules()
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X = self._normalise(self._to_nchw(X), fit=True)
        self.le_ = LabelEncoder().fit(y)
        yy = self.le_.transform(y)
        self.classes_ = self.le_.classes_

        n_bands, n_ch = X.shape[1], X.shape[2]
        dev = torch.device(self.device)
        self.model_ = Module(n_ch, n_bands, len(self.classes_),
                             self.m, self.n_strides).to(dev)

        # hold out 20% of the training fold for early stopping
        rng = np.random.default_rng(self.random_state)
        idx = rng.permutation(len(yy))
        n_val = max(int(0.2 * len(yy)), len(self.classes_) * 2)
        vi, ti = idx[:n_val], idx[n_val:]

        Xt = torch.tensor(X[ti], device=dev)
        yt = torch.tensor(yy[ti], dtype=torch.long, device=dev)
        Xv = torch.tensor(X[vi], device=dev)
        yv = torch.tensor(yy[vi], dtype=torch.long, device=dev)

        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        lossf = nn.CrossEntropyLoss()

        best, best_state, bad = np.inf, None, 0
        for epoch in range(self.max_epochs):
            self.model_.train()
            perm = torch.randperm(len(yt), device=dev)
            for s in range(0, len(yt), self.batch_size):
                b = perm[s:s + self.batch_size]
                if len(b) < 2:            # BatchNorm needs >1 sample
                    continue
                opt.zero_grad()
                loss = lossf(self.model_(Xt[b]), yt[b])
                loss.backward()
                opt.step()

            self.model_.eval()
            with torch.no_grad():
                vl = lossf(self.model_(Xv), yv).item()
            if vl < best - 1e-5:
                best, bad = vl, 0
                best_state = {k: v.detach().clone()
                              for k, v in self.model_.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
            if self.verbose and epoch % 20 == 0:
                print("    epoch %3d  val loss %.4f" % (epoch, vl))

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.eval()
        self.n_params_ = sum(p.numel() for p in self.model_.parameters())
        return self

    def decision_function(self, X):
        torch = _require_torch()
        check_is_fitted(self, "model_")
        X = self._normalise(self._to_nchw(X))
        with torch.no_grad():
            out = self.model_(torch.tensor(X, device=torch.device(self.device)))
        return out.cpu().numpy()

    def predict_proba(self, X):
        z = self.decision_function(X)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.le_.inverse_transform(self.decision_function(X).argmax(1))
