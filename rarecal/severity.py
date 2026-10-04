"""Severity functionals.

A severity functional R maps a sample to a scalar "how extreme is it" score.
It is fitted once on reference data and then applied, unchanged, to real and
generated samples alike.
"""
from __future__ import annotations

import numpy as np

__all__ = ["WhitenedRadius", "MaxPatchRadius"]


class WhitenedRadius:
    """Whitened radius on the top-K principal components of the reference data.

    R(x) = || Lambda^{-1/2} V^T (x - m) ||_2

    For images, pass flattened pixels (or any feature vector). With pixels this
    measures pixel-space atypicality; with network embeddings it measures
    atypicality in that representation.
    """

    def __init__(self, n_components: int = 50):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None    # (d, K)
        self.eigvals_ = None       # (K,)

    def fit(self, X: np.ndarray) -> "WhitenedRadius":
        X = np.asarray(X, dtype=np.float64).reshape(len(X), -1)
        self.mean_ = X.mean(0)
        C = np.cov(X - self.mean_, rowvar=False, bias=True)
        lam, V = np.linalg.eigh(C)
        order = np.argsort(lam)[::-1][: self.n_components]
        self.eigvals_ = lam[order]
        self.components_ = V[:, order]
        return self

    def whiten(self, X: np.ndarray) -> np.ndarray:
        """Whitened coordinates w (n, K)."""
        X = np.asarray(X, dtype=np.float64).reshape(len(X), -1)
        return (X - self.mean_) @ (self.components_ / np.sqrt(self.eigvals_))

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return np.linalg.norm(self.whiten(X), axis=1)

    # ---- closed-form transport: rescale the whitened coordinates -------------
    def transport(self, X: np.ndarray, r_target: np.ndarray,
                  box: tuple[float, float] | None = None,
                  n_iter: int = 120, n_damped: int = 100) -> np.ndarray:
        """Move each sample to severity r_target, keeping its direction w/||w||.

        Without a box the move is exact in one step. With a box (e.g. (0, 255)
        for 8-bit pixels) we alternate between the severity level set and the
        box, with damped steps first, and report whatever residual remains.
        """
        X = np.asarray(X, dtype=np.float64)
        shape = X.shape
        X = X.reshape(len(X), -1).copy()
        r_target = np.asarray(r_target, dtype=np.float64)
        up = self.components_ * np.sqrt(self.eigvals_)          # (d, K)
        steps = 1 if box is None else n_iter
        for it in range(steps):
            w = self.whiten(X)
            r = np.linalg.norm(w, axis=1)
            fac = r_target / np.maximum(r, 1e-12)
            if box is not None and it < n_damped:
                fac = np.sqrt(fac)
            X += ((fac - 1.0)[:, None] * w) @ up.T
            if box is not None:
                np.clip(X, box[0], box[1], out=X)
        return X.reshape(shape)


class MaxPatchRadius:
    """Worst-patch severity: the maximum of per-patch whitened radii.

    Images are (n, H, W, C); each non-overlapping patch position gets its own
    whitening basis with `n_components` components.
    """

    def __init__(self, patch: int = 8, n_components: int = 32):
        self.patch = patch
        self.n_components = n_components
        self.radii_ = []

    def _patches(self, X):
        n, H, W, C = X.shape
        P = self.patch
        out = []
        for i in range(0, H, P):
            for j in range(0, W, P):
                out.append(X[:, i:i + P, j:j + P, :].reshape(n, -1))
        return out

    def fit(self, X: np.ndarray) -> "MaxPatchRadius":
        X = np.asarray(X, dtype=np.float64)
        self.radii_ = [WhitenedRadius(self.n_components).fit(p) for p in self._patches(X)]
        return self

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        per = np.stack([r(p) for r, p in zip(self.radii_, self._patches(X))], axis=1)
        return per.max(1)
