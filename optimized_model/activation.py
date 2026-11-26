import numpy as np
from .engine import Tensor

def relu(x: Tensor) -> Tensor:
    """Rectified Linear Unit: max(0, x). Thin wrapper around Tensor.relu()."""
    return x.relu()

def softmax(x: Tensor, axis: int = -1) -> Tensor:
    """
    Softmax along `axis`, with the standard max-subtraction trick for
    numerical stability (prevents exp() overflow for large logits).

    Built entirely from existing differentiable Tensor ops (no new
    `_backward` needed here) — subtracting the max is implemented as a
    plain NumPy op on a *detached* constant, which is safe because
    softmax(x) == softmax(x - c) for any constant c, so subtracting the
    max doesn't change the true gradient, only improves numerics.
    """
    shifted_data = x.data - np.max(x.data, axis=axis, keepdims=True)
    shifted = Tensor(shifted_data, (x,), "softmax_shift")

    def _shift_backward():
        if x.requires_grad:
            x.grad += shifted.grad
    shifted._backward = _shift_backward

    exps = shifted.exp()
    return exps / exps.sum(axis=axis, keepdims=True)