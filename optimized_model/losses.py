import numpy as np
from .engine import Tensor

def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """
    logits  : Tensor of shape (N, num_classes) — raw, unnormalized scores
    targets : integer NumPy array of shape (N,) — class indices in [0, num_classes)

    Returns a scalar Tensor: mean negative log-likelihood over the batch.
    """
    N, num_classes = logits.shape
    targets = np.asarray(targets, dtype=np.int64)

    # Stable log-softmax via the log-sum-exp trick.
    shifted = logits.data - np.max(logits.data, axis=1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    log_probs = shifted - log_sum_exp                      # (N, num_classes)

    nll = -log_probs[np.arange(N), targets]                # (N,)
    loss_value = nll.mean()

    out = Tensor(loss_value, (logits,), "cross_entropy")

    def _backward():
        if logits.requires_grad:
            # d(mean NLL)/d(logits) = (softmax(logits) - one_hot(targets)) / N
            probs = np.exp(log_probs)                       # softmax(logits), stable
            grad = probs.copy()
            grad[np.arange(N), targets] -= 1.0
            grad /= N
            logits.grad += grad * out.grad   # out.grad is a scalar (seeded to 1.0 on .backward())
    out._backward = _backward
    return out


def accuracy(logits: Tensor, targets: np.ndarray) -> float:
    """Fraction of correct predictions — convenience metric, not differentiable."""
    preds = np.argmax(logits.data, axis=1)
    targets = np.asarray(targets)
    return float(np.mean(preds == targets))