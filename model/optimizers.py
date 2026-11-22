import numpy as np

#NEED TO ADD CODE FOR HUFFMAN OPTIMIZATION

class Optimizer:
    """Base class — shared parameter storage and zero_grad."""

    def __init__(self, parameters):
        self.parameters = list(parameters)

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()

    def step(self):
        raise NotImplementedError

class Adam(Optimizer):
    """
    Adam (Adaptive Moment Estimation), Kingma & Ba (2014).

    Maintains exponential moving averages of the gradient (first moment,
    `m`) and the squared gradient (second moment, `v`), with bias
    correction for their initialization at zero:

        m <- beta1 * m + (1 - beta1) * grad
        v <- beta2 * v + (1 - beta2) * grad^2
        m_hat <- m / (1 - beta1^t)
        v_hat <- v / (1 - beta2^t)
        p.data <- p.data - lr * m_hat / (sqrt(v_hat) + eps)

    Adam adapts the effective learning rate per-parameter, which generally
    makes it a robust default for training CNNs without much learning-rate
    tuning.
    """

    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._m = [np.zeros_like(p.data) for p in self.parameters]
        self._v = [np.zeros_like(p.data) for p in self.parameters]
        self._t = 0

    def step(self):
        self._t += 1
        for p, m, v in zip(self.parameters, self._m, self._v):
            m *= self.beta1
            m += (1 - self.beta1) * p.grad
            v *= self.beta2
            v += (1 - self.beta2) * (p.grad ** 2)

            m_hat = m / (1 - self.beta1 ** self._t)
            v_hat = v / (1 - self.beta2 ** self._t)

            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)