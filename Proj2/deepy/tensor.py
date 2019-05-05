import torch


class Variable:
    """Trainable variable."""
    def __init__(self, data, requires_grad=False, is_leaf=True):
        self.data = data
        self.grad = torch.empty_like(data)  # acumulated gradient
        self.grad_fn = None  # Operation which created that variable
        self.requires_grad = requires_grad
        self.is_leaf = is_leaf

    def zero_grad(self):
        if not self.requires_grad:
            raise RuntimeError("This Tensor doesn't have gradient")
        self.grad.zero_()
    
    def update(self, eta):
        if not self.requires_grad:
            raise RuntimeError("This Tensor doesn't have gradient")
        self.data = self.data - eta * self.grad

    def backward(self):
        # check that data is a scalar or raise an exception
        self.grad = 1.0
        if self.grad_fn is None:
            raise RuntimeError("No grad_fn define for that tensor")

        # Compute dl
        self.grad_fn(1.0)
