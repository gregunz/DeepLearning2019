import torch


class Variable:
    """This class is a wrapper around a Tensor
    Params:
        data: Tensor
            Data of any type, it can be a weight of a network or in input to the network.
        requires_grad: bool
            Set to True in order to make the variable trainable (Update possible by backprop if its a leaf node)
        is_leaf: bool
            Leaf node of the computation graph have that property set to True.
    """

    def __init__(self, data, requires_grad=False, is_leaf=True):
        self.data = data
        if is_leaf:
            self.grad = torch.empty_like(data)  # acumulated gradient
        else:
            self.grad = None
        self.grad_fn = None  # Operation which created that variable
        self.requires_grad = requires_grad
        self.is_leaf = is_leaf

    def backward(self):
        # check that data is a scalar or raise an exception
        self.grad = 1.0
        if self.grad_fn is None:
            raise RuntimeError("No grad_fn define for that tensor")

        # Compute dl
        self.grad_fn(1.0)


class Parameter(Variable):
    """
    A special Variable that when used inside a deepy.nn.Module is automatically added to the parameter list of that module.
    """
    pass
