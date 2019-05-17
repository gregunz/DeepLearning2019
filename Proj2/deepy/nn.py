import math

import torch

from deepy.tensor import Variable, Parameter


class Module(object):
    """Basis class for all Neural Networks of the library"""

    def __init__(self, **kwargs):
        self.name = kwargs.get('name', 'Module')

    def forward(self, *inputs):
        """
        Define the forward pass for the module.
        """
        raise NotImplementedError()

    def parameters(self):
        """
        Function that return all the trainable Variables of the network.

        Return:
            params: List
                The trainable parameters of the Module
        """
        params = []
        for elem in self.__dict__.values():
            if isinstance(elem, Module):
                params += elem.parameters()
            if isinstance(elem, Parameter):
                params.append(elem)
        return params

    def __call__(self, *inputs):
        return self.forward(*inputs)


class Function(object):
    """
    Base class for backward pass of all module.
    Those class have for purpose to retain enough context in order to
    compute the backward pass from only the gradient of there output.
    """

    def inputs(self):
        """
        List of all the inputs of that function.
        This is useful in order to draw the backward graph (for visualisation purpose).
        """
        raise NotImplementedError()


class TanhBackward(Function):
    def __init__(self, x):
        self.x = x

    def _dtanh(self, x):
        """
        Derivative of the tanh function with respect to the input x.
        """
        return 4 * (x.exp() + x.mul(-1).exp()).pow(-2)

    def inputs(self):
        return [self.x]

    def __call__(self, dl):
        dl = dl * self._dtanh(self.x.data)
        if self.x.requires_grad:
            self.x.grad_fn(dl)

    def __repr__(self):
        return "Tanh()"


class Tanh(Module):
    def forward(self, x):
        out = torch.tanh(x.data)
        out = Variable(out, requires_grad=x.requires_grad, is_leaf=False)
        out.grad_fn = TanhBackward(x)
        return out

    def parameters(self):
        return []


class ReluBackward(Function):
    def __init__(self, x):
        self.x = x

    def _drelu(self, x):
        return (x > 0).type(torch.float)

    def inputs(self):
        return [self.x]

    def __call__(self, dl):
        dl = dl * self._drelu(self.x.data)
        if self.x.requires_grad:
            self.x.grad_fn(dl)

    def __repr__(self):
        return "Relu()"


class Relu(Module):
    def forward(self, x):
        out = torch.relu(x.data)
        out = Variable(out, requires_grad=x.requires_grad, is_leaf=False)
        out.grad_fn = ReluBackward(x)
        return out


class LinearBackward(Function):
    def __init__(self, x, w, b):
        self.x = x
        self.w = w
        self.b = b
        self.batch_size = x.data.size(0)

    def inputs(self):
        return [self.x]

    def backward(self, dl_ds):
        dl_dx = (self.w.data @ dl_ds.t()).t()
        self.w.grad.add_(self.x.data.permute(1, 0).view(-1, self.batch_size) @ dl_ds.view(self.batch_size, -1))
        if self.b:
            self.b.grad.add_(dl_ds.sum(0))
        return dl_dx

    def __call__(self, dl):
        dl = self.backward(dl)
        if self.x.requires_grad:
            self.x.grad_fn(dl)

    def __repr__(self):
        return f"Linear({self.w.data.shape[0]}, {self.w.data.shape[1]})"


class Linear(Module):
    """
    Define a fully connected layer
    """

    def __init__(self, in_features, out_features, bias=True, weight_init='uniform', bias_init='uniform', **kwargs):
        """
        Params:
            in_features: int
                Number of feature as inputs
            out_features: int
                Number of feature to outputs
            bias: bool
                When set to true in addtion to the weight, a bias is learned.
            weight_init: {'uniform', 'normal'}
                Distribution to use for the weight initialization
            bias_init: {'uniform', 'normal'}
                Distribution to use for the bias initialization
        """
        super().__init__(**kwargs)
        data = torch.empty(in_features, out_features).normal_()

        stdv = 1. / math.sqrt(data.size(1))

        if weight_init == 'uniform':
            data.uniform_(-stdv, stdv)

        self.w = Parameter(data,
                           requires_grad=True)
        if bias:
            bias_data = torch.empty(out_features).normal_()
            if bias_init == 'uniform':
                bias_data.uniform_(-stdv, stdv)
            self.b = Parameter(bias_data, requires_grad=True)
        else:
            self.b = None

    def forward(self, x):
        self.x = x
        out = x.data @ self.w.data
        if self.b:
            out.add_(self.b.data)
        out = Variable(out, requires_grad=self.w.requires_grad or x.requires_grad,
                       is_leaf=False)
        out.grad_fn = LinearBackward(x, self.w, self.b)
        return out


class Sequential(Module):
    """
    Combines multiples module in a sequential manner.
    The modules are chain one after the other and call with the output of the previous one as input.
    """

    def __init__(self, elems):
        """
        Params:
            elems: List
                List of Modules to combines.
                The order of the element in the list defines the order of calls to the modules.

        """
        self.elems = elems

    def forward(self, x):
        out = x
        for elem in self.elems:
            out = elem(out)
        return out

    def parameters(self):
        p = []
        for elem in self.elems:
            p += elem.parameters()
        return p
