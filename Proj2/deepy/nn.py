import torch
from deepy.tensor import Variable


class Module(object):
    
    def __init__(self, **kwargs):
        self.name = kwargs.get('name', 'Module')

    def forward(self, *inputs):
        raise NotImplementedError()

    def param(self):
        params = []
        for elem in self.__dict__.values():
            if isinstance(elem, Module):
                params += elem.param()
            if isinstance(elem, Variable):
                if elem.requires_grad and elem.is_leaf:
                    params.append(elem)
        return params

    def __call__(self, *inputs):
        return self.forward(*inputs)

    
class Function:
    def __init__(self, *ctx):
        pass

    def inputs(self):
        raise NotImplementedError()

class TanhBackward(Function):    
    def __init__(self, x):
        self.x = x

    def _dtanh(self, x):
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
    def __init__(self):
        super(Tanh).__init__()
         
    def forward(self, x):
        self.x = x
        out = torch.tanh(x.data)
        out = Variable(out, requires_grad=x.requires_grad, is_leaf=False)
        out.grad_fn = TanhBackward(x)
        return out

    def param(self):
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
        self.x = x  # WARNING this tensor should not appear in param
        out = torch.relu(x.data)
        out = Variable(out, requires_grad=x.requires_grad, is_leaf=False)
        out.grad_fn = ReluBackward(x)
        return out


class LinearBackward(Function):
    def __init__(self, x, w, b):
        self.x = x
        self.w = w
        self.b = b

    def inputs(self):
        return [self.x]

    def backward(self, dl_ds):
        dl_dx = self.w.data @ dl_ds
        self.w.grad.add_(self.x.data.view(-1, 1) @ dl_ds.view(1, -1))
        if self.b:
            self.b.grad.add_(dl_ds)
        return dl_dx

    def __call__(self, dl):
        dl = self.backward(dl)
        if self.x.requires_grad:
            self.x.grad_fn(dl)
    
    def __repr__(self):
        return f"Linear({self.w.data.shape[0]}, {self.w.data.shape[1]})"


class Linear(Module):
    
    def __init__(self, in_features, out_features, bias=True):
        #self.bias = bias
        if bias:
            self.b = Variable(torch.empty(out_features).normal_(0, 1e-6),
                              requires_grad=True)
        self.w = Variable(torch.empty(in_features, out_features).normal_(0, 1e-6),
                          requires_grad=True)

    def forward(self, x):
        self.x = x
        out = x.data @ self.w.data
        if self.b:
            out += self.b.data
        # TODO check with bias requires_grad
        out = Variable(out, requires_grad=self.w.requires_grad or x.requires_grad,
                is_leaf=False)
        out.grad_fn = LinearBackward(x, self.w, self.b)
        return out
    
    #def param(self):
    #    if self.bias:
    #        return [self.w, self.b]
    #    else:
    #        return [self.w]


class Sequential(Module):
    def __init__(self, elems):
        self.elems = elems
    
    def forward(self, x):
        out = x
        for elem in self.elems:
            out = elem(out)
        return out
    
    def param(self):
        p = []
        for elem in self.elems:
            p += elem.param()
        return p
