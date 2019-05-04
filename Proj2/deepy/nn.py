import torch
from deepy.tensor import Variable

class Module(object):
    
    def forward(self, *input):
        raise NotImplementedError()

    def backward(self, *grad_wrt_output):
        raise NotImplementedError()

    def param(self):
        return []

class TanhBackward:
    
    def __init__(self, ctx):
        self.ctx = ctx

    def _dtanh(self, x):
        return 4 * (x.exp() + x.mul(-1).exp()).pow(-2)
    
    def __call__(self, dl):
        dl = dl * self._dtanh(self.ctx.data)
        if self.ctx.requires_grad:
            self.ctx.grad_fn(dl)

class Tanh(Module):


    def __init__(self):
        super(Tanh).__init__()
        
    
    def forward(self, x):
        self.x = x
        out = torch.tanh(x.data)
        out = Variable(out, requires_grad=x.requires_grad, is_leaf=False)
        out.grad_fn = TanhBackward(x)
        return out
    
    
    def backward(self, dl_dx):
        dl_ds = dl_dx * self._dtanh(self.x)
        return dl_ds
    
    def __call__(self, x):
        return self.forward(x)

    def param(self):
        return []


class Function:
    def __init__(self, *ctx):
        pass



class LinearBackward(Function):
    def __init__(self, x, w, b):
        self.x = x
        self.w = w
        self.b = b

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


class Linear(Module):
    
    def __init__(self, in_features, out_features, bias=True):
        if bias:
            self.b = Variable(torch.empty(out_features).normal_(0, 1e-6), requires_grad=True)
        self.w = Variable(torch.empty(in_features, out_features).normal_(0, 1e-6), requires_grad=True)
        
    #def reset_grad():  # TODO remove 
    #    self.w.grad.zero_()
    #    if self.b:
    #        self.b.grad.zero_()

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

    def __call__(self, x):
        return self.forward(x)

    
    def param(self):
        return [self.w, self.b]



class Sequential:
    """This class is not a real module as it doesn't necesarly require a backward function"""

    def __init__(self, elems):
        self.elems = elems
    
    def forward(self, x):
        out = x
        for elem in self.elems:
            out = elem(out)
        
        return out

    def __call__(self, x):
        return self.forward(x)
    
    #def backward(self, dl_dx):
    #    dl = dl_dx
    #    for elem in reversed(self.elems):
    #        dl = elem.backward(dl)
    
    def param(self):
        p = []
        for elem in self.elems:
            p += elem.param()
        return p
