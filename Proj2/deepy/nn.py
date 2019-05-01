import torch

class Module(object):
    def forward(self, *input):
        raise NotImplementedError()

    def backward(self, *grad_wrt_output):
        raise NotImplementedError()

    def param(self):
        return []
    
class Variable:
    """Trainable variable."""
    def __init__(self, data, requires_grad=False):
        self.data = data
        self.grad = torch.empty_like(data)  # acumulated gradient

    def zero_grad(self):
        self.grad.zero_()
    
    def update(self, eta):
        self.data = self.data - eta * self.grad
        

class Tanh(Module):
    def forward(self, x):
        self.x = x
        return torch.tanh(x)
    
    def _dtanh(self, x):
        return 4 * (x.exp() + x.mul(-1).exp()).pow(-2)
    
    def backward(self, dl_dx):
        dl_ds = dl_dx * self._dtanh(self.x)
        return dl_ds
    
    def __call__(self, x):
        return self.forward(x)

    def param(self):
        return []

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        if bias:
            self.b = Variable(torch.empty(out_features).normal_(0, 1e-6))
        self.w = Variable(torch.empty(in_features, out_features).normal_(0, 1e-6))
        
        #self.dl_dw = torch.empty_like(self.W)
        #if bias:
        #    self.bias = torch.empty(out_features)
        #    self.bias_grad(torch.empty_like(self.bias))
        #    self.dl_db = torch.empty_like(self.bias)
    
    def reset_grad():
        self.w.grad.zero_()
        if self.b:
            self.b.grad.zero_()

    def forward(self, x):
        self.x = x
        out = x @ self.w.data
        if self.b:
            out += self.b.data
        return out

    def __call__(self, x):
        return self.forward(x)

    def backward(self, dl_ds):
        dl_dx = self.w.data @ dl_ds
        self.w.grad.add_(self.x.view(-1, 1) @ dl_ds.view(1, -1))
        if self.b:
            self.b.grad.add_(dl_ds)
        return dl_dx
    
    def param(self):
        return [self.w, self.b]

class Sequential:
    def __init__(self, elems):
        self.elems = elems
    
    def forward(self, x):
        out = x
        for elem in self.elems:
            out = elem(out)
        return out

    def __call__(self, x):
        return self.forward(x)
    
    def backward(self, dl_dx):
        dl = dl_dx
        for elem in reversed(self.elems):
            dl = elem.backward(dl)
    
    def param(self):
        p = []
        for elem in self.elems:
            p += elem.param()
        return p