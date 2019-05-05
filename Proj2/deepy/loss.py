from deepy.nn import Module, Function
from deepy.tensor import Variable

class MSEBackward(Function):
    def __init__(self, v, t):
        self.v = v
        self.t = t
    
    def dloss(self, v, t):
        return 2 * (v.data - t.data)
    
    def inputs(self):
        return [self.v, self.t]
    
    def __call__(self, dl):
        # TODO check if dl is other than 1.0 what append
        dl = self.dloss(self.v.data, self.t.data)

        if self.v.requires_grad:
            self.v.grad_fn(dl)


class MSE(Module):
    """
    Mean Square Error loss function.
    """
    def __init__(self):
        super(MSE, self).__init__()
    
    def loss(self, v, t):
        x = (t.data - v.data)
        x = x.unsqueeze(1)
        x = x.t() @ x
        out = Variable(x, requires_grad=t.requires_grad or v.requires_grad, is_leaf=False)
        out.grad_fn = MSEBackward(v, t) 
        return out
    
    def forward(self, x, targets):
        return self.loss(x, targets)
    
    def __call__(self, inputs, targets):
        return self.forward(inputs, targets)
        
