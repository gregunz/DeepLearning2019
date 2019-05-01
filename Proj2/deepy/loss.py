from deepy.nn import Module

class MSE(Module):
    def __init__(self):
        super(MSE, self).__init__()
    
    def loss(self, v, t):
        x = (t - v)
        x = x.unsqueeze(1)
        return x.t() @ x
    
    def dloss(self, v, t):
        return 2 * (v - t)
    
    def backward(self):
        pass
        
    def forward(self, x, targets):
        return self.loss(x, targets), self.dloss(x, targets)
    
    def __call__(self, inputs, targets):
        return self.forward(inputs, targets)
        