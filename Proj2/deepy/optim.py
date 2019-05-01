class SGD:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr
    
    def zero_grad(self):
        for var in self.params:
            var.zero_grad()
    
    def step(self):
        for var in self.params:
            var.update(self.lr)