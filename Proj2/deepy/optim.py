class SGD:
    def __init__(self, params, lr, weight_decay=0.0):
        self.params = params
        self.lr = lr
        self.weight_decay = weight_decay * 2
    
    def zero_grad(self):
        for var in self.params:
            var.grad.zero_()
    
    def step(self):
        for var in self.params:
            grad = var.grad
            
            if self.weight_decay > 0:
                # L2 regularization
                grad.add_(2 * self.weight_decay * var.data)
            var.data = var.data - self.lr * grad