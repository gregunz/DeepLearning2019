class SGD:
    """
    Implementation of the vanilla Stochastic Gradient Descent algorithm.
    """
    def __init__(self, params, lr, weight_decay=0.0):
        """
        Params:
            params: List
                A list of Variable all that requires_grad and are leaf.
                The Variables present in that list will get there weight (data) updated at each step
            lr: flaot
                The learning rate
            weight_decay: float
                The penaly to add to the gradient for the L2 regularization.
            """
        self.params = params
        self.lr = lr
        self.weight_decay = weight_decay

    def zero_grad(self):
        """
        Reset the gradent of the tracked Variables to zero.
        """
        for var in self.params:
            var.grad.zero_()

    def step(self):
        """
        Perform a step of the SGD algorithm by updating the tracked weights
        """
        for var in self.params:
            grad = var.grad

            # L2 regularization
            if self.weight_decay > 0:
                grad.add_(2 * self.weight_decay * var.data)

            var.data = var.data - self.lr * grad