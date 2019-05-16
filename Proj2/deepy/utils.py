import torch


def graph_repr(model, X):
    """
    Return a string representation of the backward graph.
    """
    out = model(X)
    functions = []
    while not out.is_leaf:
        fn = out.grad_fn
        functions.append(fn)
        inputs = fn.inputs()
        if len(inputs) > 1:
            raise NotImplementedError("Graph with operator of more than 1 input can't be draw for now")
        out = inputs[0]

    # construction of the string
    s = "X => "
    for elem in reversed(functions):
        s += f"{elem} => "
    s += "y"
    return s


def accuracy(pred, target):
    return (pred.argmax(1) == target).type(torch.float).mean()
