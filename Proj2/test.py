from deepy.nn import Linear, Sequential, Relu, Tanh
from deepy.utils import graph_repr, accuracy
from deepy.optim import SGD
from deepy.loss import MSE
from deepy.tensor import Variable
from data import gen_train_test, transform_target
from tqdm import tqdm

train_input, train_target, test_input, test_target = gen_train_test()

# define the network
l1 = Linear(2, 25)
l2 = Linear(25, 25)
l3 = Linear(25, 2)
model = Sequential([l1, l2, l3])

print(graph_repr(model, train_input[0]))

# change the label to be -1 and 1 because the output of tanh is [-1,1]
train_target, test_target = (transform_target(train_target), transform_target(test_target))

optimizer = SGD(model.param(), lr=0.1 / train_input.shape[0], weight_decay=0.01)
criterion = MSE()

for i in tqdm(range(1000)):
    optimizer.zero_grad()
    
    # we train with batches of the size of the dataset
    x = Variable(train_input)
    out = model(x)
    loss = criterion(out, train_target)
    loss.backward()
    optimizer.step()


pred = model(test_input)
print(f"Accuracy on test set: {accuracy(pred.data, test_target.argmax(1))}")

pred = model(train_input)
print(f"Accuracy on train set: {accuracy(pred.data, train_target.argmax(1))}")
