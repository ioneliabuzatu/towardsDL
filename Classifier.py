import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
# from lr_utils import load_dataset


# %matplotlib inline


# self dataset download need to be reshaped

with h5py.File('train_catvnoncat.h5', "r") as train_dataset:
    train_set_x = np.array(train_dataset["train_set_x"][:])
    train_set_y = np.array(train_dataset["train_set_y"][:])

with h5py.File('test_catvnoncat.h5', "r") as test_dataset:
    test_set_x = np.array(test_dataset["test_set_x"][:])
    test_set_y = np.array(test_dataset["test_set_y"][:])
    classes = np.array(test_dataset["list_classes"][:])

train_set_y = train_set_y.reshape((1, train_set_y.shape[0]))
test_set_y = test_set_y.reshape((1, test_set_y.shape[0]))
print("HERE, test_set_y shape: " + str(test_set_y.shape))
print("HERE, test_set_y sqeeze shape: " + str(np.squeeze(test_set_y).shape))



index = 25
plt.imshow(train_set_x[index])
plt.show()
print ("y = " + str(train_set_y[:,index]) + ", it's a '" + classes[np.squeeze(train_set_y[:,index])].decode("utf-8") +  "' picture.")

# get the sets size
m_train = train_set_y.shape[1]
print("# number of training examples is: " + str(m_train))
m_test = test_set_y.shape[1]
print("# number of test examples is: " + str(m_test))
num_px = train_set_x.shape[1]
print("# width or length of a training image is: " + str(num_px))

# flatten the images
train_set_x_flatten = train_set_x.reshape(train_set_x.shape[0], -1).T
test_set_x_flatten = test_set_x.reshape(test_set_x.shape[0], -1).T
print("train_set_x_flatten:" + str(train_set_x_flatten.shape))
print("test_set_x_flatten:" + str(test_set_x_flatten.shape))

train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

# compute sigmoid to make prediction on z
def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s
# maybe this is needed for lete rfor the prediction
def initialize_zeros(dim):
    w = np.zeros(shape = (dim,1))
    b = 0
    assert w.shape == (dim,1)
    assert (isinstance(b, float) or isinstance(b, int))
    return w,b

# now forward and backward for learning the parameters
def propagations(w, b, X, Y):
    """Arguments:
    w : weights, a numpy array of size (num_px * num_px * 3, 1)
    b : bias, a scalar
    X : data of size (num_px * num_px * 3, number of examples)
    Y : true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, nuber of examples)"""

    m = X.shape[1]
    a = sigmoid(np.dot(w.T,X)+b)
    cost = (-1/m) * np.sum(Y*np.log(a)+(1-Y)*(np.log(1-a)))

    # gradient descent
    dw = (1/m) * np.dot(X, (a - Y).T)
    db = (1/m) * np.sum(a - Y)

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    # tidy things up
    grads = {"dw": dw,
             "db": db}

    return grads, cost

w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
print(propagations(w,b,X,Y))


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    This function optimizes w and b by running a gradient descent algorithm

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters"""

    # $ \theta = \theta - \alpha \text{ } d\theta$, where $\alpha$ is the learning rate.
    Costs = []

    for i in range(num_iterations):

        grads,cost = propagations(w, b, X, Y)

        # get previous w and b
        db = grads['db']
        dw = grads['dw']

        # update the parameters w and b
        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            Costs.append(cost)

        if print_cost and i%100 == 0:
            print("something")


    grads = {'dw':dw,
             'db':db}

    params = {'w':w,
              'b':b}

    return params, grads, Costs


params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)
print(optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False))

def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    # convert a into 0 if activation <= 0.5 else convert into 1
    # get only the index 1, the number of examples
    m = X.shape[1]
    print("m is:" + str(m))

    # X = np.squeeze(X)
    # print("X shape is: {} %" .format(X.shape))

    # create a vector of 0 of lenght equal to m
    Y_prediction = np.zeros((1, m))

    # get the right dimention
    w = w.reshape(X.shape[0], 1)
    print("w is " + str(w))
    # make prediction
    A = sigmoid(np.dot(w.T, X)+b)
    print("A shape is: {} %" .format(A.shape))


    for parameter in range(X.shape[1]):
        Y_prediction[0,parameter] = 1 if A[0,parameter] > 0.5 else 0

    assert (Y_prediction.shape == (1, m))
    print("Y_prediction shape {} %" .format(Y_prediction.shape))
    return Y_prediction

print("predictions = " + str(predict(w, b, X)))


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Builds the logistic regression model by calling the function you've implemented previously

    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations

    Returns:
    d -- dictionary containing information about the model.

    Data description:
    Number of training examples: m_train = 209
    Number of testing examples: m_test = 50
    Height/Width of each image: num_px = 64
    Each image is of size: (64, 64, 3)
    train_set_x shape: (209, 64, 64, 3)
    train_set_y shape: (1, 209)
    test_set_x shape: (50, 64, 64, 3)
    test_set_y shape: (1, 50)
    """


    # initialize zeros
    w, b = initialize_zeros(X_train.shape[0])
    # propagations
    # grads, cost = propagations(w, b, X_train, Y_train)
    # optimize
    params, grads, Costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate)

    w = params["w"]
    b = params["b"]

    # predict
    train = predict(w,b, X_train)

    test = predict(w,b,X_test)
    # dict with all outputs missing -  "test_prediction":test,

    print("training set accuracy is: {} %".format(100 - np.mean(np.abs(train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(test - Y_test)) * 100))

    output = {"train_prediction":train, "test_prediction":test, "w":w, "b":b, "cost":Costs, "grads":grads, "num_iterations":num_iterations, "learning_rate":learning_rate}

    # return dict
    return output


model_output = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)
print(model_output)


# test the right predictions and wrong ones
index = 5
plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
plt.show()
index = 10
plt.imshow(test_set_x[:,5].reshape((num_px, num_px, 3)))
plt.show()

#  plot the cost function and the gradients
cost = np.squeeze(model_output["cost"])
print(len(cost))
print("cost of the entire model is {} %" .format(cost))

plt.plot(cost)
plt.title("Cost Function of the Model")
plt.show()




learning_rates = [0.01, 0.001, 0.0001]

all_models = {}

for rate in learning_rates:
    print("now the learning rate is: " + str(rate))
    all_models[str(rate)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=rate, print_cost=False)
print(model_output)


for i in all_models:
    plt.plot(np.squeeze(all_models[str(i)]["cost"]))
    # plt.savefig("all_learning_rates")
#plt.show()
# you cannot run simultaneously show() and savefig() unless you save the image first fig1 = plt.gcf()
plt.savefig("all_learning_rates")