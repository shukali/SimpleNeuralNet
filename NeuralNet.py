import numpy as np
import matplotlib.pyplot as plt

'''
Expects data in the form: (see also data_moon.txt)

...                         ...                         ...
8.558757427169911836e-01    3.152883206649254588e-01    -1.000000000000000000e+00
1.910346694104992693e+00    1.986821953518217720e-01    1.000000000000000000e+00
...                         ...                         ...

where the last column contains the class labels for each row. The class labels need 
to be 1 and -1 and no more, as this is a binary classification neural network.

Parameters that can be set are, next to the data source:

num_iterations      The number of iterations to run the weight optimization for
size_hidden_layer   The size (number of neurons) of the single hidden layer
alpha               The learning parameter alpha, better leave about 0.05

'''
# PARAMETERS
num_iterations = 200 # number of iterations to run the gradient descent for
size_hidden_layer = 100
alpha = 0.05
data = np.loadtxt("data/data_moon.txt")
#
np.random.shuffle(data)
eighty_percent_index = int(round(len(data)*0.8))
data_training, data_test = data[:eighty_percent_index,:], data[eighty_percent_index:,:]
X_training, Y_training = data_training[:, :2], data_training[:, 2]
X_test, Y_test = data_test[:, :2], data_test[:, 2]
#
h = [X_training.shape[1], size_hidden_layer, 1] # height of each layer
W = [np.zeros((h[1], h[0])), np.zeros((h[2], h[1]))] # list of weight matrices for each layer
z = [np.zeros(h[0]), np.zeros(h[1]), np.zeros(h[2])] # list of input vectors of each layer
x = [np.zeros(h[0]), np.zeros(h[1]), np.zeros(h[2])] # list of output vectors of each layer

# init weights randomly
W[0] = (np.random.rand(h[1], h[0])*-2)+1 # [-1, 1]
W[1] = (np.random.rand(h[2], h[1])*-2)+1 # [-1, 1]

sigmoid = lambda z: 1 / (1 + np.power(np.e, -z))

def forward(x, w):
    '''Computes the forward propagation of the network. Returns an array of the size of the last layer,
       in our case an array of length 1.
    '''
    z[1] = np.dot(w[0], x[0]) # (100, 1)
    x[1] = sigmoid(z[1]) # (100, 1)
    z[2] = np.dot(W[1], x[1]) # (1)
    # no sigmoid for the output neuron
    #x[2] = sigmoid(z[2])
    return z[2]

def backward(delta_L_plus_1, x, w):
    '''Computes the backward propagation of the network. Returns an array of matrices where each matrix correlates
       to the gradient with respect to the weights w. 
    '''
    loss_gradients = [np.zeros(h[0]), np.zeros(h[1]), np.zeros(h[2])]
    loss_gradients[2] = delta_L_plus_1
    for l in list(reversed(range(len(w) ) ) ): # go through all layers last to first
        delta_l = np.dot(delta_L_plus_1, w[l])
        delta_l = np.multiply(delta_l, np.multiply(x[l], (1-x[l])).T) # delta_l is a transposed vector
        loss_gradients[l] = delta_l
        delta_L_plus_1 = delta_l # recursively compute the loss gradients
    
    # the loss gradients are used to compute the weight gradients
    weight_gradients = [np.zeros((h[1], h[0])), np.zeros((h[2], h[1]))]
    for l in list(reversed(range(len(w) ) ) ):     
        for i in range(w[l].shape[0]):
             for j in range(w[l].shape[1]):
                 # very slow way of setting the gradients, but makes it more clear
                 weight_gradients[l][i][j] = loss_gradients[l+1][i] * x[l][j] 
    return weight_gradients

def weightUpdate(w, gradients, alpha):
    '''Performs the optimization of the parameters (w) for the given gradients with gradient descent.'''
    for l in range(len(w)):
        w[l] = np.subtract(w[l], alpha*gradients[l])

def hinge_loss(f, y):
    '''Computes the hinge loss l(f, y) = max(0, 1-fy) between the predicted result f and the ground truth y.'''
    return np.amax([0, 1 - f*y])

def classifyDatapoint(x):
    '''Classifies the given datapoint x into a class probability [0, 1] based on the learned weights.'''
    x_full = [x, np.zeros(h[1]), np.zeros(h[2])]
    return sigmoid(forward(x_full, W))

def classifyDatapointMatrix(X):
    '''Classifies the given datapoint matrix X into an array class probability [0, 1] based on the learned weights.'''
    Y = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        Y[i] = sigmoid(forward([X[i], np.zeros(h[1]), np.zeros(h[2])], W))
    return Y

loss = np.zeros(num_iterations)
for i in range(num_iterations):
    x[0] = X_training[0]
    loss_sum = 0
    gradients_sum = [np.zeros((h[1], h[0])), np.zeros((h[2], h[1]))]
    for j in range(X_training.shape[0]):
        # go through all data points
        x[0] = X_training[j]
        f = forward(x, W) # forward propagation of x
        delta_l_plus_1 = np.array([-Y_training[j]]).T if (1 - Y_training[j]*f) > 0 else np.array([0]).T # loss gradient for last layer
        gradients = backward(delta_l_plus_1, x, W) # back propagation step
        for l in range(len(gradients)):
            gradients_sum[l] = np.add(gradients_sum[l], gradients[l])
        loss_sum += hinge_loss(f, Y_training[j]) # error between prediction and ground-truth

    # update weight with respect to computed gradients
    weightUpdate(W, gradients_sum, alpha)
    loss[i] = loss_sum
    print("{}. iteration, Hinge loss: {}".format(i+1, loss_sum))

# EVALUATING TEST DATA
Y_test_predicted = classifyDatapointMatrix(X_test)
y_colored_test = [1.0 if y > 0.5 else -1.0 for y in Y_test_predicted]
cnt_correct_labeled = 0.0
for i in range(Y_test.shape[0]):
    cnt_correct_labeled += 1.0 if Y_test[i] == y_colored_test[i] else 0.0
print("\n Accuracy on test data: {}".format(cnt_correct_labeled / len(X_test)))

# VISUALIZATION
point_range = np.arange(np.amin(X_training)-0.3, np.amax(X_training)+0.3, 0.1)
x1, x2 = np.meshgrid(point_range, point_range)
X_grid = np.column_stack([x1.flatten(), x2.flatten()])
Y_predicted = np.zeros(len(X_grid))
for i in range(X_grid.shape[0]):
    Y_predicted[i] = 0 if classifyDatapoint(np.array([X_grid[i][0], X_grid[i][1]])) < 0.0 else 1.0
    Y_predicted[i] = classifyDatapoint(np.array([X_grid[i][0], X_grid[i][1]]))

# draw colormap
colormap = plt.get_cmap('coolwarm')
contour = plt.contourf(point_range, point_range, Y_predicted.reshape(len(point_range), len(point_range)), 25, cmap=colormap, vmin=0, vmax=1) # plot contourmap
# draw scatter plot
y_colored_training = ['orange' if c == 1.0 else 'purple' for c in Y_training]
plt.scatter(X_training[:, 0], X_training[:, 1], c=y_colored_training, s=50, edgecolor="white", linewidth=1) # plot training data
y_colored_test = ['orange' if c == 1.0 else 'purple' for c in Y_test]
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_colored_test, s=50, edgecolor="white", linewidth=1) # plot test data
# draw loss chart
f, ax3 = plt.subplots(figsize=(8, 6))
ax3.plot(range(num_iterations), loss) # plot hinge loss development
ax3.set_title("Hinge-loss per iteration")
plt.show()