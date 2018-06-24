# Simple Neural Network for binary classification

Here you find an implementation of a **simple neural network for binary classification**. The network comprises a **single hidden layer** of size 100.

### What is this?
This script makes up a simple neural network

-	with a dynamic **input dimensionality** (for visualization reasons limited to 2)
-	with a **single hidden layer of size 100** (but feel free to change to another value)
-	and with a **single output neuron** for a classification of the input.

The network can be trained for a given number of periods and the results will be ***nicely visualized***. 

It uses backpropagation as a weight optimization technique and the hinge-loss as an error measurement.

The implementation of the net is not meant to be a fast one, instead it should give a better understanding of what is happening during the training.

### How to start it?

Just start the Python-script `python NeuralNet.py` and see the results. You can specify the training data file in the beginning of the script, as well as specify some parameters.

### Prerequisites

You need to have **[numpy](http://www.numpy.org/)** and **[matplotlib](https://matplotlib.org/)** installed, nothing more. The code was tested with Python 3.6.4.

## Data for the training
The data set is a numerical 2D set (points) with a third numerical label. The input dimension is not limited to two dimensions, however, for a better visualization at the end, the data must be 2D. One simple dataset is provided, namely

- [data_moon.txt](/data/data_moon.txt)

which consists of 500 2D-points arranged in two swirl-like, noisy patterns. If you like to use your own set of data points, please **note**: the class labels always need to be 1 and -1. Just stick to the structure of [data_moon.txt](/data/data_moon.txt).

A random 80/20 split up of the training data provides the test data, the accuracy for the test data is also provided after training.

## Visualization
The visualizations at the end show the development of the hinge loss during learning for each iteration and the classification in a contour map respectively. The drawn datapoints are both training and test data points, the probability for the class is shown as contour lines. 
The data points are colored according to their *ground-truth* class labels.


## Authors
* **Marcus Rottschäfer** - [GitHub profile](https://github.com/shukali)

If you find any errors, have any ideas or questions regarding the code, feel free to contact me!


## License
This project is licensed under the MIT License. See the [LICENSE.md](LICENSE) file for details.
