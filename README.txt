The code is based off of Michael Nielsen's python code at: 
https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network.py
which demonstrates feed forward neural networks and more. 

I rewrote the program in Java as a learning exercise to understand feed forward neural networks. 

Matrix.java provides methods for matrix operations.
Sigmoid.java calculates the sigmoid function and its derivative.
QuadraticCost.java calculates the means squared error and its derivative.
FFNN.java is a feed forward neural network using back propagation and stochastic gradient descent.
Main.java runs an example training a feed forward neural network to learn the 'and' function.