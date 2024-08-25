import numpy as np
import random as ran
import matplotlib.pyplot as plt

x1train=(np.array([ran.uniform(-1, 1) for _ in range(2000)]).reshape(-1,1))
x2train=(np.array([ran.uniform(-1, 1) for _ in range(2000)]).reshape(-1,1))
x1test=(np.array([ran.uniform(-1, 1) for _ in range(100)]).reshape(-1,1))
x2test=(np.array([ran.uniform(-1, 1) for _ in range(100)]).reshape(-1,1))
Xtrain= np.column_stack((x1train,x2train))
Xtest=np.column_stack((x1test,x2test))
ytrain=(1-x1train)**2 + 100*((x2train-x1train**2)**2)
ytest=(1-x1test)**2 + 100*(x2test-x1test**2)**2

def relu(z):
	return np.maximum(0,z)
def relu_derivative(z):
	z[z<=0] = 0
	z[z>0] = 1
	return z

def forward_propagation(X):
	#input to hidden layer
	u = np.dot(X, weights_input_to_hidden) + biases_hidden
	z = relu(u)
	v = np.dot(z, weights_hidden_to_output) + biases_output
	yhat = v
	return  z,yhat
    
def backpropagation(X, Y, z, yhat):
	e =2*(yhat-Y)
	dw_output = np.dot(z.transpose(), e)
	db_output = np.sum(e, axis=0, keepdims=True)
	dz_hidden = np.dot(e, weights_hidden_to_output.T)*relu_derivative(z)
	dw_hidden = np.dot(X.T, dz_hidden)
	db_hidden = np.sum(dz_hidden, axis=0, keepdims=True)
	return dw_hidden, db_hidden, dw_output, db_output
    
    
def update_parameters(dw_hidden, db_hidden, dw_output, db_output, learning_rate):
	global weights_input_to_hidden, biases_hidden, weights_hidden_to_output, biases_output
	weights_input_to_hidden -= learning_rate*dw_hidden
	biases_hidden -= learning_rate*db_hidden
	weights_hidden_to_output -= learning_rate*dw_output
	biases_output -= learning_rate*db_output
    

def train(X, Y, epochs, learning_rate,epoch_print):
	for epoch in range(epochs):
		z, yhat = forward_propagation(X)
		# Backpropagation
		loss=np.mean((Y-yhat)**2)
		dw_hidden, db_hidden, dw_output, db_output = backpropagation(X, Y, z, yhat)
		# Update parameters
		update_parameters(dw_hidden, db_hidden, dw_output, db_output, learning_rate)
		if epoch % epoch_print == 0:
			print(f"Loss at epoch {epoch}: {loss}")


input_size = 2  # Number of features in the input
hidden_size = 10 # Number of neurons in the hidden layer
output_size = 1  # Number of neurons in the output layer
weights_input_to_hidden = np.random.randn(input_size, hidden_size) * 0.01
weights_hidden_to_output = np.random.randn(hidden_size, output_size) * 0.01
biases_hidden = np.random.randn(1, hidden_size)* 0.1
biases_output = np.random.randn(1, output_size)* 0.1

epochs=100000
learning_rate=0.000001
epoch_print=10000
train(Xtrain, ytrain, epochs, learning_rate,epoch_print)
xx,predtest=forward_propagation(Xtest)
xx,predtrain=forward_propagation(Xtrain)
plt.plot(predtrain,"o")
plt.plot(ytrain,"*")
plt.legend(['Predictions of Training data','Trainig Data'])
plt.figure()
plt.plot(predtest,"o")
plt.plot(ytest,"*")
plt.legend(['Predictions of Test data','Test Data'])
plt.show()

