import numpy as np

X = np.array([[1.6,0.6],
             [1.9,0.4],
             [1.4,0.1],
             [2.4,0.4],
             [1.8,.27],
             [3.5,1.1],
             [3.2,0.8],
             [3.9,0.7],
             [4.4,0.8],
             [3.3,1.1]]) # training inputs

X = np.hstack((np.ones((len(X),1)),X))
#X.shape = (100,4)
target = np.array([-1,-1,-1,-1,-1,1,1,1,1,1])


#prediction of examples
def predict(X,weights):
    p = np.dot(X,weights)
    return p

#calculating the objective function
def objective_function(target,predicted):
    E = 0.5 * np.sum((target-predicted)**2)
    return E

#updating weights with gradient descent
def update_weights(X,weights,target,predicted,learning_rate):
    # finding the gradients 
    gradients = -np.dot(X.T,(target-predicted))
    # updating weights with gradient descent formula
    new_weights = weights - (learning_rate * gradients)
    return new_weights

def decision_function(predicted):
    return np.array([-1 if i<0 else 1 for i in predicted])

def perceptron(X,target,learning_rate,max_iteration):
    n,m = X.shape # no of examples and number of attributes to the data
    weights= np.random.uniform(-1,1,size = (m)) # initialize random weights 
    iteration = 0
    while iteration < max_iteration:
        predicted = predict(X,weights) #prediction of the output 
        E = objective_function(target,predicted) # calculating the objective function
        # updateing the weights by using gradient descent
        predicted_output = decision_function(predicted)
        print(sum(predicted_output==target))
        weights = update_weights(X,weights,target,predicted,learning_rate)
        
        iteration = iteration + 1
        print(iteration,E)
    return predicted_output,weights

predicted_output,converged_weights = perceptron(X,target,0.01,10000)
        


