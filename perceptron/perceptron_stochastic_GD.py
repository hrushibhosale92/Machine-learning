

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


def predict(X,weights):
    #prediction of all the examples
    p = np.dot(X,weights)
    return p   

def sign_function(net):
    return -1 if net<=0 else 1 

def objective_function(actual,predicted):
    E = 0.5 * np.sum((actual - predicted)**2)
    return E

def update_weights(X,weights,target,predicted,learning_rate):
    #calculate the derivative of objective function
    gradients = -np.dot(X.T,(target-predicted))
    #taking a smaller step to the direction of gradients 
    new_weights = weights - (learning_rate * gradients)
    return new_weights

#stopping criterion
def within_tolerance(old_error,new_error,epsilon):
    return abs(old_error-new_error) < epsilon


def perceptron(X,target,learning_rate,max_iterations,epsilon):
    n,m = X.shape #number of row and columns of input data
    # initialise with random weights of size m (m = number of columns)
    weights = np.random.uniform(-1,1,size=(m))
    net = np.zeros(n)
    predicted = np.zeros(n)
    iteration = 0
    old_error = 0
    close_enough = False 
    while iteration < max_iterations and not close_enough:   
        for j in range(n):
        #predicting output with weights
            net[j] = predict(X[j],weights)
            predicted[j] = sign_function(net[j])
            #updating weights with batch gradient descent
            weights = update_weights(X[j],weights,target[j],net[j],learning_rate)
        #objectve function
        new_error = objective_function(target,net)
        print('Objective function at ', iteration , 'th iteration is ',new_error)    
        print('accuracy : ',np.sum(predicted==target)/n)
        close_enough = within_tolerance(old_error,new_error,epsilon)
        old_error = new_error
        #updating weights with batch gradient descent
        iteration = iteration + 1  
    return predicted
    
predicted_output = perceptron(X,target,0.01,50,1e-6)  
print("Model performance: ",sum(predicted_output==target)/len(target))