
import numpy as np

X = np.array([[0.402262,	0.352496],
              [0.867487,	0.0575684],
              [0.0113909,	0.579944],
              [0.591981,	0.0773841],
              [0.925751,	0.525091],
              [0.72213,	    	0.979145],
              [0.649659,	0.2901],
              [0.0496755,	0.418382],
              [0.875639,	0.389704],
              [0.462654,	0.773322]])



X = np.hstack((np.ones((len(X),1)),X))

target = np.array([ 11.7694,2.8541,17.1778,6.8166,4.0622,9.5500,7.6245,15.8547,3.0675,12.0985])


def predict(X,weights):
    #prediction of all the examples
    p = np.dot(X,weights)
    return p   

def objective_function(actual,predicted):
    error = 0.5 * np.sum((actual - predicted)**2)
    return error

def update_weights(X,weights,actual,predicted,learning_rate):
    #calculate the derivative of objective function
    gradients = -np.dot(X.T,(target-predicted))
    #taking a smaller step to the direction of gradients 
    new_weights = weights - (learning_rate * gradients)
    return new_weights

#stopping criterion
def within_tolerance(old_error,new_error,epsilon):
    return abs(old_error-new_error) <epsilon

def Linear_regresssion(X,target,learning_rate,max_iterations,epsilon):
    n,m = X.shape #number of roww and columns of input data
    # initialise with random weights of size m (m = number of columns)
    weights = np.random.uniform(-1,1,size=(m))
    iteration = 0
    old_error = 0
    close_enough = False 
    while iteration < max_iterations and not close_enough:
        print(iteration)
        #predicting output with weights
        predicted = predict(X,weights)
        #objectve function
        new_error = objective_function(target,predicted)

        print('Objective function at ', iteration , 'th iteration is ',new_error)
        
        close_enough = within_tolerance(old_error,new_error,epsilon)
        old_error = new_error
        #updating weights with batch gradient descent
        weights = update_weights(X,weights,target,predicted,learning_rate)
        iteration = iteration + 1
    return predicted
    
predicted_output = Linear_regresssion(X,target,0.01,1000,1e-4)  
print("MSE = ",np.mean((predicted_output-target)**2))