
import numpy as np

X = np.array([[0.402262,	0.352496],
              [0.867487,	0.0575684],
              [0.0113909,	0.579944],
              [0.591981,	0.0773841],
              [0.925751,	0.525091],
              [0.72213,	   	0.979145],
              [0.649659,	0.2901],
              [0.0496755,	0.418382],
              [0.875639,	0.389704],
              [0.462654,	0.773322]])

X = np.hstack((np.ones((len(X),1)),X))

target = np.array([ 11.7694,2.8541,17.1778,6.8166,4.0622,9.5500,7.6245,15.8547,3.0675,12.0985])


learning_rate = 0.0001

def predict(X,weights):
    #prediction of a example
    p = np.dot(X,weights)
    return p   

def objective_function(actual,predicted):
    error = 0.5 * np.sum((actual - predicted)**2)
    return error

def update_weights(X_j,weights,actual,predicted,learning_rate):
    #calculate the derivative of objective function
    gradients = -(X_j*(actual-predicted))
    #taking a smaller step to the direction of gradients 
    new_weights = weights - (learning_rate * gradients)
    return new_weights


def Linear_regresssion(X,target,learning_rate,max_iterations):
    n,m = X.shape #number of roww and columns of input data
    # initialise with random weights of size m (m = number of columns)
    weights = np.random.uniform(-1,1,size=(m))
    predicted = np.zeros(n)
    for i in range(max_iterations):
        for j in range(n):
            #predicting output with weights
            predicted[j] = predict(X[j],weights)
            #updating weights with batch gradient descent
            weights = update_weights(X[j],weights,target[j],predicted[j],learning_rate)
        #objectve function
        error = objective_function(target,predicted)
        print(error)
    #predicted_output = [0 if i<=0.5 else 1 for i in predicted]
    return predicted
    
predicted_output = Linear_regresssion(X,target,0.01,5000)  
print("MSE = ",np.mean((predicted_output-target)**2))
