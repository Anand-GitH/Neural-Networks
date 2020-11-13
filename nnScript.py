import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return 1/(1+np.exp(-z))


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples. 
    tcount1=0
    tcount2=0
    
    for mkey in mat:
        if "train" in mkey:
            if tcount1==0:
                tdataset=mat.get(mkey)
                tlabel=np.full((tdataset.shape[0],1),mkey[-1])
                tcount1+=1
            else:
                data=mat.get(mkey)
                tdataset=np.vstack((tdataset,data))
                tlabel=np.vstack((tlabel,np.full((data.shape[0],1),mkey[-1])))
                
        elif "test" in mkey:
            if tcount2==0:
                testdat=mat.get(mkey)
                testlbl=np.full((testdat.shape[0],1),mkey[-1])
                tcount2+=1
            else:
                data=mat.get(mkey)
                testdat=np.vstack((testdat,data))
                testlbl=np.vstack((testlbl,np.full((data.shape[0],1),mkey[-1])))
    
    indx=np.random.choice(tdataset.shape[0], tdataset.shape[0], replace=False)
    
    train_data=tdataset[indx[0:50000],:]
    train_label=tlabel[indx[0:50000],:]
    
    validation_data=tdataset[indx[50000:60000],:]
    validation_label=tlabel[indx[50000:60000],:]
    
    indx=np.random.choice(testdat.shape[0], testdat.shape[0], replace=False)
    
    test_data=testdat[indx,:]
    test_label=testlbl[indx,:]

    # Feature selection # Eliminate the features which does not add variations in dataset
    elim_cols=[]
    
    for i in range(tdataset.shape[1]):
        unique,count=np.unique(tdataset[:,i],return_counts=True)
        if len(unique)==1 and len(count)==1 and unique[0]==0:
            elim_cols.append(i)
    
    for i in range(len(elim_cols)):
        train_data      = np.delete(train_data,elim_cols[i]-i,1)
        validation_data = np.delete(validation_data,elim_cols[i]-i,1)
        test_data       = np.delete(test_data,elim_cols[i]-i,1)
    

    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    #Start of Feed Forward Propogation
    
    training_data=np.hstack((training_data,np.ones((training_data.shape[0],1)))) #adding bias to the input layer
    
    #output at first hidden layer
    aj=train_data @ w1.transpose()
    
    #Sigmoid activation function at hidden layer
    zj=sigmoid(aj)
    
    zj=np.hstack((zj,np.ones((zj.shape[0],1)))) #adding bias to the hidden layers
    
    #Output layer
    bl=zj @ w2.transpose()
    
    #Sigmoid Activation function at output layer
    ol=sigmoid(bl)
    
    #End of Feed Forward
    
    #We have 10 classes - And based on output only one class will be assigned which has higher sigmoid value
    #Processing label of training data
    yl = np.zeros((training_label.shape[0],n_class))
    
    for erow in range(training_label.shape[0]):
        yl[erow][int(training_label[erow])]=1
        
    
    #Error function as negative log likelihood
    error=np.sum( (yl @ np.log(ol)) + ((1-yl) @ np.log(1-ol)))/training_label.shape[0]
    error=np.negative(error)
    
    
    #Gradient Descent for weights
    #wrt w2 - hidden layer weights 
    deltal = ol - yl
    grad_w2= np.dot(deltal.transpose(),zj)
    
    #wrt w1 - input layer weights
    zprod    = (1 - zj) * zj
    zdeltaw2 = ((np.dot(deltal,w2))*zprod)
    grad_w1  = np.dot(zdeltaw2.transpose(),training_data)
    grad_w1  = grad_w1[0:n_hidden,:]
    
    #Regularization - Lambda to control the value of weights to reduce overfitting
    sqrdw1=np.sum(np.square(w1))
    sqrdw2=np.sum(np.square(w2))
    
    regpart=lambdaval*(sqrdw1+sqrdw2)/(2*train_data.shape[0])
    
    ##Error + regularization
    obj_val=error+regpart

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)

    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    labels = np.array([])
   
    #input layer
    data=np.hstack((data,np.ones((data.shape[0],1)))) #adding bias to the data
    a=data @ w1.transpose()
                
    #hidden layer
    z=sigmoid(a)
    z=np.hstack((z,np.ones((z.shape[0],1)))) #adding bias to the hidden layer
    
    #Output layer
    b=z @ w2.transpose()
    o=sigmoid(b)

    #Output labels
    labels=np.argmax(o,axis=1)
    
    return labels

"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 0

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
