"""
Fall 2023, 10-417/617
Assignment-2
Programming - CNN
TAs in charge: Jared Mejia, Kaiwen Geng

IMPORTANT:
    DO NOT change any function signatures but feel free to add instance variables and methods to the classes.

October 2023
"""

from re import L
import numpy as np
import copy
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
# import im2col_helper  # uncomment this line if you wish to make use of the im2col_helper.pyc file for experiments
# import imp

# im2col_helper = imp.load_compiled('im2col_helper', "/Users/gauravmolugu/Downloads/10617/F23__10417__Homework_2/programming/im2col_helper.pyc")

CLASS_IDS = {
 'cat': 0,
 'dog': 1,
 'car': 2,
 'bus': 3,
 'train': 4,
 'boat': 5,
 'ball': 6,
 'pizza': 7,
 'chair': 8,
 'table': 9
 }

IDS_CLASS = {
    0: 'cat',
    1: 'dog',
    2: 'car',
    3: 'bus',
    4: 'train',
    5: 'boat',
    6: 'ball',
    7: 'pizza',
    8: 'chair',
    9: 'table'
}
epsilon = 1e-5
softmax = lambda x: np.exp(x) / (np.sum(np.exp(x), axis = 1).reshape(-1,1)+epsilon)

def random_weight_init(input, output):
    b = np.sqrt(6)/np.sqrt(input+output)
    return np.random.uniform(-b, b, (input, output))

def convolution_weight_init(input_shape, filter_shape):
    """
    input_shape is a tuple: (channels, height, width)
    filter_shape is a tuple: (num of filters, filter height, filter width)
    weights shape (number of filters, number of input channels, filter height, filter width)
    """
    C, H, W = input_shape
    n, k_height, k_width = filter_shape
    b = np.sqrt(6)/np.sqrt((n+C)*k_height*k_width)

    return np.random.uniform(-b, b, (n, C, k_height, k_width))

def zeros_bias_init(outd):
    return np.zeros((outd, 1))

def im2col(X, k_height, k_width, padding=1, stride=1):
    '''
    Construct the im2col matrix of intput feature map X.
    X: 4D tensor of shape [N, C, H, W], input feature map
    k_height, k_width: height and width of convolution kernel
    return a 2D array of shape (C*k_height*k_width, H*W*N)
    The axes ordering need to be (C, k_height, k_width, H, W, N) here, while in
    reality it can be other ways if it weren't for autograding tests.

    Note: You must implement im2col yourself. If you use any functions from im2col_helper, you will lose 50
    points on this assignment.
    '''
    # #TODO
    # raise NotImplementedError

    N,C,H,W = X.shape
    p = padding
    X_padded = np.pad(X, ((0,0), (0,0), (p,p), (p,p)), mode='constant')
    H = H + 2*p
    W = W + 2*p

    out_H = (H - k_height)//stride + 1
    out_W = (W - k_width)//stride + 1

    shape = (C, k_height, k_width, out_H, out_W, N)
    strides = (H*W, W, 1, stride*W, stride, H*W*C)
    strides = X.itemsize*np.array(strides)

    X_stride = np.lib.stride_tricks.as_strided(X_padded, shape=shape, strides=strides)
    X_cols = np.ascontiguousarray(X_stride)

    X_cols.shape = (C*k_height*k_width, out_H*out_W*N)

    return X_cols

def im2col_bw(grad_X_col, X_shape, k_height, k_width, padding=1, stride=1):
    '''
    Map gradient w.r.t. im2col output back to the feature map.
    grad_X_col: a 2D array
    return X_grad as a 4D array in X_shape

    Note: You must implement im2col yourself. If you use any functions from im2col_helper, you will lose 50
    points on this assignment.
    '''
    # #TODO
    # raise NotImplementedError

    N,C,H_org,W_org = X_shape
    
    X = np.zeros((N, C, H_org, W_org), dtype=grad_X_col.dtype)
    
    p = padding
    H = H_org + 2*p
    W = W_org + 2*p

    out_H = (H - k_height)//stride + 1
    out_W = (W - k_width)//stride + 1
    X_bw = np.pad(X, ((0,0), (0,0), (p,p), (p,p)), mode='constant')

    col = 0
    
    for j in range(out_H):
        for i in range(out_W):
            for batch in range(N):
                X_bw[batch,:,stride*j:stride*j+k_height, stride*i:stride*i+k_width] += grad_X_col[:, col].reshape(-1,k_height,k_width)
                col+=1

    return X_bw[:,:, padding:H-padding, padding:W-padding]

class Transform:
    """
    This is the base class. You do not need to change anything.
    Read the comments in this class carefully.
    """
    def __init__(self):
        """
        Initialize any parameters
        """
        pass

    def forward(self, x):
        """
        x should be passed as column vectors
        """
        pass

    def backward(self, grad_wrt_out):
        """
        Note: we are not going to be accumulating gradients (where in hw1 we did)
        In each forward and backward pass, the gradients will be replaced.
        Therefore, there is no need to call on zero_grad().
        This is functionally the same as hw1 given that there is a step along the optimizer in each call of forward, backward, step
        """
        pass

    def update(self, learning_rate, momentum_coeff):
        """
        Apply gradients to update the parameters
        """
        pass

class ReLU(Transform):
    """
    Implement this class
    """
    def __init__(self):
        Transform.__init__(self)

    def forward(self, x):
        # #TODO
        # raise NotImplementedError
        self.x = x
        
        return np.maximum(0,x)

    def backward(self, grad_wrt_out):
        # # TODO
        # raise NotImplementedError
        return np.multiply(grad_wrt_out, (self.x>0))
    
class Dropout(Transform):
    """
    Implement this class. You may use your implementation from HW1
    """

    def __init__(self, p=0.1):
        Transform.__init__(self)
        """
        p is the Dropout probability
        """
        self.p = p
        self.mask = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x, train=True):
        """
        Get and apply a mask generated from np.random.binomial during training
        Scale your output accordingly during testing
        """
        # #TODO
        # raise NotImplementedError  

        if train:
            self.mask = (np.random.uniform(0,1,x.shape) > self.p)
            return (1/(1-self.p))*self.mask*x
        else:
            return x

    def backward(self, grad_wrt_out):
        """
        This method is only called during trianing.
        """
        # # TODO
        # raise NotImplementedError

        return (1/(1-self.p))*self.mask*grad_wrt_out

class Flatten(Transform):
    """
    Implement this class
    """
    def forward(self, x):
        """
        returns Flatten(x)
        """
        # #TODO
        # raise NotImplementedError
        self.flatten_input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dloss):
        """
        dLoss is the gradients wrt the output of Flatten
        returns gradients wrt the input to Flatten
        """
        # #TODO
        # raise NotImplementedError
        return dloss.reshape(self.flatten_input_shape)

class Conv(Transform):
    """
    Implement this class - Convolution Layer
    """
    def __init__(self, input_shape, filter_shape, rand_seed=0):
        """
        input_shape is a tuple: (channels, height, width)
        filter_shape is a tuple: (num of filters, filter height, filter width)
        weights shape (number of filters, number of input channels, filter height, filter width)
        Use Xavier initialization for weights, as instructed on handout
        Initialze biases as an array of zeros in shape of (num of filters, 1)
        """
        np.random.seed(rand_seed) # keep this line for autograding; you may remove it for training
        # #TODO
        # raise NotImplementedError
        self.img_shape = input_shape
        self.filter_shape = filter_shape
        self.weight = convolution_weight_init(input_shape=self.img_shape, filter_shape=filter_shape)
        self.bias = zeros_bias_init(filter_shape[0])

        self.momentum_w = 0
        self.momentum_b = 0
        
    def forward(self, inputs, stride=1, pad=2):
        """
        Forward pass of convolution between input and filters
        inputs is in the shape of (batch_size, num of channels, height, width)
        Return the output of convolution operation in shape (batch_size, num of filters, height, width)
        we recommend you use im2col here
        """
        # #TODO
        # raise NotImplementedError

        self.padding = pad
        self.stride = stride
        n, k_height, k_width = self.filter_shape
        self.input_shape = inputs.shape
        self.input_img = inputs
        N, C, H, W = inputs.shape

        out_H = (H + 2*pad - k_height)//stride + 1
        out_W = (W + 2*pad - k_width)//stride + 1

        img_col = im2col(inputs, k_height, k_width, pad, stride)

        weight_col = self.weight.reshape(n, -1)
        
        convolved = np.dot(weight_col, img_col) + self.bias

        convolved = convolved.reshape((n, out_H, out_W, N))

        return np.moveaxis(convolved, -1, 0)


    def backward(self, dloss):
        """
        Read Transform.backward()'s docstring in this file
        dloss shape (batch_size, num of filters, output height, output width)
        Return [gradient wrt weights, gradient wrt biases, gradient wrt input to this layer]
        """
        # #TODO
        # raise NotImplementedError

        n, k_height, k_width = self.filter_shape
        (N, n_filters, out_h, out_w) = dloss.shape

        grad_wrt_conv = np.moveaxis(dloss, 0, -1).reshape(n, -1)
        weight = self.weight.reshape(n, -1)
        
        dX_col = np.dot(weight.T, grad_wrt_conv)
        self.grad_wrt_input = im2col_bw(dX_col, self.input_shape, k_height, k_width, self.padding, self.stride)

        X_col = im2col(self.input_img, k_height, k_width, self.padding, self.stride)
        self.grad_wrt_w = np.dot(grad_wrt_conv, X_col.T).reshape(n, -1, k_height, k_width)

        self.grad_wrt_b = np.sum(grad_wrt_conv, axis = 1).reshape(-1,1)
        return [self.grad_wrt_w, self.grad_wrt_b, self.grad_wrt_input]


    def update(self, learning_rate=0.01, momentum_coeff=0.5):
        """
        Update weights and biases with gradients calculated by backward()
        Here we divide gradients by batch_size.
        """
        # #TODO
        # raise NotImplementedError
        batch_size = self.input_shape[0]

        self.momentum_w = momentum_coeff*self.momentum_w + self.grad_wrt_w/batch_size
        self.momentum_b = momentum_coeff*self.momentum_b + self.grad_wrt_b/batch_size

        self.weight = self.weight - learning_rate*self.momentum_w
        self.bias = self.bias - learning_rate*self.momentum_b

    def get_wb_conv(self):
        """
        Return weights and biases
        """
        # #TODO
        # raise NotImplementedError
        return [self.weight, self.bias]

class MaxPool(Transform):
    """
    Implement this class - MaxPool layer
    """
    def __init__(self, filter_shape, stride):
        """
        filter_shape is (filter_height, filter_width)
        stride is a scalar
        """
        # #TODO
        # raise NotImplementedError
        self.filter_shape = filter_shape
        self.stride = stride

        # self.filter_height, self.filter_width = filter_shape
        # self.stride = stride

    def forward(self, inputs):
        """
        forward pass of MaxPool
        inputs: (batch_size, C, H, W)
        """
        # #TODO
        # raise NotImplementedError

        self.input_shape = inputs.shape
        N, C, H, W = inputs.shape
        k_height, k_width = self.filter_shape[0], self.filter_shape[1]
        stride = self.stride
        out_H, out_W = (H - k_height)//stride + 1, (W - k_width)//stride + 1

        strides=(inputs.strides[0], inputs.strides[1],
                              self.stride * inputs.strides[2],
                              self.stride * inputs.strides[3],
                              inputs.strides[2], inputs.strides[3])
        
        windows = np.lib.stride_tricks.as_strided(inputs,
                     shape=(N, C, out_H, out_W, k_height, k_width),
                     strides=strides)
        
        out = np.max(windows, axis=(4, 5))

        maxs = out.repeat(k_height, axis=2).repeat(k_width, axis=3)
        x_window = inputs[:, :, :out_H * self.stride, :out_W * self.stride]
        mask = np.equal(x_window, maxs).astype(int)

        self.mask = mask

        # out = np.zeros((N, C, out_H, out_W))
        # cache = np.full((N, C, H, W), fill_value=False)
        # for j in range(out_H):
        #     for i in range(out_W):
        #         window = inputs[:, :, stride*j:stride*j+k_height, stride*i:stride*i+k_width]
        #         out[:, :, j, i] = np.max(window, axis=(2,3))
        #         
        # return out

        return out
    

    def backward(self, dloss):
        """
        dloss is the gradients wrt the output of forward()
        """
        # #TODO
        # raise NotImplementedError
        
        mask = self.mask
        k_height, k_width = self.filter_shape[0], self.filter_shape[1]
        dA = dloss.repeat(k_height, axis=2).repeat(k_width, axis=3)
        dA = np.multiply(dA, mask)
        pad = np.zeros(self.input_shape)
        pad[:, :, :dA.shape[2], :dA.shape[3]] = dA

        return pad

class LinearLayer(Transform):
    """
    Implement this class - Linear layer
    """
    def __init__(self, indim, outdim, rand_seed=0):
        """
        indim, outdim: input and output dimensions
        weights shape (indim,outdim)
        Use Xavier initialization for weights, as instructed on handout
        Initialze biases as an array of zeros in shape of (outdim,1)
        """
        np.random.seed(rand_seed) # keep this line for autograding; you may remove it for training
        # #TODO
        # raise NotImplementedError
        self.indim = indim
        self.outdim = outdim
        self.weight = random_weight_init(indim, outdim)
        self.bias = zeros_bias_init(outdim)

        self.momentum_w = 0
        self.momentum_b = 0        

    def forward(self, inputs):
        """
        Forward pass of linear layer
        inputs shape (batch_size, indim)
        """
        # #TODO
        # raise NotImplementedError
        
        self.input_linear = inputs
        self.batch_size = inputs.shape[0]
        out = np.dot(inputs, self.weight) + self.bias.reshape(1,-1)
        return out 
    
    def backward(self, dloss):
        """
        Read Transform.backward()'s docstring in this file
        dloss shape (batch_size, outdim)
        Return [gradient wrt weights, gradient wrt biases, gradient wrt input to this layer]
        """
        # #TODO
        # raise NotImplementedError
        self.grad_wrt_h = np.dot(dloss, self.weight.T)
        self.grad_wrt_w = np.dot(self.input_linear.T, dloss)
        self.grad_wrt_b = np.sum(dloss, axis=0).reshape(-1,1)

        return [self.grad_wrt_w, self.grad_wrt_b, self.grad_wrt_h]

    def update(self, learning_rate=0.01, momentum_coeff=0.5):
        """
        Similar to Conv.update()
        """
        # #TODO
        # raise NotImplementedError
        self.momentum_w = momentum_coeff * self.momentum_w +  self.grad_wrt_w/self.batch_size
        self.momentum_b = momentum_coeff * self.momentum_b +  self.grad_wrt_b/self.batch_size

        self.weight = self.weight - learning_rate*self.momentum_w
        self.bias = self.bias - learning_rate*self.momentum_b


    def get_wb_fc(self):
        """
        Return weights and biases as a tuple
        """
        # #TODO
        # raise NotImplementedError

        return (self.weight, self.bias)

class SoftMaxCrossEntropyLoss():
    """
    Implement this class
    """
    def forward(self, logits, labels, get_predictions=False):
        """
        logits are pre-softmax scores, labels are true labels of given inputs
        labels are one-hot encoded
        logits and labels are in  the shape of (batch_size, num_classes)
        returns loss as scalar
        (your loss should just be a sum of a batch, don't use mean)
        """
        # #TODO
        # raise NotImplementedError

        self.y = labels
        softmax_scores = softmax(logits)
        self.p = softmax_scores
        self.loss = -np.sum(np.multiply(labels, np.log(softmax_scores)), axis=1)

        return self.loss


    def backward(self):
        """
        return shape (batch_size, num_classes)
        (don't divide by batch_size here in order to pass autograding)
        """
        # #TODO
        # raise NotImplementedError

        grad_wrt_o = self.p - self.y
        
        return grad_wrt_o.astype(np.float32)



    def getAccu(self):
        """
        Implement as you wish, not autograded.
        """
        # #TODO
        # raise NotImplementedError
        preds = np.argmax(self.p, axis=1).astype(np.float32)
        labels = np.argmax(self.y, axis=1).astype(np.float32)

        accuracy = (preds == labels).mean()
        return accuracy

class ConvNet:
    """
    Class to implement forward and backward pass of the following network -
    Conv -> Relu -> MaxPool -> Linear -> Softmax
    For the above network run forward, backward and update
    """
    def __init__(self, out_dim = 20):
        """
        Initialize Conv, ReLU, MaxPool, LinearLayer, SoftMaxCrossEntropy objects
        Conv of input shape 3x32x32 with filter size of 1x5x5
        then apply Relu
        then perform MaxPooling with a 2x2 filter of stride 2
        then initialize linear layer with output 20 neurons
        Initialize SotMaxCrossEntropy object
        """
        # #TODO
        # raise NotImplementedError
        
        self.Conv_layer = Conv(input_shape=(3,32,32), filter_shape=(1,5,5))
        self.MaxPool_layer = MaxPool(filter_shape=(2,2), stride=2)
        self.ReLU = ReLU()
        self.Linear_layer = LinearLayer(indim=256, outdim=out_dim)
        self.loss_fn = SoftMaxCrossEntropyLoss()
        self.Flatten = Flatten()


    def forward(self, inputs, y_labels):
        """
        Implement forward function and return loss and predicted labels
        Arguments -
        1. inputs => input images of shape batch x channels x height x width
        2. labels => True labels

        Return loss and predicted labels after one forward pass
        """
        # #TODO
        # raise NotImplementedError

        N, C, H, W = inputs.shape
        Conv_out = self.Conv_layer.forward(inputs=inputs)
        ReLU_out = self.ReLU.forward(Conv_out)
        MaxPool_out = self.MaxPool_layer.forward(ReLU_out)
        Flatten_out = self.Flatten.forward(MaxPool_out)
        logits = self.Linear_layer.forward(Flatten_out)
        loss = self.loss_fn.forward(logits, y_labels)
        
        return [loss, self.loss_fn.p]


    def backward(self):
        """
        Implement this function to compute the backward pass
        Hint: Make sure you access the right values returned from the forward function
        DO NOT return anything from this function
        """
        # #TODO
        # raise NotImplementedError
        dwrt_out = self.loss_fn.backward()
        _, _, dwrt_lin = self.Linear_layer.backward(dwrt_out)
        dwrt_flatten = self.Flatten.backward(dwrt_lin)
        dwrt_maxpool = self.MaxPool_layer.backward(dwrt_flatten)
        dwrt_relu = self.ReLU.backward(dwrt_maxpool)
        _, _, dwrt_conv = self.Conv_layer.backward(dwrt_relu)



    def update(self, learning_rate, momentum_coeff):
        """
        Implement this function to update weights and biases with the computed gradients
        Arguments -
        1. learning_rate
        2. momentum_coefficient
        """
        # #TODO
        # raise NotImplementedError

        self.Linear_layer.update(learning_rate=learning_rate, momentum_coeff=momentum_coeff)
        self.Conv_layer.update(learning_rate=learning_rate, momentum_coeff=momentum_coeff)

class ConvNetTwo:
    """
    Class to implement forward and backward pass of the following network -
    Conv -> Relu -> MaxPool ->Conv -> Relu -> MaxPool -> Linear -> Softmax
    For the above network run forward, backward and update
    """
    def __init__(self, out_dim=20):
        """
        Initialize Conv, ReLU, MaxPool, Conv, ReLU,LinearLayer, SoftMaxCrossEntropy objects
        Conv of input shape 3x32x32 with filter size of 1x5x5
        then apply Relu
        then perform MaxPooling with a 2x2 filter of stride 2
        then Conv with filter size of 1x5x5
        then apply Relu
        then perform MaxPooling with a 2x2 filter of stride 2
        then initialize linear layer with output 20 neurons
        Initialize SotMaxCrossEntropy object
        """
        # #TODO
        # raise NotImplementedError
        self.Conv_layer_1 = Conv(input_shape=(3,32,32), filter_shape=(1,5,5))
        self.ReLU_1 = ReLU()
        self.MaxPool_layer_1 = MaxPool(filter_shape=(2,2), stride=2)
        
        self.Conv_layer_2 = Conv(input_shape=(1,16,16), filter_shape=(1,5,5)) 
        self.ReLU_2 = ReLU()
        self.MaxPool_layer_2 = MaxPool(filter_shape=(2,2), stride=2)
        
        self.Flatten = Flatten()
        self.Linear_layer = LinearLayer(indim=64, outdim=out_dim)
        self.loss_fn = SoftMaxCrossEntropyLoss()
        


    def forward(self, inputs, y_labels):
        """
        Implement forward function and return loss and predicted labels
        Arguments -
        1. inputs => input images of shape batch x channels x height x width
        2. labels => True labels

        Return loss and predicted labels after one forward pass
        """
        # #TODO
        # raise NotImplementedError

        Conv1_out = self.Conv_layer.forward(inputs=inputs)
        ReLU1_out = self.ReLU.forward(Conv1_out)
        MaxPool1_out = self.MaxPool_layer.forward(ReLU1_out)

        Conv2_out = self.Conv_layer.forward(inputs=MaxPool1_out)
        ReLU2_out = self.ReLU.forward(Conv2_out)
        MaxPool2_out = self.MaxPool_layer.forward(ReLU2_out)

        Flatten_out = self.Flatten.forward(MaxPool2_out)
        logits = self.Linear_layer.forward(Flatten_out)
        loss = self.loss_fn.forward(logits, y_labels)
        
        return [loss, self.loss_fn.p]

    def backward(self):
        """
        Implement this function to compute the backward pass
        Hint: Make sure you access the right values returned from the forward function
        DO NOT return anything from this function
        """
        # #TODO
        # raise NotImplementedError
        dwrt_out = self.loss_fn.backward()
        _, _, dwrt_lin = self.Linear_layer.backward(dwrt_out)
        dwrt_flatten = self.Flatten.backward(dwrt_lin)

        dwrt_maxpool2 = self.MaxPool_layer_2.backward(dwrt_flatten)
        dwrt_relu2 = self.ReLU_2.backward(dwrt_maxpool2)
        _, _, dwrt_conv2 = self.Conv_layer_2.backward(dwrt_relu2)

        dwrt_maxpool1 = self.MaxPool_layer_1.backward(dwrt_conv2)
        dwrt_relu1 = self.ReLU_1.backward(dwrt_maxpool1)
        _, _, dwrt_conv1 = self.Conv_layer_1.backward(dwrt_relu1)

    def update(self, learning_rate, momentum_coeff):
        """
        Implement this function to update weights and biases with the computed gradients
        Arguments -
        1. learning_rate
        2. momentum_coefficient
        """
        # #TODO
        # raise NotImplementedError
        self.Linear_layer.update(learning_rate=learning_rate, momentum_coeff=momentum_coeff)
        self.Conv_layer_1.update(learning_rate=learning_rate, momentum_coeff=momentum_coeff)
        self.Conv_layer_2.update(learning_rate=learning_rate, momentum_coeff=momentum_coeff)

class ConvNetThree:
    """
    Class to implement forward and backward pass of the following network -
    (Conv -> Relu -> MaxPool -> Dropout)x3 -> Linear -> Softmax
    For the above network run forward, backward and update
    """
    def __init__(self, out_dim=10, dropout=0.1):
        """
        Initialize Conv, ReLU, MaxPool, Conv, ReLU, Conv, ReLU, LinearLayer, SoftMaxCrossEntropy objects
        Conv of input shape 3x32x32 with 16 filters of size 3x3
        then apply Relu
        then perform MaxPooling with a 2x2 filter of stride 2
        then apply Dropout with probability 0.1
        then Conv with filter size of 16 filters of size 3x3
        then apply Relu
        then perform MaxPooling with a 2x2 filter of stride 2
        then apply Dropout with probability 0.1
        then Conv with filter size of 16 filters of size 3x3
        then apply Relu
        then apply Dropout with probability 0.1
        then initialize linear layer with output 10 neurons
        Initialize SotMaxCrossEntropy object
        """
        # #TODO
        # raise NotImplementedError

        self.Conv_layer_1 = Conv(input_shape=(3,32,32), filter_shape=(16,3,3))
        self.ReLU_1 = ReLU()
        self.MaxPool_layer_1 = MaxPool(filter_shape=(2,2), stride=2)
        self.Dropout_layer_1 = Dropout(p = dropout)
        
        self.Conv_layer_2 = Conv(input_shape=(16,17,17), filter_shape=(16,3,3)) 
        self.ReLU_2 = ReLU()
        self.MaxPool_layer_2 = MaxPool(filter_shape=(2,2), stride=2)
        self.Dropout_layer_2 = Dropout(p = dropout)
        
        self.Conv_layer_3 = Conv(input_shape=(16,9,9), filter_shape=(16,3,3)) 
        self.ReLU_3 = ReLU()
        # self.MaxPool_layer_3 = MaxPool(filter_shape=(2,2), stride=2)
        self.Dropout_layer_3 = Dropout(p = dropout)

        self.Flatten = Flatten()
        self.Linear_layer = LinearLayer(indim=16*5*5, outdim=out_dim)
        self.loss_fn = SoftMaxCrossEntropyLoss()

    def forward(self, inputs, y_labels):
        """
        Implement forward function and return loss and predicted labels
        Arguments -
        1. inputs => input images of shape batch x channels x height x width
        2. labels => True labels

        Return loss and predicted labels after one forward pass
        """
        # #TODO
        # raise NotImplementedError

        Conv1_out = self.Conv_layer_1.forward(inputs=inputs)
        ReLU1_out = self.ReLU_1.forward(Conv1_out)
        MaxPool1_out = self.MaxPool_layer_1.forward(ReLU1_out)
        Dropout1_out = self.Dropout_layer_1.forward(MaxPool1_out)

        Conv2_out = self.Conv_layer_2.forward(inputs=Dropout1_out)
        ReLU2_out = self.ReLU_2.forward(Conv2_out)
        MaxPool2_out = self.MaxPool_layer_2.forward(ReLU2_out)
        Dropout2_out = self.Dropout_layer_2.forward(MaxPool2_out)

        Conv3_out = self.Conv_layer_3.forward(inputs=Dropout2_out)
        ReLU3_out = self.ReLU_3.forward(Conv3_out)
        # MaxPool3_out = self.MaxPool_layer_3.forward(ReLU3_out)
        # Dropout3_out = self.Dropout_layer_3.forward(MaxPool3_out)

        Dropout3_out = self.Dropout_layer_3.forward(ReLU3_out)

        Flatten_out = self.Flatten.forward(Dropout3_out)
        logits = self.Linear_layer.forward(Flatten_out)
        loss = self.loss_fn.forward(logits, y_labels)
        
        return [loss, self.loss_fn.p]


    def backward(self):
        """
        Implement this function to compute the backward pass
        Hint: Make sure you access the right values returned from the forward function
        DO NOT return anything from this function
        """
        # #TODO
        # raise NotImplementedError

        dwrt_out = self.loss_fn.backward()
        _, _, dwrt_lin = self.Linear_layer.backward(dwrt_out)
        dwrt_flatten = self.Flatten.backward(dwrt_lin)

        dwrt_dropout3 = self.Dropout_layer_3.backward(dwrt_flatten)
        # dwrt_maxpool3 = self.MaxPool_layer_3.backward(dwrt_dropout3)
        # dwrt_relu3 = self.ReLU_3.backward(dwrt_maxpool3)
        dwrt_relu3 = self.ReLU_3.backward(dwrt_dropout3)

        _, _, dwrt_conv3 = self.Conv_layer_3.backward(dwrt_relu3)

        dwrat_dropout2 = self.Dropout_layer_2.backward(dwrt_conv3)
        dwrt_maxpool2 = self.MaxPool_layer_2.backward(dwrat_dropout2)
        dwrt_relu2 = self.ReLU_2.backward(dwrt_maxpool2)
        _, _, dwrt_conv2 = self.Conv_layer_2.backward(dwrt_relu2)

        dwrt_dropout1 = self.Dropout_layer_1.backward(dwrt_conv2)
        dwrt_maxpool1 = self.MaxPool_layer_1.backward(dwrt_dropout1)
        dwrt_relu1 = self.ReLU_1.backward(dwrt_maxpool1)
        _, _, dwrt_conv1 = self.Conv_layer_1.backward(dwrt_relu1)
       

    def update(self, learning_rate, momentum_coeff):
        """
        Implement this function to update weights and biases with the computed gradients
        Arguments -
        1. learning_rate
        2. momentum_coefficient
        """
        # #TODO
        # raise NotImplementedError

        self.Linear_layer.update(learning_rate=learning_rate, momentum_coeff=momentum_coeff)
        self.Conv_layer_1.update(learning_rate=learning_rate, momentum_coeff=momentum_coeff)
        self.Conv_layer_2.update(learning_rate=learning_rate, momentum_coeff=momentum_coeff)
        self.Conv_layer_3.update(learning_rate=learning_rate, momentum_coeff=momentum_coeff)

class ConvNetFour:
    """
    Class to implement forward and backward pass of the following network -
    (Conv -> Relu -> MaxPool -> Dropout)x3 -> Linear -> Softmax
    For the above network run forward, backward and update
    """
    def __init__(self, out_dim=10, dropout=0.1):
        """
        Initialize Conv, ReLU, MaxPool, Conv, ReLU, Conv, ReLU, LinearLayer, SoftMaxCrossEntropy objects
        Conv of input shape 3x32x32 with 64 filters of size 3x3
        then apply Relu
        then perform MaxPooling with a 2x2 filter of stride 2
        then apply Dropout with probability 0.1
        then Conv with filter size of 32 filters of size 3x3
        then apply Relu
        then perform MaxPooling with a 2x2 filter of stride 2
        then apply Dropout with probability 0.1
        then Conv with filter size of 16 filters of size 3x3
        then apply Relu
        then apply Dropout with probability 0.1
        then initialize linear layer with output 10 neurons
        Initialize SotMaxCrossEntropy object
        """
        # #TODO
        # raise NotImplementedError

        self.Conv_layer_1 = Conv(input_shape=(3,32,32), filter_shape=(64,3,3))
        self.ReLU_1 = ReLU()
        self.MaxPool_layer_1 = MaxPool(filter_shape=(2,2), stride=2)
        self.Dropout_layer_1 = Dropout(p = dropout)
        
        self.Conv_layer_2 = Conv(input_shape=(64,17,17), filter_shape=(32,3,3)) 
        self.ReLU_2 = ReLU()
        self.MaxPool_layer_2 = MaxPool(filter_shape=(2,2), stride=2)
        self.Dropout_layer_2 = Dropout(p = dropout)
        
        self.Conv_layer_3 = Conv(input_shape=(32,9,9), filter_shape=(16,3,3)) 
        self.ReLU_3 = ReLU()
        # self.MaxPool_layer_3 = MaxPool(filter_shape=(2,2), stride=2)
        self.Dropout_layer_3 = Dropout(p = dropout)

        self.Flatten = Flatten()
        self.Linear_layer = LinearLayer(indim=16*11*11, outdim=out_dim)
        self.loss_fn = SoftMaxCrossEntropyLoss()

    def forward(self, inputs, y_labels):
        """
        Implement forward function and return loss and predicted labels
        Arguments -
        1. inputs => input images of shape batch x channels x height x width
        2. labels => True labels

        Return loss and predicted labels after one forward pass
        """
        # #TODO
        # raise NotImplementedError

        Conv1_out = self.Conv_layer_1.forward(inputs=inputs)
        ReLU1_out = self.ReLU_1.forward(Conv1_out)
        MaxPool1_out = self.MaxPool_layer_1.forward(ReLU1_out)
        Dropout1_out = self.Dropout_layer_1.forward(MaxPool1_out)

        Conv2_out = self.Conv_layer_2.forward(inputs=Dropout1_out)
        ReLU2_out = self.ReLU_2.forward(Conv2_out)
        MaxPool2_out = self.MaxPool_layer_2.forward(ReLU2_out)
        Dropout2_out = self.Dropout_layer_2.forward(MaxPool2_out)

        Conv3_out = self.Conv_layer_3.forward(inputs=Dropout2_out)
        ReLU3_out = self.ReLU_3.forward(Conv3_out)
        # MaxPool3_out = self.MaxPool_layer_3.forward(ReLU3_out)
        # Dropout3_out = self.Dropout_layer_3.forward(MaxPool3_out)
        Dropout3_out = self.Dropout_layer_3.forward(ReLU3_out)
        Flatten_out = self.Flatten.forward(Dropout3_out)
        logits = self.Linear_layer.forward(Flatten_out)
        loss = self.loss_fn.forward(logits, y_labels)
        
        return [loss, self.loss_fn.p]


    def backward(self):
        """
        Implement this function to compute the backward pass
        Hint: Make sure you access the right values returned from the forward function
        DO NOT return anything from this function
        """
        # #TODO
        # raise NotImplementedError

        dwrt_out = self.loss_fn.backward()
        _, _, dwrt_lin = self.Linear_layer.backward(dwrt_out)
        dwrt_flatten = self.Flatten.backward(dwrt_lin)

        dwrt_dropout3 = self.Dropout_layer_3.backward(dwrt_flatten)
        # dwrt_maxpool3 = self.MaxPool_layer_3.backward(dwrt_dropout3)
        # dwrt_relu3 = self.ReLU_3.backward(dwrt_maxpool3)
        dwrt_relu3 = self.ReLU_3.backward(dwrt_dropout3)
        _, _, dwrt_conv3 = self.Conv_layer_3.backward(dwrt_relu3)

        dwrat_dropout2 = self.Dropout_layer_2.backward(dwrt_conv3)
        dwrt_maxpool2 = self.MaxPool_layer_2.backward(dwrat_dropout2)
        dwrt_relu2 = self.ReLU_2.backward(dwrt_maxpool2)
        _, _, dwrt_conv2 = self.Conv_layer_2.backward(dwrt_relu2)

        dwrt_dropout1 = self.Dropout_layer_1.backward(dwrt_conv2)
        dwrt_maxpool1 = self.MaxPool_layer_1.backward(dwrt_dropout1)
        dwrt_relu1 = self.ReLU_1.backward(dwrt_maxpool1)
        _, _, dwrt_conv1 = self.Conv_layer_1.backward(dwrt_relu1)
       

    def update(self, learning_rate, momentum_coeff):
        """
        Implement this function to update weights and biases with the computed gradients
        Arguments -
        1. learning_rate
        2. momentum_coefficient
        """
        # #TODO
        # raise NotImplementedError

        self.Linear_layer.update(learning_rate=learning_rate, momentum_coeff=momentum_coeff)
        self.Conv_layer_1.update(learning_rate=learning_rate, momentum_coeff=momentum_coeff)
        self.Conv_layer_2.update(learning_rate=learning_rate, momentum_coeff=momentum_coeff)
        self.Conv_layer_3.update(learning_rate=learning_rate, momentum_coeff=momentum_coeff)

def one_hot_encode(labels):
    """
    One hot encode labels
    """
    one_hot_labels = np.array([[i==label for i in range(len(CLASS_IDS.keys()))] for label in labels], np.int32)
    return one_hot_labels

def prep_imagenet_data(train_images, train_labels, val_images, val_labels):
    # onehot encode labels
    train_labels = one_hot_encode(train_labels)
    val_labels = one_hot_encode(val_labels)

    # standardize to [-1, 1]
    train_images = (train_images - 127.5) / 127.5
    val_images = (val_images - 127.5) / 127.5

    # put channels first
    train_images = np.transpose(train_images, (0, 3, 1, 2))
    val_images = np.transpose(val_images, (0, 3, 1, 2))

    return train_images, train_labels, val_images, val_labels


# Implement the training as you wish. This part will not be autograded
# Feel free to implement other helper libraries (i.e. matplotlib, seaborn) but please do not import other libraries (i.e. torch, tensorflow, etc.) for the training
#Note: make sure to download the data from the resources tab on piazza
if __name__ == '__main__':
    # This part may be helpful to write the training loop
    """
    # Training parameters
    parser = ArgumentParser(description='CNN')
    parser.add_argument('--batch_size', type=int, default = 128)
    parser.add_argument('--learning_rate', type=float, default = 0.001)
    parser.add_argument('--momentum', type=float, default = 0.95)
    parser.add_argument('--num_epochs', type=int, default = 50)
    parser.add_argument('--seed', type=int, default = 47)
    parser.add_argument('--dropout_p', type=float, default=0.1)
    parser.add_argument('--name_prefix', type=str, default=None)
    parser.add_argument('--num_filters', type=int, default=16)
    parser.add_argument('--filter_size', type=int, default=3)
    args = parser.parse_args()
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    MOMENTUM = args.momentum
    EPOCHS = args.num_epochs
    SEED = args.seed
    print('\n'.join([f'{k}: {v}' for k, v in vars(args).items()]))
    """

    ## DATA EXPLORATION 
    with open("10417-tiny-imagenet-train-bal.pkl", "rb") as f:
        train_dict = pickle.load(f)
        train_images = train_dict["images"]
        train_labels = train_dict["labels"]
    
    with open("10417-tiny-imagenet-val-bal.pkl", "rb") as f:
        val_dict = pickle.load(f)
        val_images = val_dict["images"]
        val_labels = val_dict["labels"]
    
    ## Problem 1a: Data Vis
    # TODO: plot data samples for train/val
    def data_vis(train_images, train_labels, val_images, val_labels):
        fig, axes = plt.subplots(10, 4)
        for i in range(10):
            ax = axes[i]
            train_images_class = train_images[train_labels == i]
            val_images_class = val_images[val_labels == i]

            ax[0].imshow(train_images_class[0])
            ax[0].set_xlabel(f"train set - {IDS_CLASS[i]}")
            ax[0].xaxis.set_label_coords(2, 0.5)

            ax[1].imshow(train_images_class[1])
            ax[1].set_xlabel(f"train set - {IDS_CLASS[i]}")
            ax[1].xaxis.set_label_coords(2, 0.5)

            ax[2].imshow(val_images_class[0])
            ax[2].set_xlabel(f"val set - {IDS_CLASS[i]}")
            ax[2].xaxis.set_label_coords(2, 0.5)

            ax[3].imshow(val_images_class[1])
            ax[3].set_xlabel(f"val set - {IDS_CLASS[i]}")
            ax[3].xaxis.set_label_coords(2, 0.5)

        plt.show()
        return
    # data_vis(train_images, train_labels, val_images, val_labels)
    ## Problem 1b: Data Statistics
    # TODO: plot/show image stats
    def data_stats(train_images, train_labels, val_images, val_labels):
        print(f"data type: {train_images.dtype}")
        print(f"training data range: [{np.min(train_images)}, {np.max(train_images)}]")
        print(f"validation data range: [{np.min(val_images)}, {np.max(val_images)}]")
        num_channels = train_images.shape[3]

        # num samples per class
        for i in range(10):
            num_train = np.sum(train_labels == i)
            num_val = np.sum(val_labels == i)
            print(f"number of {IDS_CLASS[i]} training samples: {num_train}")
            print(f"number of {IDS_CLASS[i]} validation samples: {num_val}")
        
        # mean and stddev per channel
        for c in range(num_channels):
            train_images_channel = train_images[:, :, :, c]
            val_images_channel = val_images[:, :, :, c]
            train_mean = np.round(np.mean(train_images_channel), 2)
            val_mean = np.round(np.mean(val_images_channel), 2)
            train_stddev = np.round(np.std(train_images_channel), 2)
            val_stddev = np.round(np.std(val_images_channel), 2)
            print(f"channel {c} training mean: {train_mean}")
            print(f"channel {c} validation mean: {val_mean}")
            print(f"channel {c} training standard deviation: {train_stddev}")
            print(f"channel {c} validation standard deviation: {val_stddev}")
    
    data_stats(train_images, train_labels, val_images, val_labels)
    # preprocessing imagenet data for training (don't change this)

    org_val_images = val_images
    org_val_labels = val_labels

    train_images, train_labels, val_images, val_labels = prep_imagenet_data(train_images, train_labels, val_images, val_labels)

    ## Problem 2a: Train ConvNet
    # np.random.seed(SEED)
    # TODO

    def get_minibatch(trainX, trainY, batch_size):

        num_samples = trainX.shape[0]

        # Shuffle the data
        permutation = np.random.permutation(num_samples)
        shuffled_trainX = trainX[permutation]
        shuffled_trainY = trainY[permutation]

        minibatches = []

        num_batches = num_samples // batch_size

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size

            batch_X = shuffled_trainX[start_idx:end_idx]
            batch_Y = shuffled_trainY[start_idx:end_idx]
            minibatches.append((batch_X, batch_Y))

        return minibatches
    
    def train_model(model, trainX, trainY, testX, testY, epochs, learning_rate, momentum_coeff, batch_size = 32):
        loss_train_epoch, accuracy_train_epoch = [],[]
        loss_test_epoch, accuracy_test_epoch = [],[]

        for epoch in tqdm(range(1, epochs+1)):
            train_loss, train_acc = [], []

            minibatches = get_minibatch(trainX, trainY, batch_size)
            
            for data, label in minibatches: 
                loss_tr, predictions  = model.forward(data, label)
                # loss_tr = model.loss_func.forward(model_out, label)
                model.backward()
                model.update(learning_rate, momentum_coeff)

                train_loss.append(loss_tr)
                acc_tr = model.loss_fn.getAccu()
                train_acc.append(acc_tr)

            loss_te, predictions = model.forward(testX, testY)
            acc_te = model.loss_fn.getAccu()
            
            loss_train_epoch.append(np.mean(train_loss))
            loss_test_epoch.append(np.mean(loss_te))
            accuracy_train_epoch.append(np.mean(train_acc))
            accuracy_test_epoch.append(np.mean(acc_te))

            print(f"Best Training Loss: {np.min(train_loss)}")
            print(f"Best Validation Loss: {np.min(loss_te)}")
            print(f"Best Training Accuracy: {np.max(train_acc)}")
            print(f"Best Validation Accuracy: {np.max(acc_te)}")

            # save it for later just in case we need them
            with open("convnet_loss_acc.pkl", "wb") as file:
                pickle.dump([np.mean(train_loss), np.mean(loss_te),np.mean(train_acc), np.max(acc_te)], file)

            with open("convnet_model.pkl", "wb") as file:
                pickle.dump(model, file)

        return loss_train_epoch, loss_test_epoch, accuracy_train_epoch, accuracy_test_epoch

    def plot_loss(ax_loss, ax_acc, loss_train, loss_test, accuracy_train, accuracy_test, loss_title, acc_title, label):
        
        ax_loss.plot(loss_train, label = label + "_train")
        ax_loss.plot(loss_test, label = label + "_test")
        ax_loss.set_title(loss_title)
        ax_loss.legend()
        ax_loss.set_xlabel("Epochs")
        ax_loss.set_ylabel("Cross Entropy Loss")

        ax_acc.plot(accuracy_train, label = label + "_train")
        ax_acc.plot(accuracy_test, label = label + "_test")
        ax_acc.set_title(acc_title)
        ax_acc.legend()
        ax_acc.set_xlabel("Epochs")
        ax_acc.set_ylabel("Accuracy")

    # ConvNet = ConvNet(out_dim=10)

    # loss_train, loss_test, acc_train, acc_test = train_model(ConvNet, train_images, train_labels, val_images, val_labels, 5, 0.01, 0.5,  32)
    
    # fig_loss, ax_loss = plt.subplots()
    # fig_acc, ax_acc = plt.subplots()

    # plot_loss(ax_loss, ax_acc, loss_train, loss_test, acc_train, acc_test, 
    #         loss_title="Loss vs Epochs - ConvNet", acc_title="Accuracy vs Epochs - ConvNet", label = "ConvNet")
    # plt.figure(figsize=(5,3))
    # plt.show()

    ## Problem 2b: Train ConvNetThree
    # np.random.seed(SEED)
    # TODO

    # ConvNet = ConvNetThree(out_dim=10)

    # loss_train, loss_test, acc_train, acc_test = train_model(ConvNet, train_images, train_labels, val_images, val_labels, 5, 0.01, 0.5,  32)
    
    # fig_loss, ax_loss = plt.subplots()
    # fig_acc, ax_acc = plt.subplots()

    # plot_loss(ax_loss, ax_acc, loss_train, loss_test, acc_train, acc_test, 
    #         loss_title="Loss vs Epochs - ConvNetThree", acc_title="Accuracy vs Epochs - ConvNetThree", label = "ConvNetThree")
    # plt.figure(figsize=(5,3))
    # plt.show()

    ## Problem 2c: Train your best model
    # np.random.seed(SEED)
    # TODO

    # ConvNet = ConvNetThree(out_dim=10)

    # loss_train, loss_test, acc_train, acc_test = train_model(ConvNet, train_images, train_labels, val_images, val_labels, 50, 0.01, 0.5,  32)

    # fig_loss, ax_loss = plt.subplots()
    # fig_acc, ax_acc = plt.subplots()

    # plot_loss(ax_loss, ax_acc, loss_train, loss_test, acc_train, acc_test, 
    #         loss_title="Loss vs Epochs - ConvNetThree", acc_title="Accuracy vs Epochs - ConvNetThree", label = "ConvNetThree_dropout0.1")
    
    #
    # ConvNet_tune1 = ConvNetThree(out_dim=10, dropout=0.3)

    # loss_train_1, loss_test_1, acc_train_1, acc_test_1 = train_model(ConvNet_tune1, train_images, train_labels, val_images, val_labels, 50, 0.01, 0.5,  32)

    # plot_loss(ax_loss, ax_acc, loss_train_1, loss_test_1, acc_train_1, acc_test_1, 
    #         loss_title="Loss vs Epochs - ConvNetThree", acc_title="Accuracy vs Epochs - ConvNetThree", label = "ConvNetThree_dropout0.3")

    #
    # ConvNet_tune2 = ConvNetThree(out_dim=10, dropout=0.5)

    # loss_train_2, loss_test_2, acc_train_2, acc_test_2 = train_model(ConvNet_tune2, train_images, train_labels, val_images, val_labels, 50, 0.01, 0.5,  32)

    # plot_loss(ax_loss, ax_acc, loss_train_2, loss_test_2, acc_train_2, acc_test_2, 
    #         loss_title="Loss vs Epochs - ConvNetThree", acc_title="Accuracy vs Epochs - ConvNetThree", label = "ConvNetThree_dropout0.5")
    
    # plt.show()

    #
    # ConvNet_Improved = ConvNetFour(out_dim=10, dropout = 0.5)
    # loss_train_imp, loss_test_imp, acc_train_imp, acc_test_imp = train_model(ConvNet_Improved, train_images, train_labels, val_images, val_labels, 50, 0.01, 0.5,  32)
    
    # fig_loss, ax_loss = plt.subplots()
    # fig_acc, ax_acc = plt.subplots()
    
    # plot_loss(ax_loss, ax_acc, loss_train_2, loss_test_2, acc_train_2, acc_test_2, 
    #         loss_title="Loss vs Epochs - ConvNet_improved", acc_title="Accuracy vs Epochs - ConvNet_improved", label = "ConvNet_improved")
    
    # plt.show()
    ## Problem 3a: Evaluation
    # TODO: plot confusion matrix and misclassified images on imagenet data

    def eval_model(model, images, one_hot_labels, labels):
        loss, labels_pred = model.forward(images, one_hot_labels)
        
        result = np.zeros((10, 10))
        for label in range(len(labels)):
            i = np.argmax(labels_pred[label])
            j = labels[label]
            result[i, j] += 1
        print(result)

        return result
    
    def plot_misclassified(model, images_transformed, one_hot_labels, images, labels):
        loss, labels_pred = model.forward(images_transformed, one_hot_labels)
        bad_images = []
        pred = []
        ground_truth = []

        i = 0
        while(len(bad_images) < 5):
            if(np.argmax(labels_pred[i]) != labels[i]):
                bad_images.append(images[i])
                pred.append(labels_pred[i])
                ground_truth.append(labels[i])
            i += 1

        fig, axes = plt.subplots(1, 5)
        for i in range(5):
            image = bad_images[i]
            label_pred = IDS_CLASS[np.argmax(pred[i])]
            label_true = IDS_CLASS[ground_truth[i]]
            ax = axes[i]
            ax.imshow(image)
            ax.set_title(f"Predicted: {label_pred}\nActual: {label_true}")
        print(pred, ground_truth)
        plt.show()

    with open("convnet3_baseline_model.pkl", "rb") as file:
        model = pickle.load(file)

    plot_misclassified(model, val_images, val_labels, org_val_images, org_val_labels)

    with open("convnet_improved_model.pkl", "rb") as file:
        model = pickle.load(file)

    plot_misclassified(model, val_images, val_labels, org_val_images, org_val_labels)

    ## Problem 3b: Evaluate on COCO  "10417-coco.pkl"
    # TODO: Load COCO Data
    with open("10417-coco.pkl", "rb") as file:
        coco_dict = pickle.load(file)
        coco_images = coco_dict["images"]
        coco_labels = coco_dict["labels"]

    # TODO: plot COCO data

    def data_vis_single(images, labels):
        fig, axes = plt.subplots(10, 2)
        for i in range(10):
            ax = axes[i]
            images_class = images[labels == i]

            ax[0].imshow(images_class[0])
            ax[0].set_xlabel(f"{IDS_CLASS[i]}")
            ax[0].xaxis.set_label_coords(2, 0.5)

            ax[1].imshow(images_class[1])
            ax[1].set_xlabel(f"{IDS_CLASS[i]}")
            ax[1].xaxis.set_label_coords(2, 0.5)
        plt.show()

    data_vis_single(coco_images, coco_labels)

    # TODO: get/plot stats COCO

    def data_stats_single(images, labels):
        print(f"data type: {images.dtype}")
        print(f"data range: [{np.min(images)}, {np.max(images)}]")
        num_channels = images.shape[3]

        # num samples per class
        for i in range(10):
            num = np.sum(labels == i)
            print(f"number of {IDS_CLASS[i]} samples: {num}")
        
        # mean and stddev per channel
        for c in range(num_channels):
            images_channel = images[:, :, :, c]
            mean = np.round(np.mean(images_channel), 2)
            stddev = np.round(np.std(images_channel), 2)
            print(f"channel {c} training mean: {mean}")
            print(f"channel {c} training standard deviation: {stddev}")

    data_stats_single(coco_images, coco_labels)

    # TODO: preprocess COCO data, standardize, onehot encode, put channels first
    # hint: see see prep_imagenet_data() for reference (make sure data range is [-1, 1] before eval!)

    def preprocess_coco_data(images, labels):
        
        labels = one_hot_encode(labels)
        images = (images - 0.5) / 0.5
        images = np.transpose(images, (0, 3, 1, 2))

        return images, labels
    
    coco_images_preprocessed, coco_labels_preprocessed = preprocess_coco_data(coco_images, coco_labels)

    # TODO: get loss and accuracy COCO

    def loss_acc(model, images, labels):
        loss, labels_pred = model.forward(images, labels)
        loss = loss/images.shape[0]
        acc = model.loss_fn.getAccu()
        print(f"Loss: {loss}")
        print(f"Accuracy: {acc}")

    with open("convnet3_baseline_model.pkl", "rb") as file:
        model = pickle.load(file)
    
    loss_acc(model, coco_images_preprocessed, coco_labels_preprocessed)

    with open("convnet_improved_model.pkl", "rb") as file:
        model = pickle.load(file)

    loss_acc(model, coco_images_preprocessed, coco_labels_preprocessed)

    # TODO: get confusion matrix COCO and misclassified images COCO

    with open("convnet3_baseline_model.pkl", "rb") as file:
        model = pickle.load(file)
    eval_model(model, coco_images_preprocessed, coco_labels_preprocessed, coco_labels)
    plot_misclassified(model, coco_images_preprocessed, coco_labels_preprocessed, coco_images, coco_labels)

    with open("convnet_improved_model.pkl", "rb") as file:
        model = pickle.load(file)
    eval_model(model, coco_images_preprocessed, coco_labels_preprocessed, coco_labels)
    plot_misclassified(model, coco_images_preprocessed, coco_labels_preprocessed, coco_images, coco_labels)
    
