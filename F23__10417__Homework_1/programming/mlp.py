"""
Fall 2023, 10-417/617
Assignment-1

IMPORTANT:
    DO NOT change any function signatures

September 2023
"""


import numpy as np

def random_weight_init(input, output):
    b = np.sqrt(6)/np.sqrt(input+output)
    return np.random.uniform(-b, b, (output, input))

def zeros_bias_init(outd):
    return np.zeros((outd, 1))

def labels2onehot(labels):
    return np.array([[i==lab for i in range(12)] for lab in labels])

epsilon = 1e-8

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
        In this function, we accumulate the gradient values instead of assigning
        the gradient values. This allows us to call forward and backward multiple
        times while only update parameters once.
        Compute and save the gradients wrt the parameters for step()
        Return grad_wrt_x which will be the grad_wrt_out for previous Transform
        """
        pass

    def step(self):
        """
        Apply gradients to update the parameters
        """
        pass

    def zerograd(self):
        """
        This is used to Reset the gradients.
        Usually called before backward()
        """
        pass

class ReLU(Transform):
    """
    ReLU non-linearity, combined with dropout
    IMPORTANT the Autograder assumes these function signatures
    """
    def __init__(self, dropout_probability=0):
        Transform.__init__(self)
        self.dropout = dropout_probability

    def forward(self, x, train=True):
        # IMPORTANT the autograder assumes that you call np.random.uniform(0,1,x.shape) exactly once in this function
        """
        x shape (indim, batch_size)
        """
        self.x = x
        self.train = train
        self.mask = (np.random.uniform(0,1,x.shape) > self.dropout)
        if self.train:
            out = np.multiply(self.x, self.mask)*(1/(1-self.dropout))
        else:
            out = self.x

        return np.maximum(0,out)

    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (outdim, batch_size)
        """        
        if self.dropout > 0:
            grad_wrt_out = grad_wrt_out * self.mask 
        
        return np.multiply(grad_wrt_out, (self.x>0))*(1/(1-self.dropout))

        

class LinearMap(Transform):
    """
    Implement this class
    For consistency, please use random_xxx_init() functions given on top for initialization
    """
    def __init__(self, indim, outdim, alpha=0, lr=0.01):
        """
        indim: input dimension
        outdim: output dimension
        alpha: parameter for momentum updates
        lr: learning rate
        """
        Transform.__init__(self)
        self.input_dim = indim
        self.output_dim = outdim
        self.alpha = alpha
        self.lr = lr
        self.momentum_w = 0.0
        self.momentum_b = 0.0

    def forward(self, x):
        """
        x shape (indim, batch_size)
        return shape (outdim, batch_size)
        """
        pot = np.dot(self.weight, x) + self.bias
        self.x = x
        return pot

    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (outdim, batch_size)
        return shape (indim, batch_size)
        Your backward call should accumulate gradients.
        """
        self.grad_wrt_h = np.dot(np.transpose(self.weight), grad_wrt_out)
        self.grad_wrt_w = np.dot(grad_wrt_out, np.transpose(self.x))
        self.grad_wrt_b = np.sum(grad_wrt_out, axis=1).reshape(-1,1)

        return self.grad_wrt_h

    def step(self):
        """
        apply gradients calculated by backward() to update the parameters

        Make sure your gradient step takes into account momentum.
        Use alpha as the momentum parameter.
        """
        self.momentum_w = self.alpha * self.momentum_w +  self.grad_wrt_w
        self.momentum_b = self.alpha * self.momentum_b +  self.grad_wrt_b

        self.weight = self.weight - self.lr *(self.momentum_w)
        self.bias = self.bias - self.lr *(self.momentum_b)

    def zerograd(self):
        self.grad_wrt_h = 0
        self.grad_wrt_w = 0
        self.grad_wrt_b = 0

    def getW(self):
        """
        return W shape (outdim, indim)
        """
        return self.weight

    def getb(self):
        """
        return b shape (outdim, 1)
        """
        return self.bias

    def loadparams(self, w, b):
        self.weight = w
        self.bias = b

class SoftmaxCrossEntropyLoss:
    """
    Implement this class
    """
    def forward(self, logits, labels):
        """
        logits are pre-softmax scores, labels are true labels of given inputs
        labels are one-hot encoded
        logits and labels are in the shape of (num_classes,batch_size)
        returns loss as scalar
        (your loss should be a mean value on batch_size)
        """
        self.y = labels
        self.input = logits

        softmax_scores = np.divide(np.exp(logits),np.sum(np.exp(logits),axis = 0) + epsilon)
        loss = -np.sum(np.multiply(labels, np.log(softmax_scores)), axis=0)

        self.p = softmax_scores
        self.loss = np.mean(loss)

        return self.loss

    def backward(self):
        """
        return shape (num_classes,batch_size)
        (don't forget to divide by batch_size because your loss is a mean)
        """
        grad_wrt_o = self.p - self.y
        
        return np.divide(grad_wrt_o, self.y.shape[1]).astype(np.float32)

    def getAccu(self):
        """
        return accuracy here (as you wish)
        This part is not autograded.
        """
        preds = np.argmax(self.p, axis=0).astype(np.float32)
        labels = np.argmax(self.y, axis=0).astype(np.float32)

        accuracy = (preds == labels).mean()
        return accuracy 

class SingleLayerMLP(Transform):
    """
    Implement this class
    """
    def __init__(self, indim, outdim, hiddenlayer=100, alpha=0.1, dropout_probability=0, lr=0.01):
        Transform.__init__(self)
        self.indim = indim
        self.outdim = outdim
        self.hiddendim = hiddenlayer
        self.alpha = alpha
        self.dropout = dropout_probability
        self.lr = lr
        self.layers = [hiddenlayer]

        self.layer0 = LinearMap(indim=self.indim, outdim=self.hiddendim, alpha=self.alpha, lr=self.lr)
        self.layer1 = LinearMap(indim=self.hiddendim, outdim=self.outdim, alpha=self.alpha, lr=self.lr)
        self.activation_func = ReLU(dropout_probability=self.dropout)

    def forward(self, x, train=True):
        """
        x shape (indim, batch_size)
        """
        self.train = train

        pot = self.layer0.forward(x = x)
        
        h = self.activation_func.forward(pot, train=self.train)
        
        logits = self.layer1.forward(x = h)

        return logits

    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (outdim, batch_size)
        """
        grad_wrt_l1 = self.layer1.backward(grad_wrt_out=grad_wrt_out)
        grad_wrt_activation = self.activation_func.backward(grad_wrt_out=grad_wrt_l1)
        grad_wrt_l0 = self.layer0.backward(grad_wrt_out=grad_wrt_activation)

        return grad_wrt_l0
        

    def step(self):
        self.layer1.step()
        self.layer0.step()

    def zerograd(self):
        self.layer1.zerograd()
        self.layer0.zerograd()

    def loadparams(self, Ws, bs):
        """
        use LinearMap.loadparams() to implement this
        Ws is a list, whose element is weights array of a layer, first layer first
        bs for bias similarly
        e.g., Ws may be [LinearMap1.W, LinearMap2.W]
        Used for autograder.
        """
        self.layer0.loadparams(Ws[0], bs[0])
        self.layer1.loadparams(Ws[1], bs[1])

    def getWs(self):
        """
        Return the weights for each layer
        Return a list containing weights for first layer then second and so on...
        """
        return [self.layer0.getW(), self.layer1.getW()]

    def getbs(self):
        """
        Return the biases for each layer
        Return a list containing bias for first layer then second and so on...
        """
        return [self.layer0.getb(), self.layer1.getb()]

class TwoLayerMLP(Transform):
    """
    Implement this class
    Everything similar to SingleLayerMLP
    """
    def __init__(self, indim, outdim, hiddenlayers=[100,100], alpha=0.1, dropout_probability=0, lr=0.01):
        Transform.__init__(self)
        self.indim = indim
        self.outdim = outdim
        self.hiddendim1 = hiddenlayers[0]
        self.hiddendim2 = hiddenlayers[1]
        self.alpha = alpha
        self.dropout = dropout_probability
        self.lr = lr
        self.layers = hiddenlayers

        self.layer0 = LinearMap(indim=self.indim, outdim=self.hiddendim1, alpha=self.alpha, lr=self.lr)
        self.layer1 = LinearMap(indim=self.hiddendim1, outdim=self.hiddendim2, alpha=self.alpha, lr=self.lr)
        self.layer2 = LinearMap(indim=self.hiddendim2, outdim=self.outdim, alpha=self.alpha, lr=self.lr)
        self.activation_func1 = ReLU(dropout_probability=self.dropout)
        self.activation_func2 = ReLU(dropout_probability=self.dropout)

    def forward(self, x, train=True):
        self.train = train

        pot1 = self.layer0.forward(x = x)
        
        h1 = self.activation_func1.forward(pot1, train=self.train)
        
        pot2 = self.layer1.forward(x = h1)

        h2 = self.activation_func2.forward(pot2, train=self.train)

        logits = self.layer2.forward(x = h2)

        return logits

    def backward(self, grad_wrt_out):
        grad_wrt_l2 = self.layer2.backward(grad_wrt_out=grad_wrt_out)
        grad_wrt_activation_2 = self.activation_func2.backward(grad_wrt_out=grad_wrt_l2)
        grad_wrt_l1 = self.layer1.backward(grad_wrt_out=grad_wrt_activation_2)
        grad_wrt_activation_1 = self.activation_func1.backward(grad_wrt_out=grad_wrt_l1)
        grad_wrt_l0 = self.layer0.backward(grad_wrt_out=grad_wrt_activation_1)

        return grad_wrt_l0

    def step(self):
        self.layer2.step()
        self.layer1.step()
        self.layer0.step()

    def zerograd(self):
        self.layer2.zerograd()
        self.layer1.zerograd()
        self.layer0.zerograd()

    def loadparams(self, Ws, bs):
        self.layer0.loadparams(Ws[0], bs[0])
        self.layer1.loadparams(Ws[1], bs[1])
        self.layer2.loadparams(Ws[2], bs[2])

    def getWs(self):
        return [self.layer0.getW(), self.layer1.getW(), self.layer2.getW()]

    def getbs(self):
        return [self.layer0.getb(), self.layer1.getb(), self.layer2.getb()]


if __name__ == '__main__':
    """
    You can implement your training and testing loop here.
    You MUST use your class implementations to train the model and to get the results.
    DO NOT use pytorch or tensorflow get the results. The results generated using these
    libraries will be different as compared to your implementation.
    """
    pass