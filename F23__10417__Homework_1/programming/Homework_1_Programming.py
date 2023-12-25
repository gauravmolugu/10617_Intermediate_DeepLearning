import pickle as pk
import numpy as np
import mlp
from tqdm import tqdm

import pickle as pk
with open('/Users/gauravmolugu/Downloads/10617/omniglot_12.pkl','rb') as f: 
    data = pk.load(f) 
    ((trainX,trainY),(testX,testY)) = data # dimensions of trainX (6660, 105*105) and trainY (6660,)

import random
from matplotlib import pyplot as plt

'''
Plots of few examples from the dataset

for _ in range(10):
    img = trainX[random.randint(0,6660)]
    plt.imshow(img.reshape(105,105))
    plt.show()

'''
# np.random.seed(0)

#---------------------------------------------------------------------------------------------------#

def get_minibatch(trainX, trainY, batch_size):

    num_samples = trainX.shape[0]
    # Shuffle the data
    permutation = np.random.permutation(num_samples)
    shuffled_trainX = trainX[permutation]
    trainY = mlp.labels2onehot(trainY)
    shuffled_trainY = trainY[permutation]

    minibatches = []

    num_batches = num_samples // batch_size

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size

        batch_X = shuffled_trainX[start_idx:end_idx]
        batch_Y = shuffled_trainY[start_idx:end_idx]

        minibatches.append((batch_X.T, batch_Y.T))

    return minibatches

def initialize_params(model):
    indim, outdim, hidden_layers = model.indim, model.outdim, model.layers

    Ws, bs = [], []
    prev_out, curr_out = indim, hidden_layers[0]
    for l in range(0, len(hidden_layers)):
        W_layer, b_layer = mlp.random_weight_init(prev_out, curr_out), mlp.zeros_bias_init(curr_out)
        Ws.append(W_layer)
        bs.append(b_layer)

        prev_out = curr_out
        curr_out = hidden_layers[l]

    W_last, b_last = mlp.random_weight_init(curr_out, outdim), mlp.zeros_bias_init(outdim)
    Ws.append(W_last)
    bs.append(b_last)
    return Ws, bs

def train_MLP(model, loss_func, trainX, trainY, testX, testY, epochs, batch_size = 128):
    loss_train_epoch, accuracy_train_epoch = [],[]
    loss_test_epoch, accuracy_test_epoch = [],[]

    for epoch in tqdm(range(1, epochs+1)):
        train_loss, train_acc = [], []

        minibatches = get_minibatch(trainX, trainY, batch_size)
        
        for data, label in minibatches: 
            model.zerograd()
            model_out = model.forward(data)
            loss_tr = loss_func.forward(model_out, label)
            grad_wrt_loss = loss_func.backward()
            grad_wrt_input = model.backward(grad_wrt_loss)

            model.step()

            train_loss.append(loss_tr)
            acc_tr = loss_func.getAccu()
            train_acc.append(acc_tr)

        model_out_test = model.forward(testX.T)
        test_label = mlp.labels2onehot(testY)
        loss_te = loss_func.forward(model_out_test, test_label.T)
        acc_te = loss_func.getAccu()
        
        loss_train_epoch.append(np.mean(train_loss))
        loss_test_epoch.append(loss_te)
        accuracy_train_epoch.append(np.mean(train_acc))
        accuracy_test_epoch.append(acc_te)

    return loss_train_epoch, loss_test_epoch, accuracy_train_epoch, accuracy_test_epoch

def plot_loss(ax_loss, ax_acc, loss_train, loss_test, accuracy_train, accuracy_test, loss_title, acc_title, problem_number,label):
    
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

if __name__ == "__main__":

    # Question 2.1

    model_1 = mlp.SingleLayerMLP(indim = 11025, outdim = 12, hiddenlayer = 60, alpha = 0, dropout_probability = 0, lr = 0.001)
    Ws, bs = initialize_params(model_1)
    model_1.loadparams(Ws, bs)

    loss_func = mlp.SoftmaxCrossEntropyLoss()
    loss_train, loss_test, acc_train, acc_test = train_MLP(model_1, loss_func, trainX, trainY, testX, testY, 50, 128)

    fig_loss, ax_loss = plt.subplots()
    fig_acc, ax_acc = plt.subplots()

    plot_loss(ax_loss, ax_acc, loss_train, loss_test, acc_train, acc_test, 
            loss_title="Loss vs Epochs - SingleLayer", acc_title="Accuracy vs Epochs - Single Layer",
            problem_number="2.1", label = "60,0,0,0.001")
    #
    model_2 = mlp.SingleLayerMLP(indim = 11025, outdim = 12, hiddenlayer = 60, alpha = 0.4, dropout_probability = 0, lr = 0.001)
    Ws, bs = initialize_params(model_2)
    model_2.loadparams(Ws, bs)

    loss_func = mlp.SoftmaxCrossEntropyLoss()
    loss_train, loss_test, acc_train, acc_test = train_MLP(model_2, loss_func, trainX, trainY, testX, testY, 200, 128)

    plot_loss(ax_loss, ax_acc, loss_train, loss_test, acc_train, acc_test, 
            loss_title="Loss vs Epochs - SingleLayer", acc_title="Accuracy vs Epochs - Single Layer",
            problem_number="2.1", label = "60,0.4,0,0.001")
    #
    model_3 = mlp.SingleLayerMLP(indim = 11025, outdim = 12, hiddenlayer = 60, alpha = 0.4, dropout_probability = 0.2, lr = 0.001)
    Ws, bs = initialize_params(model_3)
    model_3.loadparams(Ws, bs)

    loss_func = mlp.SoftmaxCrossEntropyLoss()
    loss_train, loss_test, acc_train, acc_test = train_MLP(model_3, loss_func, trainX, trainY, testX, testY, 200, 128)

    plot_loss(ax_loss, ax_acc, loss_train, loss_test, acc_train, acc_test, 
            loss_title="Loss vs Epochs - SingleLayer", acc_title="Accuracy vs Epochs - Single Layer",
            problem_number="2.1", label = "60,0.4,0.2,0.001")
    #
    model_4 = mlp.SingleLayerMLP(indim = 11025, outdim = 12, hiddenlayer = 150, alpha = 0.4, dropout_probability = 0.2, lr = 0.001)
    Ws, bs = initialize_params(model_4)
    model_4.loadparams(Ws, bs)

    loss_func = mlp.SoftmaxCrossEntropyLoss()
    loss_train, loss_test, acc_train, acc_test = train_MLP(model_4, loss_func, trainX, trainY, testX, testY, 200, 128)

    plot_loss(ax_loss, ax_acc, loss_train, loss_test, acc_train, acc_test, 
            loss_title="Loss vs Epochs - SingleLayer", acc_title="Accuracy vs Epochs - Single Layer",
            problem_number="2.1", label = "150,0.4,0.2,0.001")

    plt.show()
    
    #----------------------------------------------------------------------------------------------------------------------------#

    # Question 2.2

    model_2layer_1 = mlp.TwoLayerMLP(indim = 11025, outdim = 12, hiddenlayers = [60, 60], alpha = 0, dropout_probability = 0, lr = 0.001)
    Ws, bs = initialize_params(model_2layer_1)
    model_2layer_1.loadparams(Ws, bs)

    loss_func = mlp.SoftmaxCrossEntropyLoss()
    loss_train, loss_test, acc_train, acc_test = train_MLP(model_2layer_1, loss_func, trainX, trainY, testX, testY, 200, 128)

    fig_2layer_loss, ax_2layer_loss = plt.subplots()
    fig_2layer_acc, ax_2layer_acc = plt.subplots()

    plot_loss(ax_2layer_loss, ax_2layer_acc, loss_train, loss_test, acc_train, acc_test, 
            loss_title="Loss vs Epochs - TwoLayer", acc_title="Accuracy vs Epochs - Two Layer",
            problem_number="2.2", label = "(60,60),0,0,0.001")    
    
    model_2layer_2 = mlp.TwoLayerMLP(indim = 11025, outdim = 12, hiddenlayers = [60, 60], alpha = 0.4, dropout_probability = 0, lr = 0.001)
    Ws, bs = initialize_params(model_2layer_1)
    model_2layer_2.loadparams(Ws, bs)

    loss_func = mlp.SoftmaxCrossEntropyLoss()
    loss_train, loss_test, acc_train, acc_test = train_MLP(model_2layer_2, loss_func, trainX, trainY, testX, testY, 200, 128)

    plot_loss(ax_2layer_loss, ax_2layer_acc, loss_train, loss_test, acc_train, acc_test, 
            loss_title="Loss vs Epochs - TwoLayer", acc_title="Accuracy vs Epochs - Two Layer",
            problem_number="2.2", label = "(60,60),0.4,0,0.001")    
    
    model_2layer_3 = mlp.TwoLayerMLP(indim = 11025, outdim = 12, hiddenlayers = [60, 60], alpha = 0.4, dropout_probability = 0.2, lr = 0.001)
    Ws, bs = initialize_params(model_2layer_3)
    model_2layer_3.loadparams(Ws, bs)

    loss_func = mlp.SoftmaxCrossEntropyLoss()
    loss_train, loss_test, acc_train, acc_test = train_MLP(model_2layer_3, loss_func, trainX, trainY, testX, testY, 200, 128)

    plot_loss(ax_2layer_loss, ax_2layer_acc, loss_train, loss_test, acc_train, acc_test, 
            loss_title="Loss vs Epochs - TwoLayer", acc_title="Accuracy vs Epochs - Two Layer",
            problem_number="2.2", label = "(60,60),0.4,0.2,0.001")    
    
    model_2layer_4 = mlp.TwoLayerMLP(indim = 11025, outdim = 12, hiddenlayers = [150, 150], alpha = 0.4, dropout_probability = 0.2, lr = 0.001)
    Ws, bs = initialize_params(model_2layer_4)
    model_4.loadparams(Ws, bs)

    loss_func = mlp.SoftmaxCrossEntropyLoss()
    loss_train, loss_test, acc_train, acc_test = train_MLP(model_2layer_4, loss_func, trainX, trainY, testX, testY, 200, 128)

    plot_loss(ax_2layer_loss, ax_2layer_acc, loss_train, loss_test, acc_train, acc_test, 
            loss_title="Loss vs Epochs - TwoLayer", acc_title="Accuracy vs Epochs - Two Layer",
            problem_number="2.2", label = "(150,150),0.4,0.2,0.001")    
    
    plt.show()
    

    #----------------------------------------------------------------------------------------------------------------------------#

    # Question 2.3

    model_3layer_1 = mlp.TwoLayerMLP(indim = 11025, outdim = 12, hiddenlayers = [150, 150], alpha = 0, dropout_probability = 0, lr = 0.01)
    Ws, bs = initialize_params(model_3layer_1)
    model_3layer_1.loadparams(Ws, bs)

    loss_func = mlp.SoftmaxCrossEntropyLoss()
    loss_train, loss_test, acc_train, acc_test = train_MLP(model_3layer_1, loss_func, trainX, trainY, testX, testY, 200, 128)

    fig_2layer_loss, ax_lr_loss = plt.subplots()
    fig_2layer_acc, ax_lr_acc = plt.subplots()

    plot_loss(ax_lr_loss, ax_lr_acc, loss_train, loss_test, acc_train, acc_test, 
            loss_title="Loss vs Epochs - TwoLayer", acc_title="Accuracy vs Epochs - Two Layer",
            problem_number="2.3", label = "(150,150),0,0,0.01")
    #
    model_3layer_2 = mlp.TwoLayerMLP(indim = 11025, outdim = 12, hiddenlayers = [150, 150], alpha = 0, dropout_probability = 0, lr = 0.001)
    Ws, bs = initialize_params(model_3layer_2)
    model_3layer_2.loadparams(Ws, bs)

    loss_func = mlp.SoftmaxCrossEntropyLoss()
    loss_train, loss_test, acc_train, acc_test = train_MLP(model_3layer_2, loss_func, trainX, trainY, testX, testY, 200, 128)

    plot_loss(ax_lr_loss, ax_lr_acc, loss_train, loss_test, acc_train, acc_test, 
            loss_title="Loss vs Epochs - TwoLayer", acc_title="Accuracy vs Epochs - Two Layer",
            problem_number="2.3", label = "(150,150),0,0,0.001")
    #
    model_3layer_3 = mlp.TwoLayerMLP(indim = 11025, outdim = 12, hiddenlayers = [150, 150], alpha = 0, dropout_probability = 0, lr = 0.0001)
    Ws, bs = initialize_params(model_3layer_3)
    model_3layer_3.loadparams(Ws, bs)

    loss_func = mlp.SoftmaxCrossEntropyLoss()
    loss_train, loss_test, acc_train, acc_test = train_MLP(model_3layer_3, loss_func, trainX, trainY, testX, testY, 200, 128)

    plot_loss(ax_lr_loss, ax_lr_acc, loss_train, loss_test, acc_train, acc_test, 
            loss_title="Loss vs Epochs - TwoLayer", acc_title="Accuracy vs Epochs - Two Layer",
            problem_number="2.3", label = "(150,150),0,0,0.0001")
    
    plt.show()

    #----------------------------------------------------------------------------------------------------------------------------#
    
    # Question 2.4

    # Effect of momentum on loss and accuracy - 0, 0.3, 0.6

    model_4layer_1 = mlp.TwoLayerMLP(indim = 11025, outdim = 12, hiddenlayers = [150, 150], alpha = 0, dropout_probability = 0, lr = 0.001)
    Ws, bs = initialize_params(model_4layer_1)
    model_4layer_1.loadparams(Ws, bs)

    loss_func = mlp.SoftmaxCrossEntropyLoss()
    loss_train, loss_test, acc_train, acc_test = train_MLP(model_4layer_1, loss_func, trainX, trainY, testX, testY, 200, 128)

    fig_2layer_loss, ax_lr_loss = plt.subplots()
    fig_2layer_acc, ax_lr_acc = plt.subplots()

    plot_loss(ax_lr_loss, ax_lr_acc, loss_train, loss_test, acc_train, acc_test,
            loss_title="Loss vs Epochs - TwoLayer", acc_title="Accuracy vs Epochs - Two Layer",
            problem_number="2.4", label = "(150,150),0,0,0.001")
    #
    model_4layer_2 = mlp.TwoLayerMLP(indim = 11025, outdim = 12, hiddenlayers = [150, 150], alpha = 0.3, dropout_probability = 0, lr = 0.001)
    Ws, bs = initialize_params(model_4layer_2)
    model_4layer_2.loadparams(Ws, bs)

    loss_func = mlp.SoftmaxCrossEntropyLoss()
    loss_train, loss_test, acc_train, acc_test = train_MLP(model_4layer_2, loss_func, trainX, trainY, testX, testY, 200, 128)

    plot_loss(ax_lr_loss, ax_lr_acc, loss_train, loss_test, acc_train, acc_test,
            loss_title="Loss vs Epochs - TwoLayer", acc_title="Accuracy vs Epochs - Two Layer",
            problem_number="2.4", label = "(150,150),0.3,0,0.001")
    #
    model_4layer_3 = mlp.TwoLayerMLP(indim = 11025, outdim = 12, hiddenlayers = [150, 150], alpha = 0.6, dropout_probability = 0, lr = 0.001)
    Ws, bs = initialize_params(model_4layer_3)
    model_4layer_3.loadparams(Ws, bs)

    loss_func = mlp.SoftmaxCrossEntropyLoss()
    loss_train, loss_test, acc_train, acc_test = train_MLP(model_4layer_3, loss_func, trainX, trainY, testX, testY, 200, 128)

    plot_loss(ax_lr_loss, ax_lr_acc, loss_train, loss_test, acc_train, acc_test,
            loss_title="Loss vs Epochs - TwoLayer", acc_title="Accuracy vs Epochs - Two Layer",
            problem_number="2.4", label = "(150,150),0.6,0,0.001")

    plt.show()
    


