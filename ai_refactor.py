# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import tensorflow as tf
from os.path import isfile, isdir, exists
from scipy import io as spio
from sklearn.model_selection import train_test_split

def preprocessing(data):
    '''
    min-max scaling for every image
    data: numpy array
    output: scaled numpy array
    '''
    minV = 0
    maxV = 255
    data = (data - minV) / (maxV-minV)
    return data


def one_hot_encoding(data, numberOfClass):
    from sklearn import preprocessing
    lb = preprocessing.LabelBinarizer()
    lb.fit(range(numberOfClass))
    return lb.transform(data)


def getInputData():
    # save in pickle
    fileName = 'emnist.p'

    if exists(fileName):
        # load pickle file
        trainData, trainLabel, one_hot_trainLabel, testData = pickle.load(open(fileName, mode = 'rb'))
        return trainData, trainLabel, one_hot_trainLabel, testData
    else:
        emnist = spio.loadmat("data/emnist-digits.mat")

        # load training dataset
        x_train = emnist["dataset"][0][0][0][0][0][0]
        x_train = x_train.astype(np.float32)

        # load training labels
        y_train = emnist["dataset"][0][0][0][0][0][1]
        # load test dataset
        x_test = emnist["dataset"][0][0][1][0][0][0]
        x_test = x_test.astype(np.float32)

        # load test labels
        y_test = emnist["dataset"][0][0][1][0][0][1]

        train_labels = y_train
        test_labels = y_test

        x_train = x_train.reshape(x_train.shape[0], 1, 28, 28, order="A")
        x_test = x_test.reshape(x_test.shape[0], 1, 28, 28, order="A")

        x_train = x_train.reshape(x_train.shape[0], 28*28)
        x_test = x_test.reshape(x_test.shape[0], 28*28)

        train = pd.DataFrame(y_train, columns= ["label"]).join(pd.DataFrame(x_train))
        test = pd.DataFrame(y_test, columns= ["label"]).join(pd.DataFrame(x_test))

        # cast to numpy array
        trainData = train.values[:,1:]
        trainLabel = train.values[:,0]
        testData = x_test

        processedTrainData = preprocessing(trainData)
        processedTestData = preprocessing(testData)
        one_hot_trainLabel = one_hot_encoding(trainLabel, 10)
        
        # save data to pickle
        if not isfile(fileName):
            pickle.dump((processedTrainData, trainLabel, one_hot_trainLabel, processedTestData), open(fileName, 'wb'))
            return processedTrainData, trainLabel, one_hot_trainLabel, processedTestData

        return None


def getInputTensor(features, numberOfClass):
    '''
    Create tf.placeholder for input & label
    '''
    print(features)
    inputT = tf.placeholder(dtype = tf.float32, shape = (None, features), name = 'input')
    labelT = tf.placeholder(dtype = tf.float32, shape = (None, numberOfClass), name = 'label')
    keep_prob = tf.placeholder(dtype = tf.float32)
    
    return inputT, labelT, keep_prob


def hiddenLayer(inputT, numberOfNodes):
    '''
    Create hidden layer
    '''
    inputSize = inputT.get_shape().as_list()[1]
    # create weights & biases
    weights = tf.Variable(tf.truncated_normal((inputSize, numberOfNodes)), dtype = tf.float32)
    biases = tf.zeros((numberOfNodes), dtype = tf.float32)
    # output of hidden nodes
    hiddenNodes = tf.add(tf.matmul(inputT, weights), biases)
    hiddenOutput = tf.nn.sigmoid(hiddenNodes)
    
    return hiddenOutput


def outputLayer(hiddenOutput, numberOfClass):
    '''
    Create output layer
    hiddenOutput: output from hidden layer
    numOfClass: number of classes (0~9)
    '''
    inputSize = hiddenOutput.get_shape().as_list()[1]
    # create weights & biases
    weights = tf.Variable(tf.truncated_normal((inputSize, numberOfClass)), dtype = tf.float32)
    biases = tf.zeros((numberOfClass), dtype = tf.float32)
    # output
    output = tf.add(tf.matmul(hiddenOutput, weights), biases)
    
    return output


def build_nn(inputT, numberOfNodes, numberOfClass, keep_prob):
    '''
    build fully connected neural network
    '''
    # fully_connected layers
    fc1 = hiddenLayer(inputT, numberOfNodes)
    fc2 = hiddenLayer(fc1,numberOfNodes)
    output = outputLayer(fc2, numberOfClass)
    
    return output


trainData, trainLabel, one_hot_trainLabel, testData = getInputData()
numberOfNodes = 256
batchSize = 128
numberOfEpoch = 20
learningRate = 0.01
keep_prob_rate = 1.0
numberOfClass = 10
imageSize = (28, 28)
graph = tf.Graph()


def printResult(epoch, numberOfEpoch, trainLoss, validationLoss, validationAccuracy):
    print("Epoch: {}/{}".format(epoch+1, numberOfEpoch),
         '\tTraining Loss: {:.3f}'.format(trainLoss),
         '\tValidation Loss: {:.3f}'.format(validationLoss),
         '\tAccuracy: {:.2f}%'.format(validationAccuracy*100))


def train():
    features = np.prod(imageSize)
    tf.reset_default_graph()
    with graph.as_default():
        
        # get inputs
        inputT, labelT, keep_prob = getInputTensor(features, numberOfClass)
        
        # build fully-conneted neural network
        logits = build_nn(inputT, numberOfNodes, numberOfClass, keep_prob)
        
        # softmax with cross entropy
        probability = tf.nn.softmax(logits, name = 'probability')
        
        # Cost
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = labelT))
        
        # optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate = learningRate).minimize(cost)
        
        # accuracy
        correctPrediction = tf.equal(tf.argmax(probability, 1),tf.argmax(labelT, 1))
        accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

    save_dir = './saveEmnist'
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(numberOfEpoch):
            # training data & validation data
            train_x, val_x, train_y, val_y = train_test_split(trainData, one_hot_trainLabel, test_size = 0.2)   
            # training loss
            for i in range(0, len(train_x), batchSize):
                trainLoss, _, _ = sess.run([cost, probability, optimizer], feed_dict = {
                    inputT: train_x[i: i+batchSize],
                    labelT: train_y[i: i+batchSize],
                    keep_prob: keep_prob_rate   
                })
                
            # validation loss
            valAcc, valLoss = sess.run([accuracy, cost], feed_dict ={
                inputT: val_x,
                labelT: val_y,
                keep_prob: 1.0
            })
            
            
            # print out
            printResult(epoch, numberOfEpoch, trainLoss, valLoss, valAcc)
        # save
        saver = tf.train.Saver()
        saver.save(sess, save_dir)

# test result
def test():
    print(testData.shape)
    save_dir = './saveEmnist'

    loaded_Graph = tf.Graph()
    with tf.Session(graph=loaded_Graph) as sess:
        loader = tf.train.import_meta_graph(save_dir +'.meta')
        loader.restore(sess, save_dir)    
        # get tensors
        loaded_x = loaded_Graph.get_tensor_by_name('input:0')
        loaded_y = loaded_Graph.get_tensor_by_name('label:0')
        loaded_prob = loaded_Graph.get_tensor_by_name('probability:0')
        
        prob = sess.run(tf.argmax(loaded_prob,1), feed_dict = {loaded_x: testData})


    which = 1
    print('predicted labe: {}'.format(str(prob[which])))

    count_right = 0
    count_wrong = 0
    count_total = 0

    for i,p in enumerate(prob):
        if p == test_labels[i][0]:
            count_right += 1
        else:
            count_wrong += 1
        count_total += 1

    print("Correct:", count_right)
    print("Wrong:", count_wrong)
    print("Total:", count_total)
    print("Ratio:", count_right/count_total)

train()
test()