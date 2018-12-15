# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import tensorflow as tf
from os.path import isfile, isdir, exists
from scipy import io as spio
import time
from sklearn.model_selection import train_test_split

class AI():

    def __init__(self, numberOfNodes = 256, numberOfEpoch = 20, save_dir = "./save_emnist"):
        self.trainData, self.trainLabel, self.one_hot_trainLabel, self.testData, self.test_labels = self.getInputData()
        self.numberOfNodes = numberOfNodes
        self.batchSize = 128
        self.numberOfEpoch = numberOfEpoch
        self.learningRate = 0.01
        self.keep_prob_rate = 1.0
        self.numberOfClass = 10
        self.imageSize = (28, 28)
        self.graph = tf.Graph()
        self.save_dir = save_dir
        print("Epoches: {}, Number of nodes each hidden layer: {}".format(numberOfEpoch, numberOfNodes))

    def preprocessing(self, data):
        '''
        min-max scaling for every image
        data: numpy array
        output: scaled numpy array
        '''
        minV = 0
        maxV = 255
        data = (data - minV) / (maxV-minV)
        return data


    def one_hot_encoding(self, data, numberOfClass):
        from sklearn import preprocessing
        lb = preprocessing.LabelBinarizer()
        lb.fit(range(numberOfClass))
        return lb.transform(data)

    def getInputTensor(self, features, numberOfClass):
        '''
        Create tf.placeholder for input & label
        '''
        inputT = tf.placeholder(dtype = tf.float32, shape = (None, features), name = 'input')
        labelT = tf.placeholder(dtype = tf.float32, shape = (None, numberOfClass), name = 'label')
        keep_prob = tf.placeholder(dtype = tf.float32)
        
        return inputT, labelT, keep_prob


    def hiddenLayer(self, inputT, numberOfNodes):
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


    def outputLayer(self, hiddenOutput, numberOfClass):
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


    def build_nn(self, inputT, numberOfNodes, numberOfClass, keep_prob):
        '''
        build fully connected neural network
        '''
        # fully_connected layers
        fc1 = self.hiddenLayer(inputT, numberOfNodes)
        fc2 = self.hiddenLayer(fc1, numberOfNodes)
        
        output = self.outputLayer(fc2, numberOfClass)
        
        return output


    def printResult(self, epoch, numberOfEpoch, trainLoss, validationLoss, validationAccuracy):
        print("Epoch: {}/{}".format(epoch+1, numberOfEpoch),
            '\tTraining Loss: {:.3f}'.format(trainLoss),
            '\tValidation Loss: {:.3f}'.format(validationLoss),
            '\tAccuracy: {:.2f}%'.format(validationAccuracy*100))


    def getInputData(self):
        # save in pickle
        fileName = 'emnist.p'

        if exists(fileName):
            # load pickle file
            trainData, trainLabel, one_hot_trainLabel, testData, test_labels = pickle.load(open(fileName, mode = 'rb'))
            return trainData, trainLabel, one_hot_trainLabel, testData, test_labels
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

            processedTrainData = self.preprocessing(trainData)
            processedTestData = self.preprocessing(testData)
            one_hot_trainLabel = self.one_hot_encoding(trainLabel, 10)
            
            # save data to pickle
            if not isfile(fileName):
                pickle.dump((processedTrainData, trainLabel, one_hot_trainLabel, processedTestData, test_labels), open(fileName, 'wb'))
                return processedTrainData, trainLabel, one_hot_trainLabel, processedTestData, test_labels

            return None


    def train(self):
        features = np.prod(self.imageSize)
        tf.reset_default_graph()
        with self.graph.as_default():
            
            # get inputs
            self.inputT, self.labelT, self.keep_prob = self.getInputTensor(features, self.numberOfClass)
            
            # build fully-conneted neural network
            self.logits = self.build_nn(self.inputT, self.numberOfNodes, self.numberOfClass, self.keep_prob)
            
            # softmax with cross entropy
            self.probability = tf.nn.softmax(self.logits, name = 'probability')
            
            # Cost
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.logits, labels = self.labelT))
            # optimizer
            self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learningRate).minimize(self.cost)
            
            # accuracy
            self.correctPrediction = tf.equal(tf.argmax(self.probability, 1),tf.argmax(self.labelT, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correctPrediction, tf.float32))

        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            print("Start train")
            t0 = time.time()
           
            for epoch in range(self.numberOfEpoch):
                # training data & validation data
                train_x, val_x, train_y, val_y = train_test_split(self.trainData, self.one_hot_trainLabel, test_size = 0.2)   
                # training loss
                for i in range(0, len(train_x), self.batchSize):
                    trainLoss, _, _ = sess.run([self.cost, self.probability, self.optimizer], feed_dict = {
                        self.inputT: train_x[i: i + self.batchSize],
                        self.labelT: train_y[i: i + self.batchSize],
                        self.keep_prob: self.keep_prob_rate   
                    })
                    
                # validation loss
                valAcc, valLoss = sess.run([self.accuracy, self.cost], feed_dict ={
                    self.inputT: val_x,
                    self.labelT: val_y,
                    self.keep_prob: 1.0
                })
                
                # print out
                self.printResult(epoch, self.numberOfEpoch, trainLoss, valLoss, valAcc)
            # save
            print("Train finished")
            print ("training time:", round(time.time()-t0, 3), "s")
            saver = tf.train.Saver()
            saver.save(sess, self.save_dir)

    # test result
    def test(self):

        loaded_Graph = tf.Graph()
        with tf.Session(graph=loaded_Graph) as sess:
            loader = tf.train.import_meta_graph(self.save_dir +'.meta')
            loader.restore(sess, self.save_dir)    
            # get tensors
            loaded_x = loaded_Graph.get_tensor_by_name('input:0')
            loaded_y = loaded_Graph.get_tensor_by_name('label:0')
            loaded_prob = loaded_Graph.get_tensor_by_name('probability:0')
            
            prob = sess.run(tf.argmax(loaded_prob,1), feed_dict = {loaded_x: self.testData})

        count_right = 0
        count_wrong = 0
        count_total = 0

        for i,p in enumerate(prob):
            if p == self.test_labels[i][0]:
                count_right += 1
            else:
                count_wrong += 1
            count_total += 1

        print("Correct:", count_right)
        print("Wrong:", count_wrong)
        print("Total:", count_total)
        print("Ratio:", count_right/count_total)


AI1 = AI(save_dir="./ai1")
AI1.train()
AI1.test()

AI2 = AI(save_dir="./ai2", numberOfNodes=128)
AI2.train()
AI2.test()