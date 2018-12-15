
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import tensorflow as tf
from os.path import isfile, isdir
import time

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')


# In[2]:


# let's print the shape before we reshape and normalize
print("x_train shape", x_train.shape)
print("y_train shape", y_train.shape)
print("x_test shape", x_test.shape)
print("y_test shape", y_test.shape)

# building the input vector from the 28x28 pixels
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# print the final input shape ready for training
print("Train matrix shape", x_train.shape)
print("Test matrix shape", x_test.shape)


# In[3]:


train = pd.DataFrame(y_train, columns= ["label"]).join(pd.DataFrame(x_train))
test = pd.DataFrame(y_test, columns= ["label"]).join(pd.DataFrame(x_test))


# In[4]:


# check training data
trainLabelCounts = train['label'].value_counts(sort = False)
trainLabelCounts


# In[5]:


# check training data - plot
def getImage(data, *args):
    '''
    Get the image by specified number (Randomly)
    parameters:
        data: dataframe
        number: int, the label of the number to show
    output: 1-D numpy array
    '''
    if args:
        number = args[0]
        specifiedData = data[data['label'] == number].values
    else:
        specifiedData = data.values
    randomNumber = np.random.choice(len(specifiedData)-1, 1)
    return specifiedData[randomNumber,:]
    
def plotNumber(imageData, imageSize):
    '''
    parameters:
        data: label & 1-D array of pixels
    '''
    # show the image of the data
    if imageData.shape[1] == np.prod(imageSize):
        image = imageData[0,:].reshape(imageSize)
    elif imageData.shape[1] > np.prod(imageSize):
        label = imageData[0,0]
        image = imageData[0,1:].reshape(imageSize)
        plt.title('number: {}'.format(label))
    plt.imshow(image)


# In[6]:


# plot the training image with specified number
imageSize = (28, 28)
chosenNumber = 1
plotNumber(getImage(train, chosenNumber), imageSize)


# In[7]:


# check testing data
plotNumber(getImage(test), imageSize)


# In[8]:


# cast to numpy array
trainData = train.values[:,1:]
trainLabel = train.values[:,0]
testData = x_test
print(testData.shape)


# In[9]:


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


# In[10]:


processedTrainData = preprocessing(trainData)
processedTestData = preprocessing(testData)
one_hot_trainLabel = one_hot_encoding(trainLabel, 10)


# In[11]:


# save in pickle
fileName = 'mnist.p'
if not isfile(fileName):
    pickle.dump((processedTrainData, trainLabel, one_hot_trainLabel, processedTestData), open(fileName, 'wb'))


# In[12]:


# load pickle file
fileName = 'mnist.p'
trainData, trainLabel, one_hot_trainLabel, testData = pickle.load(open(fileName, mode = 'rb'))


# In[13]:


def getInputTensor(features, numberOfClass):
    '''
    Create tf.placeholder for input & label
    '''
    print(features)
    inputT = tf.placeholder(dtype = tf.float32, shape = (None, features), name = 'input')
    labelT = tf.placeholder(dtype = tf.float32, shape = (None, numberOfClass), name = 'label')
    keep_prob = tf.placeholder(dtype = tf.float32)
    
    return inputT, labelT, keep_prob


# In[14]:


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


# In[15]:


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


# In[16]:


def build_nn(inputT, numberOfNodes, numberOfClass, keep_prob):
    '''
    build fully connected neural network
    '''
    # fully_connected layers
    fc1 = hiddenLayer(inputT, numberOfNodes)
    fc2 = hiddenLayer(fc1,numberOfNodes)
    output = outputLayer(fc2, numberOfClass)
    
    return output


# In[17]:


numberOfNodes = 256
batchSize = 128
numberOfEpoch = 20
learningRate = 0.01
keep_prob_rate = 1.0


# In[18]:


# Build Neural Network graph
numberOfClass = 10
imageSize = (28, 28)
features = np.prod(imageSize)
graph = tf.Graph()
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


# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


def printResult(epoch, numberOfEpoch, trainLoss, validationLoss, validationAccuracy):
    print("Epoch: {}/{}".format(epoch+1, numberOfEpoch),
         '\tTraining Loss: {:.3f}'.format(trainLoss),
         '\tValidation Loss: {:.3f}'.format(validationLoss),
         '\tAccuracy: {:.2f}%'.format(validationAccuracy*100))


# In[21]:


save_dir = './save'
with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    print("Start train")
    t0 = time.time()
    for epoch in range(numberOfEpoch):
        # training data & validation data
        train_x, val_x, train_y, val_y = train_test_split(trainData, one_hot_trainLabel,                                                      test_size = 0.2)   
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
    print("Train finished")
    print ("training time:", round(time.time()-t0, 3), "s")
    saver = tf.train.Saver()
    saver.save(sess, save_dir)


# In[24]:


# test result

loaded_Graph = tf.Graph()
with tf.Session(graph=loaded_Graph) as sess:
    loader = tf.train.import_meta_graph(save_dir +'.meta')
    loader.restore(sess, save_dir)    
    # get tensors
    loaded_x = loaded_Graph.get_tensor_by_name('input:0')
    loaded_y = loaded_Graph.get_tensor_by_name('label:0')
    loaded_prob = loaded_Graph.get_tensor_by_name('probability:0')
    
    prob = sess.run(tf.argmax(loaded_prob,1), feed_dict = {loaded_x: testData})


count_right = 0
count_wrong = 0
count_total = 0

for i,p in enumerate(prob):
    if p == y_test[i]:
        count_right += 1
    else:
        count_wrong += 1
    count_total += 1

print("Correct:", count_right)
print("Wrong:", count_wrong)
print("Total:", count_total)
print("Ratio:", count_right/count_total)

