import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras import optimizers

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import confusion_matrix


#Read Data

#data = np.loadtxt('USvideos_modified.csv', dtype=delimiter=',', usecols=(15), skiprows=1)

data = pd.read_csv('USvideos_modified.csv')
X_names = ['tags']
Y_names = ['category_id'] 
for column in X_names:
    data[column] = data[column].astype(str)

X1 = data[X_names]
Y1 = data[Y_names]

#Preprocess the data
for i, row in X1.iterrows():
    row['tags'] = row['tags'].replace('|', ' ')

#TFIDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X1['tags'])
X = X.todense()
print(X.shape)


#PCA
pca = PCA(0.95)
pca.fit(X)
print(pca.explained_variance_ratio_.sum())

A = pca.transform(X)
print(A.shape)



#Perform Neural Networks

X_train, X_test, y_train, y_test = train_test_split(A, Y1, test_size=0.25, random_state=42)

# 1-of-c output encoding
Y_train = np_utils.to_categorical(y_train)
print("Y_train: ",Y_train.shape)

# define early stopping callback
earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5, \
                          verbose=1, mode='auto')
callbacks_list = [earlystop]

# Hyper-Parameters
momentum_rate = 0.9
filters = 1024
epochs = 10
batch_size = 100
learning_rate = 0.05
neurons = 3000


# Neural Network Model
def neural_network_model(hidden_units, error_function, data, label):
    startTime = time.clock()
    model = Sequential()
    model.add(Dense(neurons, input_dim=len(A[0]), activation=hidden_units))  # First hidden layer
    model.add(Dense(neurons, activation=hidden_units))  # Second hidden layer
    #model.add(Dense(neurons, activation=hidden_units))  # Third hidden layer
    model.add(Dense(44, activation='softmax'))  # Softmax function for output layer

    # Stochastic Gradient Descent for Optimization
    sgd = optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=momentum_rate, nesterov=True)

    # Compile & Fit model
    model.compile(loss=error_function, optimizer=sgd, metrics=['accuracy'])
    model.fit(X_train, Y_train, validation_split=0.2, epochs=10, batch_size=100, verbose=0, callbacks=callbacks_list,)
    endTime = time.clock()
    print("\nTime to build the model = ", endTime-startTime)
    print_statistics(model, data, label)

def print_statistics(model, x_data, y_data):
    predictions = model.predict(x_data)
    fin_prediction = []
    for row in predictions:
        fin_prediction.append(np.argmax(row))
    matrix = confusion_matrix(y_data, fin_prediction)
    print(matrix.shape, fin_prediction.shape)
    sum = 0
    print("\nClass Accuracies:")
    for i in range(len(matrix)):
        sum += matrix[i][i]
        print("Class ", i, ": ", round(matrix[i][i]/np.sum(matrix[i]), 4))
    print("\nOverall Accuracy: ", round(sum/np.sum(matrix), 4))
    print("\nConfusion Matrix:\n", matrix)


print("\nTraining Data Statistics:\n")
print("Relu Hidden Units with Cross-Entropy Error Function")
neural_network_model('relu', 'categorical_crossentropy', X_train, y_train)
print("\n--------------------------------------------------------------")




##non_bmp_map = dict.fromkeys(range(0x10000, sys.maxunicode + 1), 0xfffd)
##l = np.empty(shape=[len(data),], dtype='str')
##j=0
##for i,row in data.iterrows():
##    row['title'] = row['title'].translate(non_bmp_map)
