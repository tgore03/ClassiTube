import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


import time
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras import optimizers




LOG=True

#Read Data
data = pd.read_csv('USvideos_modified.csv')
if LOG:
    print("Columns in Data file:\n",data.dtypes)
    print()

#Delete unnecessary columns
columns=['video_id', 'last_trending_date', 'publish_date', 'publish_hour',
         'views', 'likes', 'dislikes', 'comment_count', 'comments_disabled',
         'ratings_disabled', 'tag_appeared_in_title_count', 'tag_appeared_in_title',
         'trend_day_count', 'trend.publish.diff', 'trend_tag_highest',
         'trend_tag_total','subscriber']

data = data.drop(columns, axis=1)
if LOG:
    print("Remaining Columns after deletion:\n",data.dtypes,"\n")


#Extract Labels from data and convert data to Numpy Matrix
labels = np.array(data['category_id'].tolist())   
data = data.as_matrix(columns = ['channel_title', 'title', 'tags', 'description', 'tags_count'])
data = data.astype(str)

if LOG:
    print("After label data split")
    print("Shape of Data:",data.shape)
    print("Shape of Labels:",labels.shape)
    print()

#Preprocess the data
for i in range(len(data)):
    data[i,2] = data[i,2].replace('|', ' ')

startTime = time.clock()

#Maps the values in label to range(1, 16)
label_values = set(labels)
label_values = list(label_values)

if LOG:
    print("Uniques values in label:",label_values)
    print("No of Unique values:",len(label_values))

for i in range(len(labels)):
    labels[i] = label_values.index(labels[i])

if LOG:
    print("After Mapping labels to positions of its unique values")
    print("labels: ", labels)
    print()


#TFIDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data[:,2])
if LOG:
    print("TFIDF vectors:")
    print("Vector shape:", X.shape)
    print()

X = X.todense()
if LOG:
    print("After Dense operation")
    print("Vector Shape:", X.shape)
    print()

#PCA
pca = PCA(0.95)
pca.fit(X)
pca.fit_transform(X)

if LOG:
    print("PCA")
    print("Explained Variance:", pca.explained_variance_)
    print("Explained Variance Ratio Sum:", pca.explained_variance_ratio_.sum())
    print("PCA Transformed Data Shape:\n",X.shape)
    print()


#Split dataset to train and test data
X_train, X_test, y_train, y_test = train_test_split(X, labels.T, test_size=0.25, random_state=42)
if LOG:
    print("After train test Split")
    print("X_train Shape:", X_train.shape)
    print("y_train Shape:", y_train.shape)
    print("X_test Shape:", X_test.shape)
    print("y_test Shape:", y_test.shape)
    print()
    
# 1-of-c output encoding
Y_train = np_utils.to_categorical(y_train)
if LOG:
    print("After 1 to n Encoding\n Y_train Shape: ",Y_train.shape)
    print()

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
if LOG:
    print("len(A):", len(X), "len(A[1]:", len(X[1]))
def neural_network_model(hidden_units, error_function, data, label):

    model = Sequential()
    model.add(Dense(neurons, input_dim=len(X[0]), activation=hidden_units))  # First hidden layer
    model.add(Dense(neurons, activation=hidden_units))  # Second hidden layer
    #model.add(Dense(neurons, activation=hidden_units))  # Third hidden layer
    model.add(Dense(16, activation='softmax'))  # Softmax function for output layer

    # Stochastic Gradient Descent for Optimization
    sgd = optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=momentum_rate, nesterov=True)

    # Compile & Fit model
    model.compile(loss=error_function, optimizer=sgd, metrics=['accuracy'])
    model.fit(X_train, Y_train, validation_split=0.2, epochs=10, batch_size=100, verbose=0, callbacks=callbacks_list)
    endTime = time.clock()
    print("\nTime to build the model = ", endTime-startTime)
   
    print_statistics(model, data, label)
    print_statistics(model, X_test, y_test)


def print_statistics(model, x_data, y_data):
    if LOG:
        print("Print Statistics")
        print("x_data.shape:", x_data.shape)
        print("y_data.shape:", y_data.shape)
        
    predictions = model.predict(x_data)
    if LOG:
        print("predictions shape:", predictions.shape)
    
    fin_prediction = np.empty(shape = y_data.shape)
    i=0
    for row in predictions:
        fin_prediction[i] = np.argmax(row)
        print(fin_prediction[i], y_data[i])
        i+=1

    if LOG:    
        print("fin_prediction shape:", fin_prediction.shape)
        print("fin_predictions:\n", fin_prediction)
        print()
    
    matrix = confusion_matrix(y_data, fin_prediction)
    if LOG:
        print('confusion matrix shape(',len(matrix), len(matrix[0]),')')
        #rint('len(matrix)', len(matrix), 'len(matrix[0])', len(matrix[0]))
        
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
