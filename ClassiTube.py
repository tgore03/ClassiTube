import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras import optimizers

LOG=False

# Read the input data, store it in numpy array, preprocess it and maps the labels to indexes 
def read_data(file):
    #Read Data
    data = pd.read_csv(file)
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

    #Replace "|" in Tags with " "
    for i in range(len(data)):
        data[i,2] = data[i,2].replace('|', ' ')

    #Maps the values in label to range(1, 16)
    label_values = set(labels)
    label_values = list(label_values)

    if LOG:
        print("Uniques values in label:",label_values)
        print("No of Unique values:",len(label_values))

    y = np.empty(shape=labels.shape)
    for i in range(len(labels)):
        y[i] = label_values.index(labels[i])

    if LOG:
        print("After Mapping labels to positions of its unique values")
        print("labels: ", y)
        print("unique values in labels", set(y))
        print()

    return data,y

#Store X obtained after PCA transformation
def store_instance(filename, X, y, tfidf, pca):
    np.savez(filename, X, y, tfidf.get_params(), pca.get_params())
    print(X.dtype)
    print(X.shape, y.shape)
    print(tfidf.get_params)
    print(pca.get_params)

#Read PCA Transformed X from file
def read_instance(filename):
    X, y, tfidf, pca = np.load(filename)
    tfidf = TfidfVectorizer.set_params(tfidf)
    pca = PCA.set_params(pca)
    print(X.shape, y.shape)
    print(tfidf.get_params)
    print(pca.get_params)
    return X, y, tfidf, pca

# Compute the TF-IDF vectors for data points
def tf_idf(data):
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

    return X, vectorizer

# Perform the PCA on the input data set
def do_pca(X):
    #PCA
    pca = PCA(0.95)
    pca.fit(X)
    X = pca.transform(X)

    if LOG:
        print("PCA")
        print("Explained Variance:", pca.explained_variance_)
        print("Explained Variance Ratio Sum:", pca.explained_variance_ratio_.sum())
        print("PCA Transformed Data Shape:\n",X.shape)
        print()
    return X, pca

# Build a neural network model for input data, validate it and run on test data set
def use_ann(X,y):
    #Perform Neural Networks
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
    neurons = 30
    hidden_units = 'relu'
    error_function = 'categorical_crossentropy'

    # Neural Network Model
    print("Neural Networks")
    model = Sequential()
    model.add(Dense(neurons, input_dim=len(X[0]), activation=hidden_units))  # First hidden layer
    model.add(Dense(neurons, activation=hidden_units))  # Second hidden layer
    #model.add(Dense(neurons, activation=hidden_units))  # Third hidden layer
    model.add(Dense(len(set(y)), activation='softmax'))  # Softmax function for output layer

    # Split dataset to train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y.T, test_size=0.20, random_state=13)

    # Stochastic Gradient Descent for Optimization
    sgd = optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=momentum_rate, nesterov=True)

    # Compile the model
    model.compile(loss=error_function, optimizer=sgd, metrics=['accuracy'])
 
    # 1-of-c output encoding
    Y_train = np_utils.to_categorical(y_train)
    print("Y_train: ",Y_train.shape)
        
    model.fit(X_train, Y_train, validation_split=0.2, epochs=10, batch_size=100, verbose=0, callbacks=callbacks_list) 

    predictions = model.predict(X_test)

    y_pred = decode_output(y_test,predictions)
    #print_statistics(y_train,y_pred)
    print_statistics(y_test,y_pred)
    return model

# decode c dimension output to 1 dimension
def decode_output(y_test,predictions):
    # 1-of-c output decoding
    y_pred = np.empty(shape = y_test.shape)
    i=0
    for row in predictions:
        y_pred[i] = np.argmax(row)
        i+=1
    return y_pred

# print the confusion matrix, class accuracies and overall accuracy
def print_statistics(y_test,y_pred):
    matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    sum = 0
    print("\nClass Accuracies:")
    for i in range(len(matrix)):
        sum += matrix[i][i]
        print("Class ", i, ": ", round(matrix[i][i]/np.sum(matrix[i]), 4))
    print("Confusion Matrix:\n", matrix)
    print("Overall Accuracy:\n", accuracy)
    return accuracy

# Build the K-Nearest Neighbors model
def use_kNN(X,y,l,h):
    # Split dataset to train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y.T, test_size=0.20, random_state=13)
    k_range = range(l,h)
    scores = []
    for k in k_range:
        print("K - Nearest Neighbors")
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = print_statistics(y_test, y_pred)
        scores.append(accuracy)

    return knn, k_range, scores

# plot the kNN accuracy for different values of k
def plot_kNN(k_range, scores):
    # plot the relationship between K and testing accuracy
    # plt.plot(x_axis, y_axis)
    plt.plot(k_range, scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Testing Accuracy')
    #plt.show()
    print()

# Preprocess data and store it in the file
def process_data(f, filename):
    data, y = read_data(f)
    X, tfidf = tf_idf(data)
    X, pca = do_pca(X)
    store_instance(filename, X, y, tfidf, pca)
    
# method to build ANN and kNN models
def build_models(f, X=None):
    if X == None:
        data, y = read_data(f)
        X, tfidf = tf_idf(data)
        X, pca = do_pca(X)

    # ANN
    start_time=time.clock()
    ann = use_ann(X,y)
    print("\nTime to build ANN model = ", time.clock()-start_time)

    # kNN
    kmin = 5
    kmax = 6
    start_time=time.clock()

    knn, k_range, scores = use_kNN(X,y,kmin,kmax)
    print("\nTime to build KNN model = ", time.clock()-start_time)
    plot_kNN(k_range, scores)
    
    return tfidf, pca, ann, knn 

# Test the new data point on already build models
def test_models(tfidf, pca, ann, knn):

    data, y = read_data(file_test)
    #X = tfidf.fit_transform(data[:,2])
    X = tfidf.transform(data[:,2])
    X = X.todense()

    X = pca.transform(X)
    
    # use ANN
    start_time=time.clock()
    prediction = ann.predict(X)
    y_test = y.T
    y_pred = decode_output(y_test,prediction) 
    print_statistics(y_test,y_pred)
    print("\nTime to test ANN model = ", time.clock()-start_time)

    # use kNN 
    start_time=time.clock()
    y_pred = knn.predict(X)
    y_test = y.T
    print_statistics(y_test,y_pred)
    print("\nTime to test KNN model = ", time.clock()-start_time)

# main method 
def main(file_train, file_test):
    # Preprocess data and store it in file
    #process_data(file_train, "instance_file")
    #X, y, tfidf, pca = read_instance("instance_file.npz")
    
    # Buildling ANN and kNN models
    tfidf, pca, ann, knn = build_models(file_train)

    # Predicting new data point using ANN and kNN
    test_models(tfidf,pca,ann,knn)

# Training and testing the model on input dataset
file_train ='USvideos_modified.csv' 
file_test ='USvideos_modified_test.csv' 

main(file_train, file_test)
