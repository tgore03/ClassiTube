import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

#data = np.loadtxt('USvideos_modified.csv', delimiter=",", dtype="str")
data = pd.read_csv('USvideos_modified.csv')
type_str_list = ['channel_title', 'tag_appeared_in_title', 'title', 'tags', 'description']
for column in type_str_list:
    data[column] = data[column].astype(str)

data['category_id'] = data['category_id'].astype(int)

#non_bmp_map = dict.fromkeys(range(0x10000, sys.maxunicode + 1), 0xfffd)

print(data[['channel_title', 'tag_appeared_in_title', 'tags', 'description','title']])
#for i,row in data.iterrows():
#    row['title'] = row['title'].translate(non_bmp_map)

print(data['category_id'])


#PCA
pca = PCA(0.95)
pca.fit(X)
print(X)
A = pca.transform(X)
print(A)
#x_test = pca.transform(X)

#Perform Neural Networks

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 1-of-c output encoding
Y_train = np_utils.to_categorical(y_train)

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
neurons = 200


# Neural Network Model
def neural_network_model(hidden_units, error_function, data, label):
    startTime = time.clock()
    model = Sequential()
    model.add(Dense(neurons, input_dim=64, activation=hidden_units))  # First hidden layer
    model.add(Dense(neurons, activation=hidden_units))  # Second hidden layer
    model.add(Dense(neurons, activation=hidden_units))  # Third hidden layer
    model.add(Dense(10, activation='softmax'))  # Softmax function for output layer

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
        fin_prediction.append(numpy.argmax(row))
    matrix = confusion_matrix(y_data, fin_prediction)
    sum = 0
    print("\nClass Accuracies:")
    for i in range(10):
        sum += matrix[i][i]
        print("Class ", i, ": ", round(matrix[i][i]/numpy.sum(matrix[i]), 4))
    print("\nOverall Accuracy: ", round(sum/numpy.sum(matrix), 4))
    print("\nConfusion Matrix:\n", matrix)


print("\nTraining Data Statistics:\n")
print("Relu Hidden Units with Cross-Entropy Error Function")
neural_network_model('relu', 'categorical_crossentropy', X_train, train_label)
print("\n--------------------------------------------------------------")



