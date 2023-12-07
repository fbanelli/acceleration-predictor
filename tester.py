import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Load the data
commanded_inputs = pd.read_csv('Learning/cloverData/filteredData/groundTruth_rpm.csv')
ground_truth = pd.read_csv('Learning/cloverData/filteredData/groundTruth_acc.csv')

# Remove the first column of data called timestamp
commanded_inputs = commanded_inputs.iloc[:, 1:]
ground_truth = ground_truth.iloc[:, 1:]

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
commanded_inputs_scaled = sc.fit_transform(commanded_inputs)
ground_truth_scaled = sc.fit_transform(ground_truth)

# Split into training, validation, and test sets
total_data_size = np.shape(commanded_inputs_scaled)[0]
training_size = 0.7
validation_size = 0.15
test_size = 0.15

training_index = int(total_data_size*training_size)
validation_index = int(total_data_size*(training_size+validation_size))

commanded_inputs_scaled_training = commanded_inputs_scaled[:training_index, :]
ground_truth_scaled_training = ground_truth_scaled[:training_index, :]
commanded_inputs_scaled_validation = commanded_inputs_scaled[training_index:validation_index, :]
ground_truth_scaled_validation = ground_truth_scaled[training_index:validation_index, :]
commanded_inputs_scaled_test = commanded_inputs_scaled[validation_index:, :]
ground_truth_scaled_test = ground_truth_scaled[validation_index:, :]


# Create a data structure for training with 10 timesteps and 1 output
# Setting up the training set
buffer_size = 10
X_train = []
y_train = []
for i in range(buffer_size, training_index):
    commandeded_inputs_matrix = commanded_inputs_scaled[i-buffer_size:i, :]
    X_train.append(commanded_inputs_scaled[i-buffer_size:i, :])
    y_train.append(ground_truth_scaled[i, :])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 4))

# Setting up the validation set
X_validate = []
y_validate = []
for i in range(training_index + buffer_size, validation_index):
    commandeded_inputs_matrix = commanded_inputs_scaled[i-buffer_size:i, :]
    X_validate.append(commanded_inputs_scaled[i-buffer_size:i, :])
    y_validate.append(ground_truth_scaled[i, :])
X_validate, y_validate = np.array(X_validate), np.array(y_validate)

# Reshaping
X_validate = np.reshape(X_validate, (X_validate.shape[0], X_validate.shape[1], 4))

# Setting up the test set
X_test = []
y_test = []
for i in range(validation_index + buffer_size, np.shape(commanded_inputs_scaled)[0]):
    commandeded_inputs_matrix = commanded_inputs_scaled[i-buffer_size:i, :]
    X_test.append(commanded_inputs_scaled[i-buffer_size:i, :])
    y_test.append(ground_truth_scaled[i, :])
X_test, y_test = np.array(X_test), np.array(y_test)

# Reshaping
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 4))

# Import the Keras libraries and packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

# Load the model parameters from the file 'model.h5'
model = tf.keras.models.load_model('Learning/model.h5')

# Evaluate the model
score = model.evaluate(X_test, y_test, batch_size = 32)

# Visualize results on test set
predicted = model.predict(X_test)
predicted = sc.inverse_transform(predicted)
ground_truth = sc.inverse_transform(y_test)

plt.figure()
plt.plot(ground_truth[:, 0], color = 'red', label = 'Actual X acceleration')
plt.plot(predicted[:, 0], color = 'blue', label = 'Predicted X acceleration')
plt.title('X Acceleration Prediction')
plt.xlabel('Time')
plt.ylabel('X Acceleration')
plt.legend()
plt.savefig('Learning/cloverData/results/Xtestresults.png')

plt.figure()
plt.plot(ground_truth[:, 1], color = 'red', label = 'Actual Y acceleration')
plt.plot(predicted[:, 1], color = 'blue', label = 'Predicted Y acceleration')
plt.title('Y Acceleration Prediction')
plt.xlabel('Time')
plt.ylabel('Y Acceleration')
plt.legend()
plt.savefig('Learning/cloverData/results/Ytestresults.png')

plt.figure()
plt.plot(ground_truth[:, 2], color = 'red', label = 'Actual Z acceleration')
plt.plot(predicted[:, 2], color = 'blue', label = 'Predicted Z acceleration')
plt.title('Z Acceleration Prediction')
plt.xlabel('Time')
plt.ylabel('Z Acceleration')
plt.legend()
plt.savefig('Learning/cloverData/results/Ztestresults.png')

plt.show()

# Visualize results on validation set
predicted = model.predict(X_validate)
predicted = sc.inverse_transform(predicted)
ground_truth = sc.inverse_transform(y_validate)

plt.figure()
plt.plot(ground_truth[:, 0], color = 'red', label = 'Actual X acceleration')
plt.plot(predicted[:, 0], color = 'blue', label = 'Predicted X acceleration')
plt.title('X Acceleration Prediction')
plt.xlabel('Time')
plt.ylabel('X Acceleration')
plt.legend()
plt.savefig('Learning/cloverData/results/Xvalidateresults.png')

plt.figure()
plt.plot(ground_truth[:, 1], color = 'red', label = 'Actual Y acceleration')
plt.plot(predicted[:, 1], color = 'blue', label = 'Predicted Y acceleration')
plt.title('Y Acceleration Prediction')
plt.xlabel('Time')
plt.ylabel('Y Acceleration')
plt.legend()
plt.savefig('Learning/cloverData/results/Yvalidateresults.png')

plt.figure()
plt.plot(ground_truth[:, 2], color = 'red', label = 'Actual Z acceleration')
plt.plot(predicted[:, 2], color = 'blue', label = 'Predicted Z acceleration')
plt.title('Z Acceleration Prediction')
plt.xlabel('Time')
plt.ylabel('Z Acceleration')
plt.legend()
plt.savefig('Learning/cloverData/results/Zvalidateresults.png')

plt.show()