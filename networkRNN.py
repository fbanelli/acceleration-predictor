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

print(np.shape(commanded_inputs_scaled_training))
print(np.shape(ground_truth_scaled_training))
print(np.shape(commanded_inputs_scaled_validation))
print(np.shape(ground_truth_scaled_validation))
print(np.shape(commanded_inputs_scaled_test))
print(np.shape(ground_truth_scaled_test))

# Creating a data structure for training with 90 timesteps and 1 output
buffer_size = 90
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

# Initialising the RNN
model = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 4)))
model.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50))
model.add(Dropout(0.2))

# Adding the output layer
model.add(Dense(units = 3))

# Compiling the RNN
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
model.fit(X_train, y_train, validation_data = (X_validate, y_validate), shuffle = False, epochs = 1000, batch_size = 32)

# Evaluate the model
score = model.evaluate(X_test, y_test, batch_size = 32)

## Getting the predicted values
#predicted = model.predict(X_test)
#predicted = sc.inverse_transform(predicted)
#
## Visualising the results
#plt.plot(ground_truth[20000:, 0], color = 'red', label = 'Actual acceleration')
#plt.plot(predicted[0, 0], color = 'blue', label = 'Predicted acceleration')
#plt.title('Acceleration Prediction')
#plt.xlabel('Time')
#plt.ylabel('Acceleration')
#plt.legend()
#plt.show()