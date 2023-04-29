import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt
import os
import cv2

# Define the directories for training and validation data
train_dir = 'train'
val_dir = 'val'

# Set the image size to be 32x32
img_size = (32, 32)

# Initialize empty lists to hold the images and labels for training data
images = []
labels = []

# Loop through each label (0-9)
for label in range(10):
    folder_path = os.path.join(train_dir, str(label))

    # Loop through each image file in the folder and add the image and label to the lists
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if file_path.endswith(('.tiff', '.bmp')):
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Read the image file in grayscale
            img = cv2.resize(img, img_size)  # Resize the image to 32x32

            images.append(img)
            labels.append(label)

# Convert the lists to NumPy arrays
images = np.array(images)
labels = np.array(labels)

# Save the training data to NumPy files
np.save('x_train.npy', images)
np.save('y_train.npy', labels)

# Initialize empty lists to hold the images and labels for validation data
images_val = []
labels_val = []

# Loop through each label (0-9)
for label in range(10):
    folder_path = os.path.join(val_dir, str(label))

    # Loop through each image file in the folder and add the image and label to the lists
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if file_path.endswith(('.tiff', '.bmp')):
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Read the image file in grayscale
            img = cv2.resize(img, img_size)  # Resize the image to 32x32

            images_val.append(img)
            labels_val.append(label)

# Convert the lists to NumPy arrays
images_val = np.array(images_val)
labels_val = np.array(labels_val)

# Save the validation data to NumPy files
np.save('x_test.npy', images_val)
np.save('y_test.npy', labels_val)

# Load the training and validation data from the saved NumPy files
x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')

print(len(x_train))
print(len(x_test))
print(x_train[0].shape)
plt.matshow(x_train[0])
plt.matshow(x_train[999])
print(x_train.shape)
print(x_test.shape)
print(y_train)

# Reshape the training data to be a 2D array
x_train_flat = x_train.reshape(len(x_train), 32 * 32)

# Reshape the validation data to be a 2D array
x_test_flat = x_test.reshape(len(x_test), 32 * 32)

print(x_train_flat.shape)
print(x_test_flat.shape)
print(x_train_flat[0])

# Define a simple neural network with one hidden layer
model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(1024,), activation='sigmoid')
])

# Compile the model with the Adam optimizer and sparse categorical crossentropy loss
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )
# Create a Sequential model with a Flatten layer and a Dense layer with 10 neurons
# The input shape of the Dense layer is 1024, and its activation function is sigmoid
model = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(10, input_shape=(1024,), activation='sigmoid')
])

# Compile the model using the Adam optimizer, sparse categorical cross-entropy loss function,
# and accuracy as the evaluation metric
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )

# Fit the model on the training data for 10 epochs, with the validation data for validation
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Scale the pixel values of the training data and the testing data
x_train_t = x_train/255
x_test_scaled = x_test/255

# Fit the scaled data on the model for 10 epochs with validation data
model.fit(x_train_t, y_train, epochs=10, validation_data=(x_test_scaled, y_test))

# Evaluate the model on the scaled testing data
model.evaluate(x_test_scaled, y_test)

# Create another Sequential model with a Flatten layer and two Dense layers
# The first Dense layer has 1024 neurons with the ReLU activation function
# The second Dense layer has 10 neurons with the softmax activation function
model2 = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(1024,input_shape=(1024,), activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model using the Adam optimizer, sparse categorical cross-entropy loss function,
# and accuracy as the evaluation metric
model2.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )

# Fit the scaled training data on the model2 for 10 epochs with validation data
history = model2.fit(x_train_t, y_train, epochs=10, validation_data=(x_test_scaled, y_test))

# Evaluate the model2 on the scaled testing data
model2.evaluate(x_test_scaled, y_test)

# Make predictions on the scaled testing data using the model2
y_predicted = model2.predict(x_test_scaled)

# Convert the predicted probabilities into predicted labels
y_predicted_labels=[np.argmax(i) for i in y_predicted]

# Compute the confusion matrix for the predicted labels and the true labels
conf_mat = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)

# Visualize the confusion matrix using a heatmap
plt.figure(figsize = (10,10))
sn.heatmap(conf_mat,annot=True,fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Evaluate the original model on the testing data and print the test accuracy
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

# Plot the accuracy of model2 during training on both the training data and validation data
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
