import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
import pandas as pd
from matplotlib import pyplot as plt
import os
import cv2
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

# Define paths
train_val_test = "charts/train_val"
test = "charts/test"
train_path_csv = "charts/train_val.csv"

# Read the train_val.csv file
train_val_csv = pd.read_csv(train_path_csv)

# Load train_val data
images = []
labels = []
for filename in os.listdir(train_val_test):
    if filename.endswith('.png'):
        img = cv2.imread(os.path.join(train_val_test, filename))
        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_array = np.array(img)

        images.append(img_array)
        labels.append(filename)

# Label encode the train_val data
le = LabelEncoder()
labels = le.fit_transform(labels)

images = np.array(images)
labels = np.array(labels)

# Save the train_val data
np.save('x_train.npy', images)
np.save('y_train.npy', labels)

# Load the train_val data
x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')

# Load test data
images = []
labels = []
for filename in os.listdir(test):
    if filename.endswith('.png'):
        img = cv2.imread(os.path.join(test, filename))
        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_array = np.array(img)

        images.append(img_array)
        labels.append(filename)

# Label encode the test data
le = LabelEncoder()
labels = le.fit_transform(labels)

images = np.array(images)
labels = np.array(labels)

# Save the test data
np.save('x_test.npy', images)
np.save('y_test.npy', labels)

# Load the test data
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')

# Normalize the data
x_train = x_train / 255
x_test = x_test / 255

# Define image classes and label map
image_classes = ['line', 'dot_line', 'hbar_categorical', 'vbar_categorical', 'pie']
label_map = {'line': 0, 'dot_line': 1, 'hbar_categorical': 2, 'vbar_categorical': 3, 'pie': 4}

# Convert train_val_csv type column into encoded label column
y_train = np.array([label_map[label] for label in train_val_csv['type']])


# Define a function to display an image with its label
def image_sample(x, y, index):
    plt.figure(figsize=(10, 2))
    plt.imshow(x[index])
    plt.xlabel(image_classes[y[index]])


# Display some sample images with their labels
image_sample(x_train, y_train, 0)
image_sample(x_train, y_train, 208)
image_sample(x_train, y_train, 444)

# Define the model architecture for a basic CNN model
model = Sequential([
    Flatten(input_shape=(128,128,3)),
    Dense(3000, activation='relu'),
    Dense(1000, activation='relu'),
    Dense(5, activation='softmax')
])

# Compile the model and fit it to the training data
model.compile(optimizer='SGD', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train,y_train,epochs=10)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Evaluate the model on the testing set
model.evaluate(x_test,y_test)

# Predict on the testing set and get the predicted classes
y_pred = model.predict(x_test)
y_pred_classes = [np.argmax(ele) for ele in y_pred]

# Print the shapes of the training and testing data
print("Train Images Shape:", x_train.shape)
print("Train Labels Shape:", y_train.shape)
print("Test Images Shape:", x_test.shape)
print("Test Labels Shape:", y_test.shape)

# Define the model architecture for a CNN model with multiple convolutional and pooling layers
cnn_model = Sequential([
    Conv2D(filters=16 ,kernel_size=(3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(5, activation='softmax')
])

# Compile the model and fit it to the training data
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = cnn_model.fit(x_train, y_train, batch_size=1000, epochs=50,validation_data=(x_test, y_test))

# Plot the model loss over the epochs for both the training and validation data
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# Evaluate the model on the testing set
cnn_model.evaluate(x_test,y_test)

# Visualize some sample images from the testing set
image_sample(x_test,y_test,1)
image_sample(x_test,y_test,50)
image_sample(x_test,y_test,25)
image_sample(x_test,y_test,30)

# Predict on the testing set and get the predicted classes
y_pred = cnn_model.predict(x_test)
y_pred[:5]

# Get the predicted classes
y_classes = [np.argmax(element) for element in y_pred]
y_classes[:5]

# Print the first 5 actual labels from the testing set
y_test[:5]

# Display a sample image with its actual and predicted classes
image_sample(x_test, y_test, 15) # actual
image_classes[y_classes[15]] # predicted

# Print classification report
print("Classification report:\n", classification_report(y_test, y_classes))

# Create a confusion matrix and display it
conf_mat = confusion_matrix(y_test, y_classes)
print('Confusion Matrix:')
print(conf_mat)
import seaborn as sn
plt.figure(figsize = (10,10))
sn.heatmap(conf_mat, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Import required libraries
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the VGG16 model with pre-trained weights and exclude the top layer
vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add layers to the VGG16 model
x = vgg16_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(5, activation='softmax')(x)

# Create a new model with VGG16 as base and the new layers added
pt_model = tf.keras.Model(inputs=vgg16_model.input, outputs=predictions)

# Freeze the layers of the base model
for layer in pt_model.layers:
    layer.trainable = False

# Compile the model
pt_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print the summary of the model
pt_model.summary()

# Define the data generators for training and testing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(x_train, y_train, batch_size=32)
test_generator = train_datagen.flow(x_test, y_test, batch_size=32)

# Set up early stopping
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)

# Train the model
history = pt_model.fit(train_generator, epochs=100, validation_data=test_generator, callbacks=[es])