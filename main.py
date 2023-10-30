#Import libraries
import keras
import datetime
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from keras import regularizers
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

#Loading dataset
nb_classes         = 7
img_rows, img_cols = 48, 48
batch_size         = 32
train_data_dir   = '/content/dataset/train/'
test_data_dir    = '/content/dataset/test/'
train_datagen = ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.3,
    horizontal_flip=True
)
train_set = train_datagen.flow_from_directory(
  train_data_dir,
  color_mode  = 'grayscale',
  target_size = (img_rows, img_cols),
  batch_size  = batch_size,
  class_mode  = 'categorical',
  shuffle     = True
)
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory(
	test_data_dir,
  color_mode  = 'grayscale',
  target_size = (img_rows, img_cols),
  batch_size  = batch_size,
  class_mode  = 'categorical',
  shuffle     = True
)

#Exploratory Data Analysis
train_set.class_indices
#Create dataframe of train and test sets
def count_exp(path, set_):
    dict_ = {}
    for expression in os.listdir(path):
        dir_ = path + expression
        dict_[expression] = len(os.listdir(dir_))
    df = pd.DataFrame(dict_, index=[set_])
    return df

train_count = count_exp(train_data_dir, 'train')
test_count  = count_exp(test_data_dir, 'test')
print(train_count)
print(test_count)

# Plotting training data dist
train_count.transpose().plot(kind = 'bar')

# Plotting testing data dist
test_count.transpose().plot(kind = 'bar')

#Visualize a samples of dataset:
def plot_imgs(item_dir, top = 10):
    all_item_dirs = os.listdir(item_dir)
    item_files    = [os.path.join(item_dir, file) for file in all_item_dirs][:5]
  
    plt.figure(figsize = (10, 10))
  
    for idx, img_path in enumerate(item_files):
        plt.subplot(5, 5, idx + 1)
    
        img = plt.imread(img_path)
        plt.tight_layout()         
        plt.imshow(img, cmap = 'gray')

#Building the model:
def build_model(nb_classes, input_shape):

  model= tf.keras.models.Sequential()
  model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(48, 48,1)))
  model.add(Conv2D(64,(3,3), padding='same', activation='relu' ))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Conv2D(128,(5,5), padding='same', activation='relu'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
    
  model.add(Conv2D(512,(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Conv2D(512,(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  #Faltten the model
  model.add(Flatten())
    
  model.add(Dense(256))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(0.25))
    
  model.add(Dense(512))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(0.25))

  model.add(Dense(nb_classes, activation='softmax'))

  model.compile(
    optimizer = Adam(lr=0.0001 , decay=1e-6), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
  )

  return model

# Creating an instance of the model and printing the summary
model = build_model(nb_classes, (img_rows, img_cols, 1))
print(model.summary())
plot_model(model, to_file = 'model.png', show_shapes = True, show_layer_names = True)

#Training the model
chk_path = '/content/drive/MyDrive/facial /FER13.h5'
log_dir = "/content/drive/MyDrive/facial /checkpoint/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

checkpoint = ModelCheckpoint(
    filepath       = chk_path,
    save_best_only = True,
    verbose        = 1,
    mode           = 'min',
    moniter        = 'val_loss'
)
earlystop = EarlyStopping(
    monitor              = 'val_loss', 
    min_delta            = 0, 
    patience             = 3, 
    verbose              = 1, 
    restore_best_weights = True
)                        
reduce_lr = ReduceLROnPlateau(
    monitor   = 'val_loss', 
    factor    = 0.2, 
    patience  = 6, 
    verbose   = 1, 
    min_delta = 0.0001
)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
csv_logger = CSVLogger('training.log')
callbacks = [checkpoint, reduce_lr, csv_logger]

#Train function.
def train_model(train, test, epochs, callbacks):
  steps_per_epoch  = train.n // train.batch_size
  validation_steps = test.n // test.batch_size

  hist = model.fit(
      x                = train, 
      validation_data  = test, 
      epochs           = epochs, 
      callbacks        = callbacks, 
      steps_per_epoch  = steps_per_epoch, 
      validation_steps = validation_steps
  )
  return hist
hist = train_model(train_set, test_set, 60, callbacks)

# Plotting the loss & accuracy
plt.figure(figsize=(14,5))
plt.subplot(1,2,2)
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['train', 'test'], loc='upper left')

plt.subplot(1,2,1)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Evaluating the model
train_loss, train_acc = model.evaluate(train_set)
test_loss, test_acc   = model.evaluate(test_set)
print("final train accuracy = {:.2f} , validation accuracy = {:.2f}".format(train_acc*100, test_acc*100))

# Save the weights
model.save_weights('fer2013_weights.h5')

#Confusion matrix
y_pred = model.predict(train_set)
y_pred = np.argmax(y_pred, axis=1)
class_labels = test_set.class_indices
class_labels = {v:k for k,v in class_labels.items()}
cm_train = confusion_matrix(train_set.classes, y_pred)
print('Confusion Matrix')
print(cm_train)
print('Classification Report')
target_names = list(class_labels.values())
print(classification_report(train_set.classes, y_pred, target_names=target_names))
plt.figure(figsize=(8,8))
plt.imshow(cm_train, interpolation='nearest')
plt.colorbar()
tick_mark = np.arange(len(target_names))
_ = plt.xticks(tick_mark, target_names, rotation=90)
_ = plt.yticks(tick_mark, target_names)
