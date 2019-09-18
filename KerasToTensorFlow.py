import numpy as np # We'll be storing our data as numpy arrays
import os # For handling directories
from PIL import Image # For handling the images

from keras.utils import to_categorical
from keras import layers
from keras import models
from keras.layers import Dense, Dropout, Flatten
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.tools import  freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

sess = tf.compat.v1.Session()
from keras import backend as K
K.set_session(sess)
from timeit import default_timer as timer
tf.compat.v1.disable_eager_execution()
model_version = "2"
lookup = dict()
reverselookup = dict()
count = 0
MODEL_NAME = 'keras'
start = timer()
for j in os.listdir('leapGestRecog/00'):
    if not j.startswith('.'): # If running this code locally, this is to
                              # ensure you aren't reading in hidden folders
        lookup[j] = count
        reverselookup[count] = j
        count = count + 1
x_data = []
y_data = []
datacount = 0 # We'll use this to tally how many images are in our dataset
for i in range(0, 10): # Loop over the ten top-level folders
    for j in os.listdir('leapGestRecog/0' + str(i) + '/'):
        if not j.startswith('.'): # Again avoid hidden folders
            count = 0 # To tally images of a given gesture
            for k in os.listdir('leapGestRecog/0' + str(i) + '/' + j + '/'): # Loop over the images
               # print(k)
                img = Image.open('leapGestRecog/0' + str(i) + '/' + j + '/' + k).convert('L') # Read in and convert to greyscale
                img = img.resize((224, 224))
                arr = np.array(img)
                x_data.append(arr)
                count = count + 1
                #print("Count : " , count)

            y_values = np.full((count, 1), lookup[j])
            y_data.append(y_values)
            datacount = datacount + count
           # print("datacount : " ,datacount)
x_data = np.array(x_data, dtype = 'float32')
y_data = np.array(y_data)
y_data = y_data.reshape(datacount, 1) # Reshape to be the correct size
x_data = np.array(x_data, dtype='float32')
x_data = np.stack((x_data,)*3, axis=-1)
y_data = to_categorical(y_data)
x_data = x_data.reshape(datacount, 224, 224, 3)
x_data /= 255
x_train,x_further,y_train,y_further = train_test_split(x_data,y_data,test_size = 0.2)
x_validate,x_test,y_validate,y_test = train_test_split(x_further,y_further,test_size = 0.5)

#model=models.Sequential()
#model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), activation='relu', input_shape=(224, 224,1)))
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#model.add(layers.MaxPooling2D((2, 2)))

#model.add(layers.Flatten())
#model.add(layers.Dense(128, activation='relu'))
#model.add(layers.Dense(10, activation='softmax'))


model = models.Sequential()
model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
#model.add(layers.)
model.add(Dropout(0.25, seed=21))
model.add(layers.Dense(10, activation='softmax'))


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=64, verbose=1, validation_data=(x_validate, y_validate))
#
[loss, acc] = model.evaluate(x_test,y_test,verbose=1)
#
model.save("model_tensorflow_lite.h5")


print("Done")

model.summary()
print(model.outputs)
print(model.inputs)


def export_model_for_mobile(model_name, input_node_name, output_node_name):
    tf.train.write_graph(K.get_session().graph_def, 'out', \
                         model_name + '_graph.pbtxt')

    tf.train.Saver().save(K.get_session(), 'out/' + model_name + '.chkp')

    freeze_graph.freeze_graph('out/' + model_name + '_graph.pbtxt', None, \
                              False, 'out/' + model_name + '.chkp', output_node_name, \
                              "save/restore_all", "save/Const:0", \
                              'out/frozen_' + model_name + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + model_name + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def, [input_node_name], [output_node_name],
        tf.float32.as_datatype_enum)

    with tf.gfile.GFile('out/tensorflow_lite_' + model_name + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())


export_model_for_mobile('keras_tensor_lite', "conv2d_1_input", "dense_2/Softmax")
elapsed_time = timer() - start
print("Done2")
print("Time taken to run the whole project" , elapsed_time)