# # import numpy as np # We'll be storing our data as numpy arrays
# # # import os # For handling directories
# # # from PIL import Image # For handling the images
# # #
# # # from keras.utils import to_categorical
# # # from keras import layers
# # # from keras import models
# # # import os.path as path
# # # from sklearn.model_selection import train_test_split
# # # import tensorflow as tf
# # # from tensorflow.python.tools import  freeze_graph
# # # from tensorflow.python.tools import optimize_for_inference_lib
# # #
# # # sess = tf.compat.v1.Session()
# # # from keras import backend as K
# # # K.set_session(sess)
# # # tf.compat.v1.disable_eager_execution()
# # # model_version = "2"
# # # lookup = dict()
# # # reverselookup = dict()
# # # count = 0
# # # MODEL_NAME = 'Keras_Tensorflow'
# # # BATCH_SIZE = 16
# # # NUM_STEPS = 3000
# # #
# # #
# # # def model_input(input_node_name):
# # #     x = tf.compat.v1.placeholder(tf.float32, shape=(1024, 1024) , name = input_node_name)
# # #     y_ = tf.matmul(x, x)
# # #
# # #     return x, y_
# # #
# # #
# # # def extraction_and_training():
# # #     global count
# # #     for j in os.listdir('C:/Users/Pepper/PycharmProjects/kerasToTensorflow/input/leapGestRecog/00'):
# # #         if not j.startswith('.'):  # If running this code locally, this is to
# # #             # ensure you aren't reading in hidden folders
# # #             lookup[j] = count
# # #             reverselookup[count] = j
# # #             count = count + 1
# # #     x_data = []
# # #     y_data = []
# # #     datacount = 0  # We'll use this to tally how many images are in our dataset
# # #     for i in range(0, 10):  # Loop over the ten top-level folders
# # #         for j in os.listdir('C:/Users/Pepper/PycharmProjects/kerasToTensorflow/input/leapGestRecog/0' + str(i) + '/'):
# # #             if not j.startswith('.'):  # Again avoid hidden folders
# # #                 count = 0  # To tally images of a given gesture
# # #                 for k in os.listdir('C:/Users/Pepper/PycharmProjects/kerasToTensorflow/input/leapGestRecog/0' + str(
# # #                         i) + '/' + j + '/'):  # Loop over the images
# # #                     # print(k)
# # #                     img = Image.open('C:/Users/Pepper/PycharmProjects/kerasToTensorflow/input/leapGestRecog/0' + str(
# # #                         i) + '/' + j + '/' + k).convert('L')  # Read in and convert to greyscale
# # #                     img = img.resize((320, 120))
# # #                     arr = np.array(img)
# # #                     x_data.append(arr)
# # #                     count = count + 1
# # #                     # print("Count : " , count)
# # #
# # #                 y_values = np.full((count, 1), lookup[j])
# # #                 y_data.append(y_values)
# # #                 datacount = datacount + count
# # #             # print("datacount : " ,datacount)
# # #     x_data = np.array(x_data, dtype='float32')
# # #     y_data = np.array(y_data)
# # #     y_data = y_data.reshape(datacount, 1)  # Reshape to be the correct size
# # #     y_data = to_categorical(y_data)
# # #     x_data = x_data.reshape((datacount, 120, 320, 1))
# # #     x_data /= 255
# # #     x_train, x_further, y_train, y_further = train_test_split(x_data, y_data, test_size=0.2)
# # #     x_validate, x_test, y_validate, y_test = train_test_split(x_further, y_further, test_size=0.5)
# # #     build_model(x_train, y_train,x_validate,y_validate,x_test,y_test)
# # #     return x_validate, x_test, y_validate, y_test;
# # #
# # #
# # # def build_model(x_train, y_train,x_validate,y_validate,x_test,y_test):
# # #
# # #     model = models.Sequential()
# # #     model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), activation='relu', input_shape=(120, 320,1)))
# # #     model.add(layers.MaxPooling2D((2, 2)))
# # #     model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# # #     model.add(layers.MaxPooling2D((2, 2)))
# # #
# # #     model.add(layers.Flatten())
# # #     model.add(layers.Dense(128, activation='relu'))
# # #     model.add(layers.Dense(10, activation='softmax'))
# # #     model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
# # #     model.fit(x_train, y_train, epochs=10, batch_size=64, verbose=1, validation_data=(x_validate, y_validate))
# # #     # loss and accuracy
# # #     [loss, accuracy] = model.evaluate(x_test,y_test,verbose=1)
# # #     print("Done")
# # #
# # #     merged_summary_op = model.summary()
# # #     print(model.outputs)
# # #     print(model.inputs)
# # #
# # #     return  merged_summary_op
# # #
# # #
# # # def export_model(model_name , input_node_names, output_node_name,x,y):
# # #
# # #         tf.train.write_graph(K.get_session().graph_def, '1', \
# # #                              model_name + '_graph.pbtxt')
# # #
# # #         tf.train.Saver().save(K.get_session(), '1/' + model_name + '.chkp')
# # #
# # #         freeze_graph.freeze_graph('1/' + model_name + '_graph.pbtxt', None, \
# # #                                   False, '1/' + model_name + '.chkp', output_node_name, \
# # #                                   "save/restore_all", "save/Const:0", \
# # #                                   '1/frozen_' + model_name + '.pb', True, "")
# # #
# # #         input_graph_def = tf.GraphDef()
# # #         with tf.gfile.Open('1/frozen_' + model_name + '.pb', "rb") as f:
# # #             input_graph_def.ParseFromString(f.read())
# # #             # import graph_def
# # #
# # #         output_graph_def = optimize_for_inference_lib.optimize_for_inference(
# # #             input_graph_def, [input_node_names], [output_node_name],
# # #             tf.float32.as_datatype_enum)
# # #
# # #         with tf.Graph().as_default() as graph:
# # #             tf.import_graph_def(input_graph_def)
# # #
# # #             # print operations
# # #         for op in graph.get_operations():
# # #             print(op.name)
# # #
# # #         with tf.gfile.GFile('1/tensorflow_' + model_name + '.pb', "wb") as f:
# # #             f.write(output_graph_def.SerializeToString())
# # #         print("Export is done:")
# # #
# # #
# # # def main():
# # #     if not path.exists('1'):
# # #         os.mkdir('1')
# # #     model_name = "keras_tensorflow"
# # #     input_node_name = "conv2d_1_input_1"
# # #     output_node_name = "dense_2/Softmax"
# # #     model_input(input_node_name)
# # #     x_validate, x_test, y_validate, y_test =extraction_and_training()
# # #
# # #     export_model(model_name,input_node_name, output_node_name,x_validate,y_validate)
# # #     print("All done!!")
# # #
# # #
# # # if __name__ == '__main__':
# # #     main()
#
#
# import numpy as np # We'll be storing our data as numpy arrays
# import os # For handling directories
# from PIL import Image # For handling the images
#
# from keras.utils import to_categorical
# from keras import layers
# from keras import models
#
# from sklearn.model_selection import train_test_split
# import tensorflow as tf
# from tensorflow.python.tools import  freeze_graph
# from tensorflow.python.tools import optimize_for_inference_lib
#
# sess = tf.compat.v1.Session()
# from keras import backend as K
# K.set_session(sess)
#
# tf.compat.v1.disable_eager_execution()
# model_version = "2"
# lookup = dict()
# reverselookup = dict()
# count = 0
# MODEL_NAME = 'keras'
# INPUT_SIZE=224
# # for j in os.listdir('C:/Users/Pepper/PycharmProjects/kerasToTensorflow/input/leapGestRecog/00'):
# #     if not j.startswith('.'): # If running this code locally, this is to
# #                               # ensure you aren't reading in hidden folders
# #         lookup[j] = count
# #         reverselookup[count] = j
# #         count = count + 1
# # x_data = []
# # y_data = []
# # datacount = 0 # We'll use this to tally how many images are in our dataset
# # for i in range(0, 5): # Loop over the ten top-level folders
# #     for j in os.listdir('C:/Users/Pepper/PycharmProjects/kerasToTensorflow/input/leapGestRecog/0' + str(i) + '/'):
# #         if not j.startswith('.'): # Again avoid hidden folders
# #             count = 0 # To tally images of a given gesture
# #             for k in os.listdir('C:/Users/Pepper/PycharmProjects/kerasToTensorflow/input/leapGestRecog/0' + str(i) + '/' + j + '/'): # Loop over the images
# #                # print(k)
# #                 img = Image.open('C:/Users/Pepper/PycharmProjects/kerasToTensorflow/input/leapGestRecog/0' + str(i) + '/' + j + '/' + k).convert('L') # Read in and convert to greyscale
# #                 img = img.resize((INPUT_SIZE, INPUT_SIZE))
# #                 arr = np.array(img)
# #                 x_data.append(arr)
# #                 count = count + 1
# #                 #print("Count : " , count)
# #
# #             y_values = np.full((count, 1), lookup[j])
# #             y_data.append(y_values)
# #             datacount = datacount + count
# #            # print("datacount : " ,datacount)
# # x_data = np.array(x_data, dtype = 'float32')
# # rgb = False
# # if rgb:
# #     pass
# # else:
# #     x_data = np.stack((x_data,) * 3, axis=-1)
# #
# # y_data = np.array(y_data)
# # y_data = y_data.reshape(datacount, 1) # Reshape to be the correct size
# #
# # y_data = to_categorical(y_data)
# # x_data = x_data.reshape((datacount, INPUT_SIZE, INPUT_SIZE, 1))
# # x_data /= 255
# # x_train,x_further,y_train,y_further = train_test_split(x_data,y_data,test_size = 0.2)
# # x_validate,x_test,y_validate,y_test = train_test_split(x_further,y_further,test_size = 0.5)
# #
# # model=models.Sequential()
# # model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), activation='relu', input_shape=(INPUT_SIZE, INPUT_SIZE,1)))
# # model.add(layers.MaxPooling2D((2, 2)))
# # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# # model.add(layers.MaxPooling2D((2, 2)))
# #
# # model.add(layers.Flatten())
# # model.add(layers.Dense(128, activation='relu'))
# # model.add(layers.Dense(10, activation='softmax'))
# #
# #
# # model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
# # model.fit(x_train, y_train, epochs=10, batch_size=64, verbose=1, validation_data=(x_validate, y_validate))
# # #
# # [loss, acc] = model.evaluate(x_test,y_test,verbose=1)
# # #
# # model.save("C:/Users/Pepper/PycharmProjects/kerasToTensorflow/input/model_tensorflow_lite.h5")
# #
# #
# # print("Done")
# #
# # model.summary()
# # print(model.outputs)
# # print(model.inputs)
# from keras.models import load_model
# from keras.utils.vis_utils import plot_model
#
# model = load_model('C:/Users/Pepper/PycharmProjects/kerasToTensorflow/out/VGG_cross_validated.h5')
# print(model.summary())
# print(model.outputs)
# # [<tf.Tensor 'dense_2/Softmax:0' shape=(?, 10) dtype=float32>]
# print(model.inputs)
#
# # # [<tf.Tensor 'conv2d_1_input:0' shape=(?, 28, 28, 1) dtype=float32>]
# #
# # # def export_model_for_mobile(model_name, input_node_name, output_node_name):
# # #     tf.train.write_graph(K.get_session().graph_def, '1', \
# # #                          model_name + '_graph.pbtxt')
# # #
# # #     tf.train.Saver().save(K.get_session(), '1/' + model_name + '.chkp')
# # #
# # #     freeze_graph.freeze_graph('1/' + model_name + '_graph.pbtxt', None, \
# # #                               False, '1/' + model_name + '.chkp', output_node_name, \
# # #                               "save/restore_all", "save/Const:0", \
# # #                               '1/frozen_' + model_name + '.pb', True, "")
# # #
# # #     input_graph_def = tf.GraphDef()
# # #     with tf.gfile.Open('1/frozen_' + model_name + '.pb', "rb") as f:
# # #         input_graph_def.ParseFromString(f.read())
# # #
# # #     output_graph_def = optimize_for_inference_lib.optimize_for_inference(
# # #         input_graph_def, [input_node_name], [output_node_name],
# # #         tf.float32.as_datatype_enum)
# # #
# # #     with tf.gfile.GFile('1/tensorflow_lite_' + model_name + '.pb', "wb") as f:
# # #         f.write(output_graph_def.SerializeToString())
# # #
# # # models.load_model("C:/Users/Pepper/PycharmProjects/kerasToTensorflow/out/VGG_cross_validated.h5")
# # # export_model_for_mobile('keras_tensor_lite', "conv2d_1_input", "dense_2/Softmax")
# #
# # print("Done2")
# #
# from keras import backend as K
# import tensorflow as tf
#
# def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
#     """
#     Freezes the state of a session into a pruned computation graph.
#
#     Creates a new computation graph where variable nodes are replaced by
#     constants taking their current value in the session. The new graph will be
#     pruned so subgraphs that are not necessary to compute the requested
#     outputs are removed.
#     @param session The TensorFlow session to be frozen.
#     @param keep_var_names A list of variable names that should not be frozen,
#                           or None to freeze all the variables in the graph.
#     @param output_names Names of the relevant graph outputs.
#     @param clear_devices Remove the device directives from the graph for better portability.
#     @return The frozen graph definition.
#     """
#     from tensorflow.python.framework.graph_util import convert_variables_to_constants
#     graph = session.graph
#     with graph.as_default():
#         freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
#         output_names = output_names or []
#         output_names += [v.op.name for v in tf.global_variables()]
#         # Graph -> GraphDef ProtoBuf
#         input_graph_def = graph.as_graph_def()
#         if clear_devices:
#             for node in input_graph_def.node:
#                 node.device = ""
#         frozen_graph = convert_variables_to_constants(session, input_graph_def,
#                                                       output_names, freeze_var_names)
#         return frozen_graph
#
#
# frozen_graph = freeze_session(K.get_session(),
#                               output_names=[out.op.name for out in model.outputs])
# tf.train.write_graph(frozen_graph, "model", "tf_model.pb", as_text=False)
#
# import tensorflow as tf
# from tensorflow.python.platform import gfile
#
# f = gfile.FastGFile("./model/tf_model.pb", 'rb')
# graph_def = tf.GraphDef()
# # Parses a serialized binary message into the current message.
# graph_def.ParseFromString(f.read())
# f.close()
#
# sess.graph.as_default()
# # Import a serialized TensorFlow `GraphDef` protocol buffer
# # and place into the current default `Graph`.
# tf.import_graph_def(graph_def)
# softmax_tensor = sess.graph.get_tensor_by_name('import/dense_2/Softmax:0')
# predictions = sess.run(softmax_tensor, {'import/conv2d_1_input:0': x_test[:20]})


from keras.models import load_model
import os
import numpy as np
from keras.utils import to_categorical
model = load_model('C:/Users/Pepper/PycharmProjects/kerasToTensorflow/out/VGG_cross_validated.h5')
rgb = False
from keras.preprocessing import image as image_utils
# def process_image(path):
#     img = Image.open(path)
#     img = img.resize((224, 224))
#     img = np.array(img)
#     return img
#
# def process_data(X_data, y_data):
#     X_data = np.array(X_data, dtype = 'float32')
#     if rgb:
#         pass
#     else:
#         X_data = np.stack((X_data,)*3, axis=-1)
#     X_data /= 255
#     y_data = np.array(y_data)
#     y_data = to_categorical(y_data)
#     return X_data, y_data
# def walk_file_tree(relative_path):
#     X_data = []
#     y_data = []
#     for directory, subdirectories, files in os.walk(relative_path):
#         for file in files:
#             if not file.startswith('.') and (not file.startswith('C_')):
#                 path = os.path.join(directory, file)
#                 gesture_name = gestures[file[0:2]]
#                 y_data.append(gestures_map[gesture_name])
#                 X_data.append(process_image(path))
#
#             else:
#                 continue
#
#     X_data, y_data = process_data(X_data, y_data)
#     return X_data, y_data
# relative_path = "C:/Users/Pepper/PycharmProjects/kerasToTensorflow/out/C_001.jpg"
# X_data, y_data = walk_file_tree(relative_path)
#
# def get_classification_metrics(X_test, y_test):
#     pred = model.predict(X_test)
#     pred = np.argmax(pred, axis=1)
#     y_true = np.argmax(y_test, axis=1)
#     print(confusion_matrix(y_true, pred))
#     print('\n')
#     print(classification_report(y_true, pred))
#
#
# get_classification_metrics(X_data, y_data)

gesture_names = {0: 'C',
                 1: 'Fist',
                 2: 'L',
                 3: 'Okay',
                 4: 'Palm',
                 5: 'Peace'}

def predict_rgb_image(path):
    img2rgb = image_utils.load_img(path=path, target_size=(224, 224))
    img2rgb = image_utils.img_to_array(img2rgb)
    img2rgb = img2rgb.reshape(1, 224, 224, 3)
    print(gesture_names[np.argmax(model.predict(img2rgb))])
    return gesture_names[np.argmax(model.predict(img2rgb))]

predict_rgb_image('C:/Users/Pepper/PycharmProjects/kerasToTensorflow/out/frame_04_02_0007.png')