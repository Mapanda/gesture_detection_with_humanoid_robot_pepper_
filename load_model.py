import numpy as np
import keras
from numpy import loadtxt
from keras.models import load_model
# load model
model = load_model('VGG_cross_validated.h5')
# summarize model.
model.summary()
# load dataset
#dataset = loadtxt("pima-indians-diabetes.csv", delimiter=",")

keras.utils.plot_model(model, 'my_first_model_with_shape_info.png', show_shapes=True)