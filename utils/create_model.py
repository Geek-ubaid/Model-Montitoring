import tensorflow as tf
import cv2
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import load_model,Sequential
from tensorflow.keras import layers
import json
from train_model import Train_model
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout

def regression_model(no_of_hidden_layers,hidden_unit_size,activation_function,optimizer_func):
	model=tf.keras.Sequential()
	model.add(tf.keras.layers.Dense(13,input_dim=13,activation=activation_function))
	for i in range(no_of_hidden_layers):
		model.add(tf.keras.layers.Dense(hidden_unit_size,activation=activation_function))
	model.add(tf.keras.layers.Dense(1,activation='tanh'))
	model.compile(optimizer='adam',loss='mean_squared_error',metrics=['mae'])
	return model

def classsification_model(no_of_hidden_layers,hidden_unit_size,activation_function,eval_metrics,optimizer_func):
	model=tf.keras.Sequential()
	model.add(tf.keras.layers.Flatten())
	for i in range(no_of_hidden_layers):
		model.add(tf.keras.layers.Dense(hidden_unit_size,activation=activation_function))
	model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))
	model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=eval_metrics)
	return model

def cnn(no_of_cnn_layers,no_of_cnn_filters,Activation,Optimizer,eval_metrics):
	input_shape=(28,28,1)
	model=tf.keras.Sequential()
	model.add(Conv2D(no_of_cnn_filters,(3,3),activation=Activation,input_shape=input_shape))
	for i in range(no_of_cnn_layers-1):
		model.add(Conv2D(no_of_cnn_filters,(3,3),activation=Activation))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(128,activation=Activation))
	model.add(Dropout(0.5))
	model.add(Dense(10,activation='softmax'))

	model.compile(loss='binary_crossentropy',optimizer=Optimizer,metrics=eval_metrics)
	return model

def return_model(json_file):
	with open(json_file,'r') as f:
		config=json.load(f)
	no_hidden_layers=config['no_of_hidden_layers']
	hidden_unit_size=config['hidden_unit_size']
	activation_function=config['activation_function']
	eval_metrics=config['metrics']
	category=config['category']
	optimizer_func=config['optimizer']	
	if category==1:
		model=classsification_model(no_of_hidden_layers,hidden_unit_size,activation_function,optimizer_func)
	elif category==2:
		model=regression_model(no_of_hidden_layers,hidden_unit_size,activation_function,eval_metrics,optimizer_func)
	elif category==3:
		model=cnn(no_of_hidden_layers,hidden_unit_size,activation_function,optimizer_func,eval_metrics)
	return model
