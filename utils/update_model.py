import json
from tensorflow.keras.models import load_model
import tensorflow as tf
from train_model import Train_model
def update_model(json_file,model):	
	with open(json_file,'r') as f:
		config=json.load(f)
	
	hidden_layers=config['no_of_hidden_layers']
	hidden_layer_size=config['hidden_unit_size']
	activation_function=config['activation_function']
	eval_metrics=config['metrics']
	category=config['category']
	optimizer_func=config['optimizer']
	
	orig_layer=len(model.layers)
	new_layers=hidden_layers+2 - orig_layer
	model.layers.pop()
	
	if new_layers>0:
		model.add(tf.keras.layers.Dense(hidden_layer_size,activation=activation_function))
		for i in range(1,orig_layer-1):
			model.layers[i]=tf.keras.layers.Dense(hidden_layer_size,activation=activation_function)
	elif new_layers<0:
		new_layers=abs(new_layers)
		for i in range(new_layers):
			model.layers.pop()
		for i in range(1,orig_layer-1-new_layers):
			model.layers[i]=tf.keras.layers.Dense(hidden_layer_size,activation=activation_function)
	else:
		for i in range(1,orig_layer-1):
			model.layers[i]=tf.keras.layers.Dense(hidden_layer_size,activation=activation_function)
			print(model.layers[i].name)
	if category==1:
		model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))
		model.compile(optimizer=optimizer_func,loss='sparse_categorical_crossentropy',metrics=eval_metrics)
	else:
		model.add(tf.keras.layers.Dense(1,activation='tanh'))
		model.compile(optimizer=optimizer_func,loss='mean_squared_error',metrics=['mae'])		
	return model


			
