def update_model(json_file):
	import json
	from tensorflow.keras.models import load_model
	import tensorflow as tf
	
	model=load_model('ann.h5')
	with open(json_file,'r') as f:
		config=json.load(f)
	
	input_size=config['input_size']
	classes=config['classes']
	hidden_layers=config['no_of_hidden_layers']
	hidden_layer_size=config['hidden_unit_size']
	activation_function=config['activation_function']
	eval_metrics=config['metrics']
	category=config['category']
	
	
	orig_layer=len(model.layers)
	new_layers=hidden_layers+2 - orig_layer
	model.layers.pop()
	model.layers[0]=tf.keras.layers.Dense(input_size,activation=activation_function)
	
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
	if category==1:
		model.add(tf.keras.layers.Dense(classes,activation=tf.nn.softmax))
		model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=eval_metrics)
	else:
		model.add(tf.keras.layers.Dense(classes,activation='tanh'))
		model.compile(optimizer='adam',loss='mean_squared_error',metrics=eval_metrics)
	
	for i in range(1,len(model.layers)):
    		model.layers[i].name = model.layers[i].name + '1'
		
	model.save('ann3.h5')

if __name__=='__main__':
	update_model('config.json')
			
