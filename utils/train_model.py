from tensorflow.keras.datasets import mnist
import tensorflow as tf
from tensorflow.keras.models import load_model
#from telegram_bot_callback import TelegramBotCallback
#from dl_bot import DLBot
from tensorflow.keras.datasets import boston_housing
import json

def Train_model(model,json_file,bot,update):

	epochs=3
	
	#telegram callback
	telegram_callback = TelegramBotCallback(bot,update)
	
	#loading config
	with open(json_file,'r') as f:
		config=json.load(f)
	category = config['category']

	#for classification
	if category==1:	
		(X_train,y_train),(X_test,y_test)=mnist.load_data()
		X_train=tf.keras.utils.normalize(X_train,axis=1)
		X_test=tf.keras.utils.normalize(X_test,axis=1)
		
		model.fit(X_train,y_train,\
      			epochs=epochs,\
             	validation_data=(X_test,y_test),\
                verbose=1,\
                callbacks=[telegram_callback])
		
		score = model.evaluate(X_test, y_test, verbose=0)
		bot.send_message('Test loss:' + str(score[0]))
		bot.send_message('Test accuracy:' + str(score[1]))
	
	#for regression
	elif category==2:
		(X_train,y_train),(X_test,y_test)=boston_housing.load_data()

		model.fit(X_train,y_train,\
      		epochs=3,\
            validation_data=(X_test,y_test),\
            verbose=1,\
            callbacks=[telegram_callback])
  
  		score = model.evaluate(X_test, y_test, verbose=0)
		bot.send_message('Test loss:' + str(score))
		# bot.send_message('Test accuracy:' + str(score[1]))

	elif category==3:
		num_classes=10
		(X_train,y_train),(X_test,y_test)=mnist.load_data()
		X_train=X_train[:1000]
		y_train=y_train[:1000]
		X_test=X_test[:200]
		y_test=y_test[:200]
		img_rows, img_cols = 28, 28
		X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
		X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
		X_train = X_train.astype('float32')
		X_test = X_test.astype('float32')
		X_train /= 255
		X_test /= 255

		y_train = tf.keras.utils.to_categorical(y_train, num_classes)
		y_test = tf.keras.utils.to_categorical(y_test, num_classes)
		#print(X_train.shape)	
		model.fit(X_train,y_train,\
      			epochs=3,\
             	batch_size=32,\
                validation_data=(X_test,y_test),\
                verbose=1,\
                callbacks=[telegram_callback])