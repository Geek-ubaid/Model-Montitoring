import telegram
import keras.backend as K
from keras.callbacks import Callback


class TelegramCallback(Callback):

    def __init__(self, bot, update , name=None):
        super(TelegramCallback, self).__init__()
        self.bot = bot
        self.user_id = update['message']['chat']['id']

    def send_message(self, text):
        try:
            self.bot.send_message(chat_id=self.user_id, text=text)
        except Exception as e:
            print('Message did not send. Error: {}.'.format(e))

    def on_train_begin(self, logs={}):
        text = 'Start training model {}.'.format('1')
        self.send_message(text)
        self.epochs = self.params['epochs']  

        self.loss_hist = []
        self.val_loss_hist = []

    def on_train_end(self, logs={}):

        text = 'Training model {} ended.'.format('1')
        self.send_message(text)

    def on_epoch_end(self, epoch, logs={}):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        self.bot.lr = logs['lr']  
        
        text = '{}: Epoch {}.\n'.format('1', epoch)
        for k, v in logs.items():
            if k != "lr":
                text += '{}: {:.4f}; '.format(k, v)
            else:
                text += '{}: {:.6f}; '.format(k, v) #4 decimal places too short for learning rate
        self.send_message(text)

        self.loss_hist.append(logs['loss'])
        if 'val_loss' in logs:
            self.val_loss_hist.append(logs['val_loss'])

        print(self.loss_hist)
        print(self.val_loss_hist)