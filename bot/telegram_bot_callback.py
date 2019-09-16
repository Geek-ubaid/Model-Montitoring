import telegram
from keras.callbacks import Callback


class TelegramCallback(Callback):

    def __init__(self, config, name=None):
        super(TelegramCallback, self).__init__()
        self.user_id = 645230191
        self.bot = telegram.Bot('823412051:AAHEYubDdjV-aJqcFQvbWTOLgKPX5BMTnUo')
      

    def send_message(self, text):
        try:
            self.bot.send_message(chat_id=self.user_id, text=text)
        except Exception as e:
            print('Message did not send. Error: {}.'.format(e))

    def on_train_begin(self, logs={}):
        text = 'Start training model {}.'.format('1')
        self.send_message(text)

    def on_train_end(self, logs={}):
        text = 'Training model {} ended.'.format('1')
        self.send_message(text)

    def on_epoch_end(self, epoch, logs={}):
        text = '{}: Epoch {}.\n'.format('1', epoch)
        for k, v in logs.items():
            if k != "lr":
                text += '{}: {:.4f}; '.format(k, v)
            else:
                text += '{}: {:.6f}; '.format(k, v) #4 decimal places too short for learning rate
        self.send_message(text)