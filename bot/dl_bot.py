import warnings

warnings.filterwarnings('ignore')

from telegram.ext import Updater, CommandHandler, Filters, RegexHandler,ConversationHandler, MessageHandler
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove
import keras_mnist_example as kr
from functools import wraps
from io import BytesIO
import numpy as np
import logging
import json    

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
      
modeltype, problem, makemodel, modelconfig = range(4)
           
    
class DLBot(object):

    def __init__(self, token, user_id=None):

        self.config_model_file = {'type_of_problem': '','no_of_nodes':'', 'no_of_layers':'', 'activations':'', 'metrics':''}
        assert isinstance(token, str), 'Token must be of type string'
        assert user_id is None or isinstance(user_id, int), 'user_id must be of type int (or None)'

        self.token = token  
        self.user_id = user_id
        self.param = ''
        self.filters = None
        self.chat_id = None  
        self.bot_active = False  
        
        self._status_message = "No status message was set"  
        self.lr = None
        self.modify_lr = 1.0  
        self.verbose = True   
        self.stop_train_flag = False  
        self.updater = None

        self.loss_hist = []
        self.val_loss_hist = []

        logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.startup_message = "Hi, I'm the DL bot! I will send you updates on your training process.\n" \
                               " send /make to create model.\n"\
                               " send /update to updated model configs.\n"\
                               " send /view to see the configs of your model.\n"\
                               " send /start to activate automatic updates every epoch\n" \
                               " send /help to see all options.\n" \
                               " Send /status to get the latest results.\n" \
                               " Send /quiet to stop getting automatic updates each epoch\n" \
                               " Send /plot to get a loss convergence plot.\n" \
                               
        self.model_type_message = " Select the type of model you want to build.\n" \
                                  " 1. Simple NN architecture.\n"\
                                  " 2. CNN Architecture.\n"     

        self.type_of_problem = " Select the type of problem:\n"\
        					   " 1. Classification\n"\
        					   " 2. Regression"

        self.create_message = " Choose how do you wanna create a model:\n" \
                              " 1. Create custom model.\n"\
                              " 2. Select some open source model.\nType \cancel to cancel the process"

        self.create_model = " Choose options to provide value:\n"\
                            " a. Set no of hidden layers.\n"\
                            " b. Set no of hidden nodes.\n"\
                            " c. Set the activation layer.\n"\
                            " d. Set the metric to be used.\n"\
                            " Send /done to save values."

        self.update_model = " Choose the param you want to update:\n"\
                            " 1. no of nodes\n"\
                            " 2. no of layers\n"\
                            " 3. activation function\n"\
                            " 4. regularization\n"\
                            " 5. metric\n"\
                            " Send /show to see the previous values\n"\
                            " Send /cancel to cancel the operation"\

    def activate_bot(self):

        self.updater = Updater(self.token)  
        
        dp = self.updater.dispatcher  
        dp.add_error_handler(self.showError)  

        self.filters = Filters.user(user_id=self.user_id) if self.user_id else None
        
        # Command and conversation handles
        # dp.add_handler(CommandHandler("update", self.modify, filters=self.filters))
        dp.add_handler(CommandHandler("start", self.start, filters=self.filters))
        dp.add_handler(CommandHandler("train", self.train, filters=self.filters))
        dp.add_handler(CommandHandler("help", self.help, filters=self.filters))
        dp.add_handler(CommandHandler("status", self.status, filters=self.filters))  
        dp.add_handler(CommandHandler("quiet", self.quiet, filters=self.filters))  
        dp.add_handler(CommandHandler("plot", self.plot_loss, filters=self.filters))  # /plot loss
        
        # update_handler = ConversationHandler(
        #     entry_points = [CommandHandler('update', self.modify)],
        #     states = {},
        #     fallbacks = [CommandHandler('cancel',self.cancel)]
        # )

        make_handler = ConversationHandler(
            entry_points=[CommandHandler('make',self.create)],
            states={
                modeltype: [MessageHandler(Filters.text,self.setmodeltype)],
                problem: [MessageHandler(Filters.text,self.setproblem)],
                makemodel: [MessageHandler(Filters.text,self.setupmodel)],
                modelconfig: [MessageHandler(Filters.text,self.make_config), CommandHandler('done',self.write_config)]
            },
          
            fallbacks=[CommandHandler('cancel',self.cancel)]
        )
        dp.add_handler(make_handler)
        # dp.add_handler(update_handler)

        self.updater.start_polling()
        self.bot_active = True
        print("Bot started")
    
        # Uncomment next line while debugging
        # self.updater.idle()
           

    def create(self,bot,update):
        keyboard = [['1','2']]
        update.message.reply_text(self.create_message, reply_markup=ReplyKeyboardMarkup(keyboard))
        self.chat_id = update.message.chat_id
        text = update.message.text    
        if text:
            return modeltype
        else:
            return CommandHandler.END
        
    def train(self,bot,update):
        print("train")
        score = kr.model_train(bot,update)
        if self.config_model_file['type_of_problem'] == 'Classification':
            for i in score:
                self.send_message(str(i))
    
        
    def setmodeltype(self,bot,update):
        print('inside setmodeltype')
        print(update.message.text)
        probkeyboard = [['Classification','regression']]
        update.message.reply_text(self.type_of_problem, reply_markup=ReplyKeyboardMarkup(probkeyboard))
        return problem
        
    def setproblem(self,bot,update):
        self.config_model_file['type_of_problem'] = update.message.text
        model_board = [['ANN','CNN']]
        update.message.reply_text(self.model_type_message, reply_markup=ReplyKeyboardMarkup(model_board))
        return makemodel
    
    def setupmodel(self,bot,update):
        if self.param:
            self.config_model_file[self.param] = update.message.text
            update.message.reply_text('You have entered '+ update.message.text+ ' for ' + self.param )
        param_board = [['a','b','c','d']]
        update.message.reply_text(self.create_model, reply_markup=ReplyKeyboardMarkup(param_board))
        return modelconfig

    def make_config(self,bot,update):        
        
        if update.message.text.lower() == 'a':
            param_board = [['a','b','c','d']]
            self.param = 'no_of_layers'
            update.message.reply_text("Chose any number between(1-100)",reply_markup=ReplyKeyboardRemove())
            return makemodel

        elif update.message.text.lower() == 'b':
            reply_board = [['32','64','128','256','512','1024','2048']]
            update.message.reply_text("Select the no of hidden nodes", reply_markup=ReplyKeyboardMarkup(reply_board))
            self.param = 'no_of_nodes'
            return makemodel

        elif update.message.text.lower() == 'c':
            reply_board = [['relu','softmax','sigmoid','tanh']]
            update.message.reply_text("select the activation type", reply_markup=ReplyKeyboardMarkup(reply_board))
            self.param = 'activations'
            return makemodel
     
        elif update.message.text.lower() == 'd':
            reply_board = [['Accuracy','recall','precision','roc','rmse']]
            update.message.reply_text("Select the metric required:", reply_markup=ReplyKeyboardMarkup(reply_board))
            self.param = 'metrics'
            return makemodel
             
    def write_config(self,bot,update):
        print("Value Recieved:", self.config_model_file, sep=" ")
        with open('config.json', 'w') as file:
            file.write(json.dumps(self.config_model_file))
        return ConversationHandler.END     

    def view_model(self):
        print("The model architecture is:")
    
    def modify(self,bot,update):
        print("for updating model!!")
        keyboard = [['1','2','3','4','5']]
        update.message.reply_text(self.update_model, reply_markup=ReplyKeyboardMarkup(keyboard))
        text = update.message.text    
        if text:
            print("text recieved")
        else:
            return CommandHandler.END
    
    def stop_bot(self):
        """ Function to stop the bot """
        self.updater.stop()
        self.bot_active = False
        
    def cancel_val(self,bot,update):
        
        config_model_file = {'no_of_nodes':'', 'no_of_layers':'', 'activations':'', 'metrics':''}
        update.message.reply_text('OK, Values is not modified or set.',
                                  reply_markup=ReplyKeyboardRemove())
        with open('config.txt') as file:
            for i in config_model_file.keys:
                file.write(i + ": " + config_model_file[i] + '\n')  

        return ConversationHandler.END
        

    def start(self, bot, update):
         """ Telegram bot callback for the /start command.
        Fetches chat_id, activates automatic epoch updates and sends startup message"""
        print(update)
        self.user_id = update.message.chat.id
        update.message.reply_text(self.startup_message, reply_markup=ReplyKeyboardRemove())
        self.chat_id = update.message.chat_id
        self.verbose = True

    def help(self, bot, update):
        """ Telegram bot callback for the /help command. Replies the startup message"""
        update.message.reply_text(self.startup_message, reply_markup=ReplyKeyboardRemove())
        self.chat_id = update.message.chat_id

    def quiet(self, bot, update):
        """ Telegram bot callback for the /quiet command. Stops automatic epoch updates"""
        self.verbose = False
        update.message.reply_text(" Automatic epoch updates turned off. Send /start to turn epoch updates back on.")

    def showError(self, *args):
        """Log Errors caused by Updates."""
        print(*args)
        self.logger.warning('Update "%s" caused error "%s"', args[0], args[1])

    def send_message(self,txt):
     
        assert isinstance(txt, str), 'Message text must be of type string'
        if self.chat_id is not None:
            self.updater.bot.send_message(chat_id=self.chat_id, text=txt)
        else:
            print('Send message failed, user did not send /start')

    def set_status(self, txt):
        """ Function to set a status message to be returned by the /status command """
        
        assert isinstance(txt, str), 'Status Message must be of type string'
        self._status_message = txt

    def status(self, bot, update):
        """ Telegram bot callback for the /status command. Replies with the latest status"""
        
        update.message.reply_text(self._status_message)

    def cancel(self,bot,update):
        user = update.message.from_user
        update.message.reply_text('Task Ended by user!! Config file not changed',
                                reply_markup=ReplyKeyboardRemove())
    
        return ConversationHandler.END


    # Stop training process callbacks
    # def stop_training(self, bot, update):
    #     """ Telegram bot callback for the /stoptraining command. Displays verification message with buttons"""
        
    #     reply_keyboard = [['Yes', 'No']]
    #     update.message.reply_text(
    #                 'Are you sure? '
    #                 'This will stop your training process!\n\n',
    #                 reply_markup=ReplyKeyboardMarkup(reply_keyboard))
    #     return 1

    # def stop_training_verify(self, bot, update):
    #     """ Telegram bot callback for the /stoptraining command. Handle user selection as part of conversation"""
        
    #     is_sure = update.message.text  # Get response
    #     if is_sure == 'Yes':
    #         self.stop_train_flag = True
    #         update.message.reply_text('OK, stopping training!', reply_markup=ReplyKeyboardRemove())
    #     elif is_sure == 'No':
    #         self.stop_train_flag = False  # to allow changing your mind before stop took place
    #         update.message.reply_text('OK, canceling stop request!', reply_markup=ReplyKeyboardRemove())

    #     return ConversationHandler.END

    # def cancel_stop(self, bot, update):
    #     """ Telegram bot callback for the /stoptraining command. Handle user cancellation as part of conversation"""
        
    #     self.stop_train_flag = False
    #     update.message.reply_text('OK, training will not be stopped.',
    #                               reply_markup=ReplyKeyboardRemove())
    #     return ConversationHandler.END

    # def stop_handler(self):
    #     """ Function to setup the callbacks for the /stoptraining command. Returns a conversation handler """
        
    #     conv_handler = ConversationHandler(
    #         entry_points=[CommandHandler('stoptraining', self.stop_training, filters=self.filters)],
    #         states={1: [RegexHandler('^(Yes|No)$', self.stop_training_verify)]},
    #         fallbacks=[CommandHandler('cancel', self.cancel_stop, filters=self.filters)])
    #     return conv_handler

    # Plot loss history
    def plot_loss(self, bot, update):
        """ Telegram bot callback for the /plot command. Replies with a convergence plot image"""

        if not self.loss_hist or plt is None:
            # First epoch wasn't finished or matplotlib isn't installed
            return
        loss_np = np.asarray(self.loss_hist)
        # Check if training has a validation set
        val_loss_np = np.asarray(self.val_loss_hist) if self.val_loss_hist else None
        legend_keys = ['loss', 'val_loss'] if self.val_loss_hist else ['loss']

        x = np.arange(len(loss_np))  # Epoch axes
        fig = plt.figure()
        ax = plt.axes()
        ax.plot(x, loss_np, 'b')  # Plot training loss
        if val_loss_np is not None:
            ax.plot(x, val_loss_np, 'r')  # Plot val loss
            
        plt.title('Loss Convergence')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        ax.legend(legend_keys)
        buffer = BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        update.message.reply_photo(buffer)  # Sent image to user
        
