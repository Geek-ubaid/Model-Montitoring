from dl_bot import DLBot
from telegram_bot_callback import TelegramCallback
import os
from dotenv import load_dotenv

load_dotenv()

telegram_token = os.getenv('TOKEN') 

telegram_user_id = None   # replace None with your telegram user id (integer):

bot = DLBot(token=telegram_token, user_id=telegram_user_id)
bot.activate_bot()