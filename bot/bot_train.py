from dl_bot import DLBot
from telegram_bot_callback import TelegramCallback

telegram_token = "823412051:AAHEYubDdjV-aJqcFQvbWTOLgKPX5BMTnUo"  # replace TOKEN with your bot's token

# user id is optional, however highly recommended as it limits the access to you alone.
telegram_user_id = None   # replace None with your telegram user id (integer):

# Create a DLBot instance
bot = DLBot(token=telegram_token, user_id=telegram_user_id)
# Create a TelegramBotCallback instance
bot.activate_bot()
