import logging
<<<<<<< HEAD

# Configure the logger
logging.basicConfig(
    level=logging.INFO,  # Set logging level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Define log format
    handlers=[
        logging.StreamHandler(),  # Print logs to console
        logging.FileHandler("trading_bot/logs/trading_bot.log")  # Save logs to a file
    ]
)

# Create logger instance
logger = logging.getLogger("trading_bot")
=======
import os

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

def setup_logger(name, log_file, level=logging.INFO):
    """Function to set up a logger that writes to a file."""
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger
>>>>>>> 7d64dceb (Updated feature XYZ)
