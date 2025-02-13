import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

def get_env_variable(var_name):
    """Get the environment variable or raise an error if not set."""
    value = os.getenv(var_name)
    if value is None:
        raise ValueError(f"Missing required environment variable: {var_name}")
    return value

# KuCoin API Credentials
KUCOIN_API_KEY = get_env_variable('KUCOIN_API_KEY')
KUCOIN_SECRET = get_env_variable('KUCOIN_SECRET')
KUCOIN_PASSPHRASE = get_env_variable('KUCOIN_PASSPHRASE')

# Trading settings
TOTAL_CAPITAL = 10000           # Account size
MAX_ALLOCATION = 0.5            # Use only 50% of account capital for trading
MAX_TRADES = 5                  # Maximum concurrent trades
CHECK_INTERVAL = 15 * 60        # 15-minute interval

# Machine Learning settings
ML_INPUT_SIZE = 10              # Number of input features
ML_HIDDEN_SIZE = 32
ML_OUTPUT_SIZE = 1
ONLINE_BATCH_SIZE = 32

# Social Media/News API Settings
CMC_API_KEY = get_env_variable('CMC_API_KEY')
TWITTER_API_KEY = get_env_variable('TWITTER_API_KEY')

# AWS Deployment Settings (if needed)
AWS_REGION = 'ap-northeast-1'