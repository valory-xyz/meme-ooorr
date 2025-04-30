import os

import dotenv

from packages.dvilela.connections.tweepy.tweepy_wrapper import Twitter


dotenv.load_dotenv()

twitter = Twitter(
    consumer_key=os.getenv("TWEEPY_CONSUMER_API_KEY"),
    consumer_secret=os.getenv("TWEEPY_CONSUMER_API_KEY_SECRET"),
    access_token=os.getenv("TWEEPY_ACCESS_TOKEN"),
    access_token_secret=os.getenv("TWEEPY_ACCESS_TOKEN_SECRET"),
    bearer_token=os.getenv("TWEEPY_BEARER_TOKEN"),
)

twitter.post_tweet("Hello world!")
