from typing import Optional

import tweepy


class Twitter:
    """A class to interact with Twitter API using Tweepy."""

    def __init__(
        self,
        consumer_key,
        consumer_secret,
        access_token,
        access_token_secret,
        bearer_token,
    ):
        self.oauth2_bearer_auth = tweepy.OAuth2BearerHandler(bearer_token)
        self.oauth2_app_auth = tweepy.OAuth2AppHandler(consumer_key, consumer_secret)
        self.oauth1_user_auth = tweepy.OAuth1UserHandler(
            consumer_key=consumer_key,
            consumer_secret=consumer_secret,
            access_token=access_token,
            access_token_secret=access_token_secret,
        )

        self.api = tweepy.API(auth=self.oauth1_user_auth)

        self.client = tweepy.Client(
            bearer_token=bearer_token,
            consumer_key=consumer_key,
            consumer_secret=consumer_secret,
            access_token=access_token,
            access_token_secret=access_token_secret,
        )

    def post_tweet(
        self, text: str, image_paths=None, in_replay_to_tweet_id=None
    ) -> Optional[int]:
        """
        Posts a new tweet with optional media.

        Args:
            text (str): The content of the tweet.
            media_ids (list of int, optional): A list of media IDs to attach to the tweet. Defaults to None.
        """
        try:
            tweet = self.client.create_tweet(
                text=text,
                media_ids=[
                    self.api.media_upload(filename=image_path).media_id
                    for image_path in image_paths
                ]
                if image_paths
                else None,
                in_replay_to_tweet_id=in_replay_to_tweet_id,
            )
            return tweet.data["id"]
        except tweepy.TweepyException:
            return None

    def delete_tweet(self, tweet_id) -> bool:
        """
        Deletes a specific tweet.

        Args:
            tweet_id (int): The ID of the tweet to delete.
        """
        try:
            self.client.delete_tweet(tweet_id)
            return True
        except tweepy.TweepyException:
            return False

    def like_tweet(self, tweet_id) -> bool:
        """
        Likes a specific tweet.

        Args:
            tweet_id (int): The ID of the tweet to like.
        """
        try:
            self.client.like(tweet_id)
            return True
        except tweepy.TweepyException:
            return False

    def unlike_tweet(self, tweet_id) -> bool:
        """
        Unlikes a specific tweet.

        Args:
            tweet_id (int): The ID of the tweet to like.
        """
        try:
            self.client.unlike(tweet_id)
            return True
        except tweepy.TweepyException:
            return False

    def retweet(self, tweet_id) -> bool:
        """
        Retweets a specific tweet.

        Args:
            tweet_id (int): The ID of the tweet to retweet.
        """
        try:
            self.client.retweet(tweet_id)
            return True
        except tweepy.TweepyException:
            return False

    def unretweet(self, tweet_id) -> bool:
        """
        Unlikes a specific tweet.

        Args:
            tweet_id (int): The ID of the tweet to unretweet.
        """
        try:
            self.client.unretweet(tweet_id)
            return True
        except tweepy.TweepyException:
            return False

    def follow(self, user_id) -> bool:
        """
        Follow a specific tweet.

        Args:
            user_id (int): The ID of the user to follow.
        """
        try:
            self.client.follow(user_id)
            return True
        except tweepy.TweepyException:
            return False

    def unfollow(self, user_id) -> bool:
        """
        Unfollow a specific tweet.

        Args:
            user_id (int): The ID of the user to unfollow.
        """
        try:
            self.client.unfollow(user_id)
            return True
        except tweepy.TweepyException:
            return False

    def get_user_id(self, username) -> Optional[int]:
        """
        Get a user id.

        Args:
            username (str): The username of the user to retrieve.
        """
        try:
            user = self.client.get_user(username=username)
            return user.data.id
        except tweepy.TweepyException:
            return None
