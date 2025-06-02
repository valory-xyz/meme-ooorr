#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2021-2024 David Vilela Freire
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------

"""Tweepy wrapper."""

import logging
import re
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, cast

import tweepy  # type: ignore[import]


# Type variable for generic decorator
F = TypeVar("F", bound=Callable[..., Any])

DEFAULT_LOGGER = logging.getLogger(__name__)


def is_twitter_id(twitter_id: str) -> bool:
    """
    Check if a string is a valid Twitter ID.

    Args:
        twitter_id (str): The string to check.

    Returns:
        bool: True if the string is a valid Twitter ID, False otherwise.
    """
    return bool(re.match(r"^\d{1,20}$", twitter_id))


def _log_tweepy_exceptions(default_return_on_error: Any = None) -> Callable[[F], F]:
    """Decorator to catch TweepyExceptions, log them with details, and return a default value."""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(self: "Twitter", *args: Any, **kwargs: Any) -> Any:
            logger = getattr(self, "logger", DEFAULT_LOGGER)
            method_name = func.__name__
            try:
                return func(self, *args, **kwargs)
            except tweepy.errors.TooManyRequests as e:
                log_msg = (
                    f"Tweepy API rate limit hit (TooManyRequests) in {method_name}: {e}"
                )
                if hasattr(e, "response") and e.response is not None:
                    log_msg += f" | Response: {e.response.text if hasattr(e.response, 'text') else e.response}"
                if hasattr(e, "api_errors") and e.api_errors:
                    log_msg += f" | API Errors: {e.api_errors}"
                logger.error(log_msg)
            except tweepy.errors.Unauthorized as e:
                log_msg = f"Tweepy API Unauthorized (401) in {method_name}: {e}"
                if hasattr(e, "response") and e.response is not None:
                    log_msg += f" | Response: {e.response.text if hasattr(e.response, 'text') else e.response}"
                if hasattr(e, "api_errors") and e.api_errors:
                    log_msg += f" | API Errors: {e.api_errors}"
                logger.error(log_msg)
            except tweepy.errors.Forbidden as e:
                log_msg = f"Tweepy API Forbidden (403) in {method_name}: {e}"
                if hasattr(e, "response") and e.response is not None:
                    log_msg += f" | Response: {e.response.text if hasattr(e.response, 'text') else e.response}"
                if hasattr(e, "api_errors") and e.api_errors:
                    log_msg += f" | API Errors: {e.api_errors}"
                logger.error(log_msg)
            except tweepy.errors.NotFound as e:
                log_msg = f"Tweepy API NotFound (404) in {method_name}: {e}"
                if hasattr(e, "response") and e.response is not None:
                    log_msg += f" | Response: {e.response.text if hasattr(e.response, 'text') else e.response}"
                if hasattr(e, "api_errors") and e.api_errors:
                    log_msg += f" | API Errors: {e.api_errors}"
                logger.warning(log_msg)  # NotFound might be expected in some cases
            except tweepy.errors.BadRequest as e:
                log_msg = f"Tweepy API BadRequest (400) in {method_name}: {e}"
                if hasattr(e, "response") and e.response is not None:
                    log_msg += f" | Response: {e.response.text if hasattr(e.response, 'text') else e.response}"
                if hasattr(e, "api_errors") and e.api_errors:
                    log_msg += f" | API Errors: {e.api_errors}"
                logger.error(log_msg)
            except tweepy.errors.TwitterServerError as e:
                log_msg = f"Twitter Server Error (5xx) in {method_name}: {e}"
                if hasattr(e, "response") and e.response is not None:
                    log_msg += f" | Response: {e.response.text if hasattr(e.response, 'text') else e.response}"
                if hasattr(e, "api_errors") and e.api_errors:
                    log_msg += f" | API Errors: {e.api_errors}"
                logger.error(log_msg)
            except tweepy.errors.HTTPException as e:  # Catch other HTTP exceptions
                log_msg = f"Generic Tweepy HTTPException in {method_name}: {type(e).__name__} - {e}"
                if hasattr(e, "response") and e.response is not None:
                    log_msg += f" | Response: {e.response.text if hasattr(e.response, 'text') else e.response}"
                if hasattr(e, "api_errors") and e.api_errors:
                    log_msg += f" | API Errors: {e.api_errors}"
                if hasattr(e, "api_codes") and e.api_codes:
                    log_msg += f" | API Codes: {e.api_codes}"
                if hasattr(e, "api_messages") and e.api_messages:
                    log_msg += f" | API Messages: {e.api_messages}"
                logger.error(log_msg)
            except (
                tweepy.errors.TweepyException
            ) as e:  # Catch any other Tweepy exception
                logger.error(
                    f"Non-HTTP TweepyException in {method_name}: {type(e).__name__} - {e}"
                )
            return default_return_on_error

        return cast(F, wrapper)

    return decorator


class Twitter:
    """A class to interact with Twitter API using Tweepy."""

    def __init__(
        self,
        consumer_key: str,
        consumer_secret: str,
        access_token: str,
        access_token_secret: str,
        bearer_token: str,
        logger: Optional[logging.Logger] = None,
    ):
        """Constructor"""
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
        self.logger = logger if logger else DEFAULT_LOGGER

    @_log_tweepy_exceptions(default_return_on_error=None)
    def post_tweet(
        self,
        text: str,
        image_paths: Optional[List[str]] = None,
        in_reply_to_tweet_id: Optional[int] = None,
    ) -> Optional[str]:
        """
        Posts a new tweet with optional media.

        Args:
            text (str): The content of the tweet.
            media_ids (list of int, optional): A list of media IDs to attach to the tweet. Defaults to None.
        """
        tweet = self.client.create_tweet(
            text=text,
            media_ids=(
                [
                    self.api.media_upload(filename=image_path).media_id
                    for image_path in image_paths
                ]
                if image_paths
                else None
            ),
            in_reply_to_tweet_id=in_reply_to_tweet_id,
        )
        return tweet.data["id"]

    @_log_tweepy_exceptions(default_return_on_error=False)
    def delete_tweet(self, tweet_id: str) -> bool:
        """
        Deletes a specific tweet.

        Args:
            tweet_id (int): The ID of the tweet to delete.
        """
        self.client.delete_tweet(tweet_id)
        return True

    @_log_tweepy_exceptions(default_return_on_error=False)
    def like_tweet(self, tweet_id: str) -> bool:
        """
        Likes a specific tweet.

        Args:
            tweet_id (int): The ID of the tweet to like.
        """
        self.client.like(tweet_id)
        return True

    @_log_tweepy_exceptions(default_return_on_error=False)
    def unlike_tweet(self, tweet_id: str) -> bool:
        """
        Unlikes a specific tweet.

        Args:
            tweet_id (int): The ID of the tweet to like.
        """
        self.client.unlike(tweet_id)
        return True

    @_log_tweepy_exceptions(default_return_on_error=False)
    def retweet(self, tweet_id: str) -> bool:
        """
        Retweets a specific tweet.

        Args:
            tweet_id (int): The ID of the tweet to retweet.
        """
        self.client.retweet(tweet_id)
        return True

    @_log_tweepy_exceptions(default_return_on_error=False)
    def unretweet(self, tweet_id: str) -> bool:
        """
        Unlikes a specific tweet.

        Args:
            tweet_id (int): The ID of the tweet to unretweet.
        """
        self.client.unretweet(tweet_id)
        return True

    @_log_tweepy_exceptions(default_return_on_error=False)
    def follow_by_id(self, user_id: str) -> bool:
        """
        Follow a specific user.

        Args:
            user_id (int): The ID of the user to follow.
        """
        self.client.follow(user_id)
        return True

    @_log_tweepy_exceptions(default_return_on_error=False)
    def follow_by_username(self, username: str) -> bool:
        """
        Follow a specific user.

        Args:user_name
            user_id (int): The ID of the user to follow.
        """
        user_id = self.get_user_id(username=username)
        if not user_id:
            self.logger.error(
                f"Could not follow user by username {username} because their ID could not be retrieved."
            )
            return False
        return self.follow_by_id(user_id)

    @_log_tweepy_exceptions(default_return_on_error=False)
    def unfollow_by_id(self, user_id: str) -> bool:
        """
        Unfollow a specific tweet.

        Args:
            user_id (int): The ID of the user to unfollow.
        """
        self.client.unfollow(user_id)
        return True

    @_log_tweepy_exceptions(default_return_on_error=None)
    def get_user_id(self, username: str) -> Optional[str]:
        """
        Get a user id.

        Args:
            username (str): The username of the user to retrieve.
        """
        user = self.client.get_user(username=username)
        return user.data.id

    @_log_tweepy_exceptions(default_return_on_error=None)
    def get_me(self) -> Optional[Dict]:
        """
        Get my user.

        Args:
            user_id (int): The ID of the user to unfollow.
        """
        result = self.client.get_me()
        return {"user_id": result.data.id, "username": result.data.username}

    @_log_tweepy_exceptions(default_return_on_error=None)
    def get_follower_ids(self, user: str) -> Optional[List[str]]:
        """
        Get a list of follower ids.

        Args:
            user_id (int): The ID of the user to unfollow.
        """
        user_id = user if is_twitter_id(user) else self.get_user_id(user)
        if not user_id:
            self.logger.error(
                f"Could not get follower IDs for {user} because their ID could not be retrieved."
            )
            return None
        result = self.api.get_follower_ids(user_id=user_id)
        return result
