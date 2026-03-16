#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023-2024 Valory AG
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

"""Tests for tweepy_wrapper.py."""

# pylint: disable=protected-access,redefined-outer-name

import datetime
import logging
from unittest.mock import MagicMock, patch

import pytest
import tweepy  # type: ignore[import]

from packages.dvilela.connections.tweepy.tweepy_wrapper import (
    DEFAULT_LOGGER,
    Twitter,
    is_twitter_id,
)

# ---------------------------------------------------------------------------
# is_twitter_id
# ---------------------------------------------------------------------------


class TestIsTwitterId:
    """Tests for the is_twitter_id utility function."""

    @pytest.mark.parametrize(
        "value",
        [
            "1",
            "12345678901234567890",  # 20 digits (max)
            "0",
            "99999999999",
        ],
    )
    def test_valid_ids(self, value: str) -> None:
        """Valid numeric strings up to 20 digits return True."""
        assert is_twitter_id(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            "",
            "abc",
            "123abc",
            "123456789012345678901",  # 21 digits
            " 123",
            "123 ",
            "-1",
            "12.34",
        ],
    )
    def test_invalid_ids(self, value: str) -> None:
        """Non-numeric or too-long strings return False."""
        assert is_twitter_id(value) is False


# ---------------------------------------------------------------------------
# Twitter class helpers
# ---------------------------------------------------------------------------

CONSUMER_KEY = "ck"
CONSUMER_SECRET = "cs"
ACCESS_TOKEN = "at"
ACCESS_TOKEN_SECRET = "ats"
BEARER_TOKEN = "bt"


@pytest.fixture()
def twitter_instance() -> Twitter:
    """Return a Twitter instance with all tweepy internals mocked."""
    with patch(
        "packages.dvilela.connections.tweepy.tweepy_wrapper.tweepy"
    ) as mock_tweepy:
        # Set up the mock so the constructor runs
        mock_tweepy.OAuth2BearerHandler.return_value = MagicMock()
        mock_tweepy.OAuth2AppHandler.return_value = MagicMock()
        mock_tweepy.OAuth1UserHandler.return_value = MagicMock()
        mock_tweepy.API.return_value = MagicMock()
        mock_tweepy.Client.return_value = MagicMock()
        # Keep the real exception hierarchy accessible
        mock_tweepy.errors.TweepyException = tweepy.errors.TweepyException
        mock_tweepy.Paginator = MagicMock()

        tw = Twitter(
            consumer_key=CONSUMER_KEY,
            consumer_secret=CONSUMER_SECRET,
            access_token=ACCESS_TOKEN,
            access_token_secret=ACCESS_TOKEN_SECRET,
            bearer_token=BEARER_TOKEN,
        )
        # Stash the mock for easy access in tests
        tw._mock_tweepy = mock_tweepy  # type: ignore[attr-defined]
    return tw


@pytest.fixture()
def twitter_with_logger() -> Twitter:
    """Return a Twitter instance initialised with a custom logger."""
    with patch(
        "packages.dvilela.connections.tweepy.tweepy_wrapper.tweepy"
    ) as mock_tweepy:
        mock_tweepy.OAuth2BearerHandler.return_value = MagicMock()
        mock_tweepy.OAuth2AppHandler.return_value = MagicMock()
        mock_tweepy.OAuth1UserHandler.return_value = MagicMock()
        mock_tweepy.API.return_value = MagicMock()
        mock_tweepy.Client.return_value = MagicMock()
        mock_tweepy.errors.TweepyException = tweepy.errors.TweepyException
        mock_tweepy.Paginator = MagicMock()

        custom_logger = logging.getLogger("test_custom")
        tw = Twitter(
            consumer_key=CONSUMER_KEY,
            consumer_secret=CONSUMER_SECRET,
            access_token=ACCESS_TOKEN,
            access_token_secret=ACCESS_TOKEN_SECRET,
            bearer_token=BEARER_TOKEN,
            logger=custom_logger,
        )
    return tw


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestTwitterInit:
    """Tests for the Twitter constructor."""

    def test_default_logger(self, twitter_instance: Twitter) -> None:
        """When no logger is supplied, DEFAULT_LOGGER is used."""
        assert twitter_instance.logger is DEFAULT_LOGGER

    def test_custom_logger(self, twitter_with_logger: Twitter) -> None:
        """When a logger is supplied, it is used instead of the default."""
        assert twitter_with_logger.logger is not DEFAULT_LOGGER
        assert twitter_with_logger.logger.name == "test_custom"



# ---------------------------------------------------------------------------
# post_tweet
# ---------------------------------------------------------------------------


class TestPostTweet:
    """Tests for Twitter.post_tweet."""

    def test_post_tweet_text_only(self, twitter_instance: Twitter) -> None:
        """Posting with text only returns the tweet id."""
        twitter_instance.client.create_tweet.return_value = MagicMock(
            data={"id": "111"}
        )
        result = twitter_instance.post_tweet(text="hello")
        assert result == "111"
        twitter_instance.client.create_tweet.assert_called_once_with(
            text="hello",
            media_ids=None,
            in_reply_to_tweet_id=None,
            quote_tweet_id=None,
        )

    def test_post_tweet_with_images(self, twitter_instance: Twitter) -> None:
        """Posting with image_paths uploads media and passes ids."""
        upload_result = MagicMock(media_id=42)
        twitter_instance.api.media_upload.return_value = upload_result
        twitter_instance.client.create_tweet.return_value = MagicMock(
            data={"id": "222"}
        )

        result = twitter_instance.post_tweet(text="pic", image_paths=["a.png", "b.png"])
        assert result == "222"
        assert twitter_instance.api.media_upload.call_count == 2
        twitter_instance.client.create_tweet.assert_called_once_with(
            text="pic",
            media_ids=[42, 42],
            in_reply_to_tweet_id=None,
            quote_tweet_id=None,
        )

    def test_post_tweet_with_reply_and_quote(self, twitter_instance: Twitter) -> None:
        """Optional reply-to and quote ids are forwarded."""
        twitter_instance.client.create_tweet.return_value = MagicMock(
            data={"id": "333"}
        )
        result = twitter_instance.post_tweet(
            text="reply", in_reply_to_tweet_id=10, quote_tweet_id=20
        )
        assert result == "333"
        twitter_instance.client.create_tweet.assert_called_once_with(
            text="reply",
            media_ids=None,
            in_reply_to_tweet_id=10,
            quote_tweet_id=20,
        )

    def test_post_tweet_tweepy_exception(self, twitter_instance: Twitter) -> None:
        """Verify TweepyException is logged and re-raised."""
        twitter_instance.client.create_tweet.side_effect = (
            tweepy.errors.TweepyException("boom")
        )
        with pytest.raises(tweepy.errors.TweepyException):
            twitter_instance.post_tweet(text="fail")


# ---------------------------------------------------------------------------
# delete_tweet
# ---------------------------------------------------------------------------


class TestDeleteTweet:
    """Tests for Twitter.delete_tweet."""

    def test_delete_tweet_success(self, twitter_instance: Twitter) -> None:
        """Successful deletion returns True."""
        assert twitter_instance.delete_tweet("123") is True
        twitter_instance.client.delete_tweet.assert_called_once_with("123")

    def test_delete_tweet_exception(self, twitter_instance: Twitter) -> None:
        """Verify TweepyException is logged and re-raised."""
        twitter_instance.client.delete_tweet.side_effect = (
            tweepy.errors.TweepyException("err")
        )
        with pytest.raises(tweepy.errors.TweepyException):
            twitter_instance.delete_tweet("123")


# ---------------------------------------------------------------------------
# like_tweet / unlike_tweet
# ---------------------------------------------------------------------------


class TestLikeUnlike:
    """Tests for like_tweet and unlike_tweet."""

    def test_like_tweet_success(self, twitter_instance: Twitter) -> None:
        """Successful like returns True."""
        assert twitter_instance.like_tweet("1") is True
        twitter_instance.client.like.assert_called_once_with("1")

    def test_like_tweet_exception(self, twitter_instance: Twitter) -> None:
        """Verify TweepyException on like is re-raised."""
        twitter_instance.client.like.side_effect = tweepy.errors.TweepyException("err")
        with pytest.raises(tweepy.errors.TweepyException):
            twitter_instance.like_tweet("1")

    def test_unlike_tweet_success(self, twitter_instance: Twitter) -> None:
        """Successful unlike returns True."""
        assert twitter_instance.unlike_tweet("1") is True
        twitter_instance.client.unlike.assert_called_once_with("1")

    def test_unlike_tweet_exception(self, twitter_instance: Twitter) -> None:
        """Verify TweepyException on unlike is re-raised."""
        twitter_instance.client.unlike.side_effect = tweepy.errors.TweepyException(
            "err"
        )
        with pytest.raises(tweepy.errors.TweepyException):
            twitter_instance.unlike_tweet("1")


# ---------------------------------------------------------------------------
# retweet / unretweet
# ---------------------------------------------------------------------------


class TestRetweetUnretweet:
    """Tests for retweet and unretweet."""

    def test_retweet_success(self, twitter_instance: Twitter) -> None:
        """Successful retweet returns True."""
        assert twitter_instance.retweet("1") is True
        twitter_instance.client.retweet.assert_called_once_with("1")

    def test_retweet_exception(self, twitter_instance: Twitter) -> None:
        """Verify TweepyException on retweet is re-raised."""
        twitter_instance.client.retweet.side_effect = tweepy.errors.TweepyException(
            "err"
        )
        with pytest.raises(tweepy.errors.TweepyException):
            twitter_instance.retweet("1")

    def test_unretweet_success(self, twitter_instance: Twitter) -> None:
        """Successful unretweet returns True."""
        assert twitter_instance.unretweet("1") is True
        twitter_instance.client.unretweet.assert_called_once_with("1")

    def test_unretweet_exception(self, twitter_instance: Twitter) -> None:
        """Verify TweepyException on unretweet is re-raised."""
        twitter_instance.client.unretweet.side_effect = tweepy.errors.TweepyException(
            "err"
        )
        with pytest.raises(tweepy.errors.TweepyException):
            twitter_instance.unretweet("1")


# ---------------------------------------------------------------------------
# follow_by_id / follow_by_username / unfollow_by_id
# ---------------------------------------------------------------------------


class TestFollow:
    """Tests for follow and unfollow methods."""

    def test_follow_by_id_success(self, twitter_instance: Twitter) -> None:
        """Successful follow_by_id returns True."""
        assert twitter_instance.follow_by_id("42") is True
        twitter_instance.client.follow.assert_called_once_with("42")

    def test_follow_by_id_exception(self, twitter_instance: Twitter) -> None:
        """Verify TweepyException on follow_by_id is re-raised."""
        twitter_instance.client.follow.side_effect = tweepy.errors.TweepyException(
            "err"
        )
        with pytest.raises(tweepy.errors.TweepyException):
            twitter_instance.follow_by_id("42")

    def test_follow_by_username_success(self, twitter_instance: Twitter) -> None:
        """Method follow_by_username resolves the id then delegates to follow_by_id."""
        twitter_instance.client.get_user.return_value = MagicMock(
            data=MagicMock(id="99")
        )
        assert twitter_instance.follow_by_username("alice") is True
        twitter_instance.client.follow.assert_called_once_with("99")

    def test_follow_by_username_user_not_found(self, twitter_instance: Twitter) -> None:
        """When get_user_id returns a falsy value, follow_by_username returns False."""
        twitter_instance.client.get_user.return_value = MagicMock(
            data=MagicMock(id=None)
        )
        assert twitter_instance.follow_by_username("ghost") is False

    def test_unfollow_by_id_success(self, twitter_instance: Twitter) -> None:
        """Successful unfollow_by_id returns True."""
        assert twitter_instance.unfollow_by_id("42") is True
        twitter_instance.client.unfollow.assert_called_once_with("42")

    def test_unfollow_by_id_exception(self, twitter_instance: Twitter) -> None:
        """Verify TweepyException on unfollow_by_id is re-raised."""
        twitter_instance.client.unfollow.side_effect = tweepy.errors.TweepyException(
            "err"
        )
        with pytest.raises(tweepy.errors.TweepyException):
            twitter_instance.unfollow_by_id("42")


# ---------------------------------------------------------------------------
# get_user_id
# ---------------------------------------------------------------------------


class TestGetUserId:
    """Tests for get_user_id."""

    def test_get_user_id_success(self, twitter_instance: Twitter) -> None:
        """Successful get_user_id returns the user id."""
        twitter_instance.client.get_user.return_value = MagicMock(
            data=MagicMock(id="100")
        )
        assert twitter_instance.get_user_id("bob") == "100"

    def test_get_user_id_exception(self, twitter_instance: Twitter) -> None:
        """Verify TweepyException on get_user_id is re-raised."""
        twitter_instance.client.get_user.side_effect = tweepy.errors.TweepyException(
            "err"
        )
        with pytest.raises(tweepy.errors.TweepyException):
            twitter_instance.get_user_id("bob")


# ---------------------------------------------------------------------------
# get_me
# ---------------------------------------------------------------------------


class TestGetMe:
    """Tests for get_me."""

    def test_get_me_success(self, twitter_instance: Twitter) -> None:
        """Successful get_me returns user dict."""
        mock_data = MagicMock(id="1", username="me")
        mock_data.name = "Me"
        twitter_instance.client.get_me.return_value = MagicMock(data=mock_data)
        result = twitter_instance.get_me()
        assert result == {"user_id": "1", "username": "me", "display_name": "Me"}

    def test_get_me_exception(self, twitter_instance: Twitter) -> None:
        """Verify TweepyException on get_me is re-raised."""
        twitter_instance.client.get_me.side_effect = tweepy.errors.TweepyException(
            "err"
        )
        with pytest.raises(tweepy.errors.TweepyException):
            twitter_instance.get_me()


# ---------------------------------------------------------------------------
# get_follower_ids
# ---------------------------------------------------------------------------


class TestGetFollowerIds:
    """Tests for get_follower_ids."""

    def test_get_follower_ids_with_numeric_id(self, twitter_instance: Twitter) -> None:
        """When a numeric id is passed, it is used directly (no get_user_id call)."""
        twitter_instance.api.get_follower_ids.return_value = [1, 2, 3]
        result = twitter_instance.get_follower_ids("12345")
        assert result == [1, 2, 3]
        twitter_instance.api.get_follower_ids.assert_called_once_with(user_id="12345")
        # get_user should NOT have been called
        twitter_instance.client.get_user.assert_not_called()

    def test_get_follower_ids_with_username(self, twitter_instance: Twitter) -> None:
        """When a username is passed, get_user_id is called first."""
        twitter_instance.client.get_user.return_value = MagicMock(
            data=MagicMock(id="999")
        )
        twitter_instance.api.get_follower_ids.return_value = [4, 5]
        result = twitter_instance.get_follower_ids("alice")
        assert result == [4, 5]
        twitter_instance.client.get_user.assert_called_once_with(username="alice")
        twitter_instance.api.get_follower_ids.assert_called_once_with(user_id="999")

    def test_get_follower_ids_user_not_found(self, twitter_instance: Twitter) -> None:
        """When get_user_id returns falsy, None is returned."""
        twitter_instance.client.get_user.return_value = MagicMock(
            data=MagicMock(id=None)
        )
        result = twitter_instance.get_follower_ids("ghost")
        assert result is None

    def test_get_follower_ids_exception(self, twitter_instance: Twitter) -> None:
        """Verify TweepyException from get_follower_ids is re-raised."""
        twitter_instance.api.get_follower_ids.side_effect = (
            tweepy.errors.TweepyException("err")
        )
        with pytest.raises(tweepy.errors.TweepyException):
            twitter_instance.get_follower_ids("12345")


# ---------------------------------------------------------------------------
# get_all_user_tweets
# ---------------------------------------------------------------------------


class TestGetAllUserTweets:
    """Tests for get_all_user_tweets."""

    def test_get_all_user_tweets_with_data(self, twitter_instance: Twitter) -> None:
        """Tweets from multiple pages are aggregated."""
        page1 = MagicMock(data=["t1", "t2"])
        page2 = MagicMock(data=["t3"])

        with patch(
            "packages.dvilela.connections.tweepy.tweepy_wrapper.tweepy.Paginator"
        ) as mock_paginator:
            mock_paginator.return_value = iter([page1, page2])
            result = twitter_instance.get_all_user_tweets(
                user_id="1",
                max_results=50,
                tweet_fields=["public_metrics"],
                start_time=datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc),
            )
        assert result == ["t1", "t2", "t3"]

    def test_get_all_user_tweets_empty_page(self, twitter_instance: Twitter) -> None:
        """A page with no data triggers a warning but does not break."""
        page1 = MagicMock(data=["t1"])
        page2 = MagicMock(data=None)

        with patch(
            "packages.dvilela.connections.tweepy.tweepy_wrapper.tweepy.Paginator"
        ) as mock_paginator:
            mock_paginator.return_value = iter([page1, page2])
            result = twitter_instance.get_all_user_tweets(user_id="1")
        assert result == ["t1"]

    def test_get_all_user_tweets_exception(self, twitter_instance: Twitter) -> None:
        """Verify TweepyException is logged and re-raised."""
        with patch(
            "packages.dvilela.connections.tweepy.tweepy_wrapper.tweepy.Paginator"
        ) as mock_paginator:
            mock_paginator.return_value.__iter__ = MagicMock(
                side_effect=tweepy.errors.TweepyException("err")
            )
            with pytest.raises(tweepy.errors.TweepyException):
                twitter_instance.get_all_user_tweets(user_id="1")

    def test_get_all_user_tweets_defaults(self, twitter_instance: Twitter) -> None:
        """Defaults (max_results=100, no fields, no start_time) are passed correctly."""
        with patch(
            "packages.dvilela.connections.tweepy.tweepy_wrapper.tweepy.Paginator"
        ) as mock_paginator:
            mock_paginator.return_value = iter([])
            result = twitter_instance.get_all_user_tweets(user_id="1")
        assert result == []
        mock_paginator.assert_called_once_with(
            twitter_instance.client.get_users_tweets,
            "1",
            max_results=100,
            tweet_fields=None,
            start_time=None,
        )
