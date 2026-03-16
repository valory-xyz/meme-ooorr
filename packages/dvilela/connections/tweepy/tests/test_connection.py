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

"""Tests for connection.py (TweepyConnection)."""

# pylint: disable=protected-access,unused-argument,too-few-public-methods

import json
import logging
from datetime import datetime, timezone
from typing import Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

from packages.dvilela.connections.tweepy.connection import (
    MAX_POST_RETRIES,
    PUBLIC_ID,
    SrrDialogues,
    TweepyConnection,
)
from packages.valory.protocols.srr.message import SrrMessage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_connection(
    tweepy_skip_auth: bool = False,
    twitter: Optional[MagicMock] = None,
) -> TweepyConnection:
    """Build a TweepyConnection with mocked internals.

    Since __init__ is pragma-no-cover, we create the object without calling
    __init__ and manually assign the attributes that the methods rely on.

    :param tweepy_skip_auth: whether to skip authentication.
    :param twitter: optional mock Twitter instance.
    :return: a TweepyConnection instance.
    """
    conn = object.__new__(TweepyConnection)
    conn.logger = logging.getLogger("test_connection")
    conn.tweepy_skip_auth = tweepy_skip_auth
    conn.twitter = twitter
    conn.dialogues = SrrDialogues(connection_id=PUBLIC_ID)
    return conn


def _mock_twitter() -> MagicMock:
    """Return a fresh mock Twitter wrapper."""
    return MagicMock()


# ---------------------------------------------------------------------------
# SrrDialogues
# ---------------------------------------------------------------------------


class TestSrrDialogues:
    """Tests for the SrrDialogues helper class."""

    def test_instantiation(self) -> None:
        """Verify SrrDialogues can be created with connection_id."""
        dialogues = SrrDialogues(connection_id=PUBLIC_ID)
        assert dialogues is not None


# ---------------------------------------------------------------------------
# main / on_connect / on_disconnect (no-op methods)
# ---------------------------------------------------------------------------


class TestNoOpMethods:
    """Tests for the empty lifecycle methods."""

    def test_main(self) -> None:
        """Test that main() returns None."""
        conn = _make_connection()
        conn.main()

    def test_on_connect(self) -> None:
        """Test that on_connect() returns None."""
        conn = _make_connection()
        conn.on_connect()

    def test_on_disconnect(self) -> None:
        """Test that on_disconnect() returns None."""
        conn = _make_connection()
        conn.on_disconnect()


# ---------------------------------------------------------------------------
# _get_response – validation paths
# ---------------------------------------------------------------------------


class TestGetResponseValidation:
    """Tests for _get_response validation branches."""

    def test_missing_method_key(self) -> None:
        """Payload without 'method' returns error."""
        conn = _make_connection(twitter=_mock_twitter())
        result = conn._get_response({"kwargs": {}})
        assert "error" in result
        assert (
            "missing" in result["error"].lower()
            or "required" in result["error"].lower()
        )

    def test_missing_kwargs_key(self) -> None:
        """Payload without 'kwargs' returns error."""
        conn = _make_connection(twitter=_mock_twitter())
        result = conn._get_response({"method": "get_me"})
        assert "error" in result

    def test_missing_both_keys(self) -> None:
        """Completely empty payload returns error."""
        conn = _make_connection(twitter=_mock_twitter())
        result = conn._get_response({})
        assert "error" in result

    def test_unknown_method(self) -> None:
        """An unrecognised method name returns error."""
        conn = _make_connection(twitter=_mock_twitter())
        result = conn._get_response({"method": "hack_the_planet", "kwargs": {}})
        assert "error" in result
        assert "hack_the_planet" in result["error"]

    def test_skip_auth_true(self) -> None:
        """When tweepy_skip_auth is True, an error is returned."""
        conn = _make_connection(tweepy_skip_auth=True, twitter=None)
        result = conn._get_response({"method": "get_me", "kwargs": {}})
        assert "error" in result
        assert "disabled" in result["error"].lower()

    def test_twitter_is_none(self) -> None:
        """When self.twitter is None (bad creds), an error is returned."""
        conn = _make_connection(tweepy_skip_auth=False, twitter=None)
        result = conn._get_response({"method": "get_me", "kwargs": {}})
        assert "error" in result
        assert "not initialized" in result["error"].lower()

    def test_method_dispatch_success(self) -> None:
        """A valid method is dispatched and its return value used."""
        tw = _mock_twitter()
        tw.get_me.return_value = {
            "user_id": "1",
            "username": "me",
            "display_name": "Me",
        }
        conn = _make_connection(twitter=tw)
        result = conn._get_response({"method": "get_me", "kwargs": {}})
        assert result == {"user_id": "1", "username": "me", "display_name": "Me"}

    def test_method_dispatch_exception(self) -> None:
        """When the dispatched method raises, an error dict is returned."""
        tw = _mock_twitter()
        conn = _make_connection(twitter=tw)
        # Manually make like_tweet raise
        tw.like_tweet.side_effect = Exception("api down")
        # like_tweet on the connection delegates to tw.like_tweet
        result = conn._get_response(
            {"method": "like_tweet", "kwargs": {"tweet_id": "1"}}
        )
        assert "error" in result
        assert "api down" in result["error"]


# ---------------------------------------------------------------------------
# on_send
# ---------------------------------------------------------------------------


class TestOnSend:
    """Tests for on_send."""

    @staticmethod
    def _make_envelope(
        performative: SrrMessage.Performative = SrrMessage.Performative.REQUEST,
        payload: Optional[str] = None,
    ) -> MagicMock:
        """Create a mock envelope with the given SrrMessage performative."""
        if payload is None:
            payload = json.dumps({"method": "get_me", "kwargs": {}})

        msg = MagicMock(spec=SrrMessage)
        msg.performative = performative
        msg.payload = payload
        msg.dialogue_reference = ("", "")
        msg.message_id = 1
        msg.target = 0
        msg.sender = "agent"
        msg.to = str(PUBLIC_ID)

        envelope = MagicMock()
        envelope.message = msg
        envelope.sender = "agent"
        envelope.to = str(PUBLIC_ID)
        envelope.context = None
        return envelope

    def test_non_request_performative(self) -> None:
        """Non-REQUEST performatives are rejected with a log message."""
        conn = _make_connection(twitter=_mock_twitter())
        envelope = self._make_envelope(
            performative=SrrMessage.Performative.RESPONSE,
        )
        # Update the dialogues mock so it doesn't block
        conn.dialogues = MagicMock()
        conn.put_envelope = MagicMock()  # type: ignore[method-assign]

        conn.on_send(envelope)
        # put_envelope should NOT have been called
        conn.put_envelope.assert_not_called()

    @patch("packages.dvilela.connections.tweepy.connection.Envelope")
    def test_request_performative_happy_path(
        self, mock_envelope_cls: MagicMock
    ) -> None:
        """A valid REQUEST envelope is processed and a response envelope put."""
        tw = _mock_twitter()
        tw.get_me.return_value = {
            "user_id": "1",
            "username": "me",
            "display_name": "Me",
        }
        conn = _make_connection(twitter=tw)

        envelope = self._make_envelope()

        # We need dialogues.update to return a dialogue mock
        mock_dialogue = MagicMock()
        mock_response_msg = MagicMock()
        mock_dialogue.reply.return_value = mock_response_msg
        conn.dialogues = MagicMock()
        conn.dialogues.update.return_value = mock_dialogue
        conn.put_envelope = MagicMock()  # type: ignore[method-assign]

        conn.on_send(envelope)
        conn.put_envelope.assert_called_once()

    @patch("packages.dvilela.connections.tweepy.connection.Envelope")
    def test_invalid_json_payload(self, mock_envelope_cls: MagicMock) -> None:
        """Invalid JSON payload sends error response instead of crashing."""
        conn = _make_connection(twitter=_mock_twitter())
        envelope = self._make_envelope(payload="not-valid-json{{{")

        mock_dialogue = MagicMock()
        mock_response_msg = MagicMock()
        mock_dialogue.reply.return_value = mock_response_msg
        conn.dialogues = MagicMock()
        conn.dialogues.update.return_value = mock_dialogue
        conn.put_envelope = MagicMock()  # type: ignore[method-assign]

        conn.on_send(envelope)
        conn.put_envelope.assert_called_once()
        # Verify the response indicates an error
        mock_dialogue.reply.assert_called_once()  # pylint: disable=no-member
        payload_str = mock_dialogue.reply.call_args[1][
            "payload"
        ]  # pylint: disable=no-member
        assert "error" in payload_str.lower()


# ---------------------------------------------------------------------------
# post (thread posting with rollback)
# ---------------------------------------------------------------------------


class TestPost:
    """Tests for the post method."""

    def test_single_tweet_success(self) -> None:
        """Posting a single tweet returns its id."""
        tw = _mock_twitter()
        tw.post_tweet.return_value = "100"
        conn = _make_connection(twitter=tw)
        result = conn.post(tweets=[{"text": "hello"}])
        assert result == ["100"]

    def test_thread_success(self) -> None:
        """Posting a thread chains reply_to ids."""
        tw = _mock_twitter()
        tw.post_tweet.side_effect = ["100", "200", "300"]
        conn = _make_connection(twitter=tw)

        tweets = [{"text": "t1"}, {"text": "t2"}, {"text": "t3"}]
        result = conn.post(tweets=tweets)
        assert result == ["100", "200", "300"]

        # Second tweet should have reply_to set
        calls = tw.post_tweet.call_args_list  # pylint: disable=no-member
        assert calls[1][1].get("in_reply_to_tweet_id") == "100"
        assert calls[2][1].get("in_reply_to_tweet_id") == "200"

    def test_thread_rollback_on_failure(self) -> None:
        """When a tweet in the thread fails (returns None), earlier tweets are deleted."""
        tw = _mock_twitter()
        tw.post_tweet.side_effect = ["100", None]
        tw.delete_tweet.return_value = True
        conn = _make_connection(twitter=tw)

        tweets = [{"text": "t1"}, {"text": "t2"}]
        result = conn.post(tweets=tweets)
        assert result == [None, None]
        # tweet "100" should have been deleted
        tw.delete_tweet.assert_called_once_with("100")

    def test_first_tweet_fails(self) -> None:
        """When the first tweet fails, result is [None] and no deletion of None."""
        tw = _mock_twitter()
        tw.post_tweet.return_value = None
        conn = _make_connection(twitter=tw)

        result = conn.post(tweets=[{"text": "t1"}])
        assert result == [None]
        # delete should not be called for None tweet_ids
        tw.delete_tweet.assert_not_called()

    def test_post_with_image_paths_and_quote(self) -> None:
        """Image_paths and quote_tweet_id are forwarded."""
        tw = _mock_twitter()
        tw.post_tweet.return_value = "100"
        conn = _make_connection(twitter=tw)

        result = conn.post(
            tweets=[{"text": "t1", "image_paths": ["a.png"], "quote_tweet_id": "50"}]
        )
        assert result == ["100"]
        tw.post_tweet.assert_called_once_with(  # pylint: disable=no-member
            text="t1",
            image_paths=["a.png"],
            in_reply_to_tweet_id=None,
            quote_tweet_id="50",
        )


# ---------------------------------------------------------------------------
# delete_tweet (retry logic)
# ---------------------------------------------------------------------------


class TestDeleteTweet:
    """Tests for delete_tweet with retry logic."""

    @patch("packages.dvilela.connections.tweepy.connection.time.sleep")
    def test_delete_success_first_try(self, mock_sleep: MagicMock) -> None:
        """Successful deletion on first attempt does not retry."""
        tw = _mock_twitter()
        tw.delete_tweet.return_value = True
        conn = _make_connection(twitter=tw)
        conn.delete_tweet("123")
        tw.delete_tweet.assert_called_once_with("123")
        mock_sleep.assert_not_called()

    @patch("packages.dvilela.connections.tweepy.connection.time.sleep")
    def test_delete_retries_then_succeeds(self, mock_sleep: MagicMock) -> None:
        """Deletion retries on False, then succeeds."""
        tw = _mock_twitter()
        tw.delete_tweet.side_effect = [False, False, True]
        conn = _make_connection(twitter=tw)
        conn.delete_tweet("123")
        assert tw.delete_tweet.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("packages.dvilela.connections.tweepy.connection.time.sleep")
    def test_delete_max_retries_exhausted(self, mock_sleep: MagicMock) -> None:
        """When all retries fail, method exits without error."""
        tw = _mock_twitter()
        tw.delete_tweet.return_value = False
        conn = _make_connection(twitter=tw)
        conn.delete_tweet("123")
        assert tw.delete_tweet.call_count == MAX_POST_RETRIES
        assert mock_sleep.call_count == MAX_POST_RETRIES


# ---------------------------------------------------------------------------
# Simple delegation methods
# ---------------------------------------------------------------------------


class TestSimpleDelegation:
    """Tests for like_tweet, unlike_tweet, retweet, unretweet, follow_*, unfollow_*, get_me."""

    def test_like_tweet(self) -> None:
        """Test like_tweet delegates and returns success."""
        tw = _mock_twitter()
        tw.like_tweet.return_value = True
        conn = _make_connection(twitter=tw)
        assert conn.like_tweet("1") == {"success": True}

    def test_unlike_tweet(self) -> None:
        """Test unlike_tweet delegates and returns success."""
        tw = _mock_twitter()
        tw.unlike_tweet.return_value = True
        conn = _make_connection(twitter=tw)
        assert conn.unlike_tweet("1") == {"success": True}

    def test_retweet(self) -> None:
        """Test retweet delegates and returns success."""
        tw = _mock_twitter()
        tw.retweet.return_value = True
        conn = _make_connection(twitter=tw)
        assert conn.retweet("1") == {"success": True}

    def test_unretweet(self) -> None:
        """Test unretweet delegates and returns success."""
        tw = _mock_twitter()
        tw.unretweet.return_value = True
        conn = _make_connection(twitter=tw)
        assert conn.unretweet("1") == {"success": True}

    def test_follow_by_id(self) -> None:
        """Test follow_by_id delegates and returns success."""
        tw = _mock_twitter()
        tw.follow_by_id.return_value = True
        conn = _make_connection(twitter=tw)
        assert conn.follow_by_id("42") == {"success": True}

    def test_follow_by_username(self) -> None:
        """Test follow_by_username delegates and returns success."""
        tw = _mock_twitter()
        tw.follow_by_username.return_value = True
        conn = _make_connection(twitter=tw)
        assert conn.follow_by_username("alice") == {"success": True}

    def test_unfollow_by_id(self) -> None:
        """Test unfollow_by_id delegates and returns success."""
        tw = _mock_twitter()
        tw.unfollow_by_id.return_value = True
        conn = _make_connection(twitter=tw)
        assert conn.unfollow_by_id("42") == {"success": True}

    def test_get_me_success(self) -> None:
        """Test get_me returns user info on success."""
        tw = _mock_twitter()
        tw.get_me.return_value = {
            "user_id": "1",
            "username": "me",
            "display_name": "Me",
        }
        conn = _make_connection(twitter=tw)
        assert conn.get_me() == {"user_id": "1", "username": "me", "display_name": "Me"}

    def test_get_me_returns_none(self) -> None:
        """When twitter.get_me returns None, an error dict is returned."""
        tw = _mock_twitter()
        tw.get_me.return_value = None
        conn = _make_connection(twitter=tw)
        result = conn.get_me()
        assert result is not None and "error" in result


# ---------------------------------------------------------------------------
# get_user_tweets_with_public_metrics
# ---------------------------------------------------------------------------


class TestGetUserTweetsWithPublicMetrics:
    """Tests for get_user_tweets_with_public_metrics."""

    @staticmethod
    def _make_tweet(
        tweet_id: str = "1",
        text: str = "hello",
        author_id: str = "10",
        created_at: Optional[datetime] = None,
        public_metrics: Optional[Dict] = None,
    ) -> MagicMock:
        """Create a mock Tweet object."""
        tweet = MagicMock()
        tweet.id = tweet_id
        tweet.text = text
        tweet.author_id = author_id
        tweet.created_at = created_at or datetime(2024, 1, 1, tzinfo=timezone.utc)
        tweet.public_metrics = (
            public_metrics
            if public_metrics is not None
            else {
                "like_count": 5,
                "retweet_count": 2,
                "reply_count": 1,
                "quote_count": 0,
                "impression_count": 100,
            }
        )
        return tweet

    def test_with_tweets_no_since_timestamp(self) -> None:
        """Tweets are returned with mapped fields when no since_timestamp."""
        tw = _mock_twitter()
        mock_tweet = self._make_tweet()
        tw.get_all_user_tweets.return_value = [mock_tweet]
        conn = _make_connection(twitter=tw)

        result = conn.get_user_tweets_with_public_metrics(user_id="10")
        assert len(result) == 1
        assert result[0]["id"] == "1"
        assert result[0]["text"] == "hello"
        assert result[0]["like_count"] == 5
        assert result[0]["impression_count"] == 100

        # start_time should be None
        tw.get_all_user_tweets.assert_called_once_with(
            user_id="10",
            tweet_fields=["public_metrics", "created_at"],
            start_time=None,
        )

    def test_with_since_timestamp(self) -> None:
        """Since_timestamp is converted to a UTC datetime."""
        tw = _mock_twitter()
        tw.get_all_user_tweets.return_value = []
        conn = _make_connection(twitter=tw)

        ts = 1704067200  # 2024-01-01T00:00:00 UTC
        result = conn.get_user_tweets_with_public_metrics(
            user_id="10", since_timestamp=ts
        )
        assert result == []

        expected_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        tw.get_all_user_tweets.assert_called_once_with(
            user_id="10",
            tweet_fields=["public_metrics", "created_at"],
            start_time=expected_dt,
        )

    def test_twitter_is_none(self) -> None:
        """When self.twitter is None, an empty list is returned."""
        conn = _make_connection(twitter=None)
        result = conn.get_user_tweets_with_public_metrics(user_id="10")
        assert result == []

    def test_missing_public_metric_keys(self) -> None:
        """Missing metric keys default to 0."""
        tw = _mock_twitter()
        mock_tweet = self._make_tweet(public_metrics={})
        tw.get_all_user_tweets.return_value = [mock_tweet]
        conn = _make_connection(twitter=tw)

        result = conn.get_user_tweets_with_public_metrics(user_id="10")
        assert result[0]["like_count"] == 0
        assert result[0]["retweet_count"] == 0
        assert result[0]["reply_count"] == 0
        assert result[0]["quote_count"] == 0
        assert result[0]["impression_count"] == 0

    def test_multiple_tweets(self) -> None:
        """Multiple tweets are all mapped correctly."""
        tw = _mock_twitter()
        tweets = [self._make_tweet(tweet_id=str(i), text=f"t{i}") for i in range(3)]
        tw.get_all_user_tweets.return_value = tweets
        conn = _make_connection(twitter=tw)

        result = conn.get_user_tweets_with_public_metrics(user_id="10")
        assert len(result) == 3
        assert [r["id"] for r in result] == ["0", "1", "2"]


# ---------------------------------------------------------------------------
# _get_response — all available methods dispatch
# ---------------------------------------------------------------------------


class TestGetResponseAllMethods:
    """Ensure _get_response dispatches to every available method."""

    @pytest.mark.parametrize(
        "method_name,kwargs",
        [
            ("post", {"tweets": [{"text": "hi"}]}),
            ("like_tweet", {"tweet_id": "1"}),
            ("unlike_tweet", {"tweet_id": "1"}),
            ("retweet", {"tweet_id": "1"}),
            ("unretweet", {"tweet_id": "1"}),
            ("follow_by_id", {"user_id": "1"}),
            ("follow_by_username", {"username": "alice"}),
            ("unfollow_by_id", {"user_id": "1"}),
            ("get_me", {}),
            ("get_user_tweets_with_public_metrics", {"user_id": "1"}),
        ],
    )
    def test_dispatch(self, method_name: str, kwargs: Dict) -> None:
        """Each available method is dispatched without error."""
        tw = _mock_twitter()
        # Set up reasonable return values
        tw.post_tweet.return_value = "100"
        tw.delete_tweet.return_value = True
        tw.like_tweet.return_value = True
        tw.unlike_tweet.return_value = True
        tw.retweet.return_value = True
        tw.unretweet.return_value = True
        tw.follow_by_id.return_value = True
        tw.follow_by_username.return_value = True
        tw.unfollow_by_id.return_value = True
        tw.get_me.return_value = {
            "user_id": "1",
            "username": "me",
            "display_name": "Me",
        }
        tw.get_all_user_tweets.return_value = []

        conn = _make_connection(twitter=tw)
        result = conn._get_response({"method": method_name, "kwargs": kwargs})
        assert "error" not in result or not isinstance(result.get("error"), str)
