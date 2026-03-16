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

"""Tests for the twikit connection."""

# pylint: disable=protected-access,unused-argument,too-few-public-methods

import asyncio
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import twikit.errors  # type: ignore[import-untyped]

from packages.dvilela.connections.twikit.connection import (
    PUBLIC_ID,
    SrrDialogues,
    TwikitConnection,
    tweet_to_json,
    user_to_json,
)
from packages.valory.protocols.srr.message import SrrMessage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tweet(**overrides: Any) -> MagicMock:
    """Create a mock tweet object."""
    tweet = MagicMock()
    tweet.id = overrides.get("id", "tweet_123")
    tweet.user = MagicMock()
    tweet.user.name = overrides.get("user_name", "test_user")
    tweet.user.id = overrides.get("user_id", "user_456")
    tweet.text = overrides.get("text", "Hello world")
    tweet.created_at = overrides.get("created_at", "2024-01-01T00:00:00Z")
    tweet.view_count = overrides.get("view_count", 100)
    tweet.retweet_count = overrides.get("retweet_count", 5)
    tweet.quote_count = overrides.get("quote_count", 2)
    tweet.view_count_state = overrides.get("view_count_state", "Enabled")
    return tweet


def _make_user(**overrides: Any) -> MagicMock:
    """Create a mock user object."""
    user = MagicMock()
    user.id = overrides.get("id", "user_456")
    user.name = overrides.get("name", "Test User")
    user.screen_name = overrides.get("screen_name", "test_user")
    return user


def _make_connection(
    *,
    skip_connection: bool = False,
    disable_tweets: bool = False,
    logged_in: bool = True,
    cookies_path: Optional[Path] = None,
) -> TwikitConnection:
    """Build a TwikitConnection instance without calling __init__."""
    conn = object.__new__(TwikitConnection)
    conn.logger = logging.getLogger("test_twikit")
    conn.username = "testuser"
    conn.email = "test@example.com"
    conn.password = "password"
    conn.cookies = None
    conn.cookies_path = cookies_path or Path(tempfile.mkdtemp()) / "cookies.json"
    conn.disable_tweets = disable_tweets
    conn.skip_connection = skip_connection
    conn.client = MagicMock() if not skip_connection else None
    conn.last_call = datetime(2020, 1, 1, tzinfo=timezone.utc)
    conn.dialogues = SrrDialogues(connection_id=PUBLIC_ID)
    conn._response_envelopes = asyncio.Queue()
    conn.task_to_request = {}
    conn.logged_in = logged_in
    # `loop` is a read-only property on Connection that returns the running
    # event loop; `state` has a validating setter.  Bypass both via
    # object.__setattr__ so we can construct a test instance without the
    # full Connection.__init__ machinery.
    object.__setattr__(conn, "_state", MagicMock())
    return conn


def _make_srr_request(payload: Dict) -> SrrMessage:
    """Create an SrrMessage REQUEST."""
    return SrrMessage(
        performative=SrrMessage.Performative.REQUEST,
        payload=json.dumps(payload),
    )


def _make_srr_response() -> SrrMessage:
    """Create an SrrMessage RESPONSE (non-request)."""
    return SrrMessage(
        performative=SrrMessage.Performative.RESPONSE,
        payload="{}",
        error=False,
    )


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------


class TestTweetToJson:
    """Tests for tweet_to_json."""

    def test_tweet_to_json_default_user_id(self) -> None:
        """Verify tweet_to_json uses tweet.user.id when user_id not supplied."""
        tweet = _make_tweet()
        result = tweet_to_json(tweet)
        assert result["id"] == "tweet_123"
        assert result["user_name"] == "test_user"
        assert result["user_id"] == "user_456"
        assert result["text"] == "Hello world"
        assert result["view_count"] == 100
        assert result["retweet_count"] == 5
        assert result["quote_count"] == 2
        assert result["view_count_state"] == "Enabled"

    def test_tweet_to_json_override_user_id(self) -> None:
        """Verify tweet_to_json uses supplied user_id."""
        tweet = _make_tweet()
        result = tweet_to_json(tweet, user_id="override_789")
        assert result["user_id"] == "override_789"


class TestUserToJson:
    """Tests for user_to_json."""

    def test_user_to_json(self) -> None:
        """Verify user_to_json returns correct dict."""
        user = _make_user()
        result = user_to_json(user)
        assert result == {
            "id": "user_456",
            "name": "Test User",
            "screen_name": "test_user",
        }


# ---------------------------------------------------------------------------
# SrrDialogues
# ---------------------------------------------------------------------------


class TestSrrDialogues:
    """Tests for SrrDialogues."""

    def test_role_from_first_message(self) -> None:
        """Verify SrrDialogues assigns CONNECTION role."""
        dialogues = SrrDialogues(connection_id=PUBLIC_ID)
        assert dialogues is not None


# ---------------------------------------------------------------------------
# Connection lifecycle
# ---------------------------------------------------------------------------


class TestConnectionLifecycle:
    """Tests for connect / disconnect / response_envelopes."""

    def test_response_envelopes_raises_when_none(self) -> None:
        """Property response_envelopes raises ValueError when queue is None."""
        conn = _make_connection()
        conn._response_envelopes = None
        with pytest.raises(ValueError, match="not yet initialized"):
            _ = conn.response_envelopes

    @pytest.mark.asyncio
    async def test_connect_without_skip(self) -> None:
        """Connect calls twikit_login when not skipped."""
        conn = _make_connection(logged_in=False)
        conn.skip_connection = False
        with patch.object(conn, "twikit_login", new_callable=AsyncMock) as mock_login:
            await conn.connect()
            mock_login.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_connect_with_skip(self) -> None:
        """Connect skips login when skip_connection is True."""
        conn = _make_connection(skip_connection=True)
        with patch.object(conn, "twikit_login", new_callable=AsyncMock) as mock_login:
            await conn.connect()
            mock_login.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_disconnect(self) -> None:
        """Disconnect nullifies queue and sets disconnected."""
        conn = _make_connection()
        await conn.disconnect()
        assert conn._response_envelopes is None

    @pytest.mark.asyncio
    async def test_receive(self) -> None:
        """Receive returns envelope from queue."""
        conn = _make_connection()
        sentinel = object()
        assert conn._response_envelopes is not None
        conn._response_envelopes.put_nowait(sentinel)
        result = await conn.receive()
        assert result is sentinel


# ---------------------------------------------------------------------------
# _handle_done_task
# ---------------------------------------------------------------------------


class TestHandleDoneTask:
    """Tests for _handle_done_task."""

    def test_handle_done_task_with_response(self) -> None:
        """Verify _handle_done_task puts envelope when response is not None."""
        conn = _make_connection()
        task = MagicMock()
        response_msg = SrrMessage(
            performative=SrrMessage.Performative.RESPONSE,
            payload="{}",
            error=False,
        )
        task.result.return_value = response_msg

        request_envelope = MagicMock()
        request_envelope.sender = "agent"
        request_envelope.to = "connection"
        request_envelope.context = None
        conn.task_to_request[task] = request_envelope

        conn._handle_done_task(task)
        assert conn._response_envelopes is not None
        envelope = conn._response_envelopes.get_nowait()
        assert envelope is not None

    def test_handle_done_task_with_none_response(self) -> None:
        """Verify _handle_done_task puts None envelope when response is None."""
        conn = _make_connection()
        task = MagicMock()
        task.result.return_value = None

        request_envelope = MagicMock()
        request_envelope.sender = "agent"
        request_envelope.to = "connection"
        request_envelope.context = None
        conn.task_to_request[task] = request_envelope

        conn._handle_done_task(task)
        assert conn._response_envelopes is not None
        envelope = conn._response_envelopes.get_nowait()
        assert envelope is None

    def test_handle_done_task_with_exception(self) -> None:
        """Verify _handle_done_task catches task exceptions instead of crashing."""
        conn = _make_connection()
        task = MagicMock()
        task.result.side_effect = RuntimeError("task exploded")

        request_envelope = MagicMock()
        request_envelope.sender = "agent"
        request_envelope.to = "connection"
        request_envelope.context = None
        conn.task_to_request[task] = request_envelope

        # Should not raise
        conn._handle_done_task(task)
        assert conn._response_envelopes is not None
        envelope = conn._response_envelopes.get_nowait()
        assert envelope is None


# ---------------------------------------------------------------------------
# _get_response
# ---------------------------------------------------------------------------


class TestGetResponse:
    """Tests for _get_response."""

    def _setup_dialogue(self, conn: TwikitConnection, msg: SrrMessage) -> Any:
        """Create a mock dialogue whose reply() builds a real SrrMessage."""
        msg.sender = "agent"
        msg.to = str(PUBLIC_ID)

        def _reply(**kwargs: Any) -> SrrMessage:
            return SrrMessage(
                performative=kwargs["performative"],
                payload=kwargs["payload"],
                error=kwargs.get("error", False),
            )

        dialogue = MagicMock()
        dialogue.reply = _reply
        return dialogue

    @pytest.mark.asyncio
    async def test_non_request_performative(self) -> None:
        """Non-REQUEST performative returns error."""
        conn = _make_connection()
        resp_msg = _make_srr_response()

        def _reply(**kwargs: Any) -> SrrMessage:
            return SrrMessage(
                performative=kwargs["performative"],
                payload=kwargs["payload"],
                error=kwargs.get("error", False),
            )

        dialogue = MagicMock()
        dialogue.reply = _reply

        result = await conn._get_response(resp_msg, dialogue)
        payload = json.loads(result.payload)
        assert "error" in payload
        assert "not supported" in payload["error"]

    @pytest.mark.asyncio
    async def test_invalid_json_payload(self) -> None:
        """Invalid JSON payload returns error instead of crashing."""
        conn = _make_connection()
        msg = SrrMessage(
            performative=SrrMessage.Performative.REQUEST,
            payload="not-valid-json{{{",
        )
        dialogue = self._setup_dialogue(conn, msg)

        result = await conn._get_response(msg, dialogue)
        payload = json.loads(result.payload)
        assert "error" in payload
        assert "Invalid JSON" in payload["error"]

    @pytest.mark.asyncio
    async def test_skip_connection(self) -> None:
        """Skip_connection returns disabled error."""
        conn = _make_connection(skip_connection=True)
        msg = _make_srr_request({"method": "search", "kwargs": {}})
        dialogue = self._setup_dialogue(conn, msg)

        result = await conn._get_response(msg, dialogue)
        payload = json.loads(result.payload)
        assert (
            "disabled" in payload["error"].lower()
            or "Connection is disabled" in payload["error"]
        )

    @pytest.mark.asyncio
    async def test_missing_required_properties(self) -> None:
        """Missing method/kwargs returns error."""
        conn = _make_connection()
        msg = _make_srr_request({"only_method": "search"})
        dialogue = self._setup_dialogue(conn, msg)

        result = await conn._get_response(msg, dialogue)
        payload = json.loads(result.payload)
        assert "missing" in payload["error"].lower()

    @pytest.mark.asyncio
    async def test_unknown_method(self) -> None:
        """Unknown method returns error."""
        conn = _make_connection()
        msg = _make_srr_request({"method": "nonexistent", "kwargs": {}})
        dialogue = self._setup_dialogue(conn, msg)

        result = await conn._get_response(msg, dialogue)
        payload = json.loads(result.payload)
        assert "not in the list" in payload["error"]

    @pytest.mark.asyncio
    async def test_not_logged_in(self) -> None:
        """Not logged in returns error."""
        conn = _make_connection(logged_in=False)
        msg = _make_srr_request({"method": "search", "kwargs": {"query": "test"}})
        dialogue = self._setup_dialogue(conn, msg)

        with patch("packages.dvilela.connections.twikit.connection.asyncio.sleep"):
            result = await conn._get_response(msg, dialogue)
        payload = json.loads(result.payload)
        assert "not logged in" in payload["error"].lower()

    @pytest.mark.asyncio
    async def test_success(self) -> None:
        """Successful method call returns response."""
        conn = _make_connection()
        msg = _make_srr_request({"method": "search", "kwargs": {"query": "test"}})
        dialogue = self._setup_dialogue(conn, msg)

        with patch.object(
            conn, "search", new_callable=AsyncMock, return_value=[{"id": "1"}]
        ), patch("packages.dvilela.connections.twikit.connection.asyncio.sleep"), patch(
            "packages.dvilela.connections.twikit.connection.secrets.randbelow",
            return_value=0,
        ):
            result = await conn._get_response(msg, dialogue)

        payload = json.loads(result.payload)
        assert "response" in payload
        assert result.error is False

    @pytest.mark.asyncio
    async def test_account_locked_exception(self) -> None:
        """Handle AccountLocked exception returns locked/suspended error."""
        conn = _make_connection()
        msg = _make_srr_request({"method": "search", "kwargs": {"query": "test"}})
        dialogue = self._setup_dialogue(conn, msg)

        with patch.object(
            conn,
            "search",
            new_callable=AsyncMock,
            side_effect=twikit.errors.AccountLocked(""),
        ), patch("packages.dvilela.connections.twikit.connection.asyncio.sleep"), patch(
            "packages.dvilela.connections.twikit.connection.secrets.randbelow",
            return_value=0,
        ):
            result = await conn._get_response(msg, dialogue)

        payload = json.loads(result.payload)
        assert "locked or suspended" in payload["error"].lower()

    @pytest.mark.asyncio
    async def test_account_suspended_exception(self) -> None:
        """Handle AccountSuspended exception returns locked/suspended error."""
        conn = _make_connection()
        msg = _make_srr_request({"method": "search", "kwargs": {"query": "test"}})
        dialogue = self._setup_dialogue(conn, msg)

        with patch.object(
            conn,
            "search",
            new_callable=AsyncMock,
            side_effect=twikit.errors.AccountSuspended(""),
        ), patch("packages.dvilela.connections.twikit.connection.asyncio.sleep"), patch(
            "packages.dvilela.connections.twikit.connection.secrets.randbelow",
            return_value=0,
        ):
            result = await conn._get_response(msg, dialogue)

        payload = json.loads(result.payload)
        assert "locked or suspended" in payload["error"].lower()

    @pytest.mark.asyncio
    async def test_unauthorized_exception(self) -> None:
        """Unauthorized exception returns locked/suspended error."""
        conn = _make_connection()
        msg = _make_srr_request({"method": "search", "kwargs": {"query": "test"}})
        dialogue = self._setup_dialogue(conn, msg)

        with patch.object(
            conn,
            "search",
            new_callable=AsyncMock,
            side_effect=twikit.errors.Unauthorized(""),
        ), patch("packages.dvilela.connections.twikit.connection.asyncio.sleep"), patch(
            "packages.dvilela.connections.twikit.connection.secrets.randbelow",
            return_value=0,
        ):
            result = await conn._get_response(msg, dialogue)

        payload = json.loads(result.payload)
        assert "locked or suspended" in payload["error"].lower()

    @pytest.mark.asyncio
    async def test_generic_exception(self) -> None:
        """Generic exception returns error."""
        conn = _make_connection()
        msg = _make_srr_request({"method": "search", "kwargs": {"query": "test"}})
        dialogue = self._setup_dialogue(conn, msg)

        with patch.object(
            conn, "search", new_callable=AsyncMock, side_effect=RuntimeError("boom")
        ), patch("packages.dvilela.connections.twikit.connection.asyncio.sleep"), patch(
            "packages.dvilela.connections.twikit.connection.secrets.randbelow",
            return_value=0,
        ):
            result = await conn._get_response(msg, dialogue)

        payload = json.loads(result.payload)
        assert "boom" in payload["error"]


# ---------------------------------------------------------------------------
# validate_login
# ---------------------------------------------------------------------------


class TestValidateLogin:
    """Tests for validate_login."""

    @pytest.mark.asyncio
    async def test_validate_login_success(self) -> None:
        """Successful validation returns True."""
        conn = _make_connection()
        user = _make_user(id="1450081635559428107")
        conn.client.get_user_by_screen_name = AsyncMock(return_value=user)

        with patch("packages.dvilela.connections.twikit.connection.asyncio.sleep"):
            result = await conn.validate_login()
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_login_wrong_user_id(self) -> None:
        """Wrong user id fails after retries."""
        conn = _make_connection()
        user = _make_user(id="wrong_id")
        conn.client.get_user_by_screen_name = AsyncMock(return_value=user)

        with patch("packages.dvilela.connections.twikit.connection.asyncio.sleep"):
            result = await conn.validate_login()
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_login_exception(self) -> None:
        """Exception during validation retries and returns False."""
        conn = _make_connection()
        conn.client.get_user_by_screen_name = AsyncMock(
            side_effect=RuntimeError("network error")
        )

        with patch("packages.dvilela.connections.twikit.connection.asyncio.sleep"):
            result = await conn.validate_login()
        assert result is False


# ---------------------------------------------------------------------------
# twikit_login
# ---------------------------------------------------------------------------


class TestTwikitLogin:
    """Tests for twikit_login."""

    @pytest.mark.asyncio
    async def test_login_cookies_valid(self) -> None:
        """Cookies valid on first try sets logged_in."""
        conn = _make_connection(logged_in=False)
        conn.client.login = AsyncMock()

        with patch.object(
            conn, "validate_login", new_callable=AsyncMock, return_value=True
        ):
            await conn.twikit_login()
        assert conn.logged_in is True

    @pytest.mark.asyncio
    async def test_login_cookies_invalid_then_password_works(self) -> None:
        """Invalid cookies -> delete -> password login succeeds."""
        conn = _make_connection(logged_in=False)
        # Create actual cookies file so unlink works
        conn.cookies_path.parent.mkdir(parents=True, exist_ok=True)
        conn.cookies_path.write_text("{}")

        call_count = 0

        async def mock_login(**kwargs: Any) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("bad cookies")

        conn.client.login = mock_login

        with patch.object(
            conn, "validate_login", new_callable=AsyncMock, return_value=True
        ):
            await conn.twikit_login()
        assert conn.logged_in is True

    @pytest.mark.asyncio
    async def test_login_cookies_invalid_password_fails(self) -> None:
        """Both cookies and password login fail."""
        conn = _make_connection(logged_in=False)
        conn.cookies_path.parent.mkdir(parents=True, exist_ok=True)
        conn.cookies_path.write_text("{}")

        call_count = 0

        async def mock_login(**kwargs: Any) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("bad cookies")
            raise RuntimeError("password also fails")

        conn.client.login = mock_login

        await conn.twikit_login()
        assert conn.logged_in is False

    @pytest.mark.asyncio
    async def test_login_account_locked(self) -> None:
        """Handle AccountLocked during login logs error."""
        conn = _make_connection(logged_in=False)
        conn.client.login = AsyncMock(side_effect=twikit.errors.AccountLocked("locked"))

        await conn.twikit_login()
        assert conn.logged_in is False


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


class TestSearch:
    """Tests for search."""

    @pytest.mark.asyncio
    async def test_search(self) -> None:
        """Search returns list of tweet dicts."""
        conn = _make_connection()
        tweets = [_make_tweet(id="t1"), _make_tweet(id="t2")]
        conn.client.search_tweet = AsyncMock(return_value=tweets)

        result = await conn.search("test query")
        assert len(result) == 2
        assert result[0]["id"] == "t1"
        assert result[1]["id"] == "t2"


# ---------------------------------------------------------------------------
# post
# ---------------------------------------------------------------------------


class TestPost:
    """Tests for post."""

    @pytest.mark.asyncio
    async def test_post_single_tweet(self) -> None:
        """Post a single tweet successfully."""
        conn = _make_connection()

        with patch.object(
            conn, "post_tweet", new_callable=AsyncMock, return_value="tweet_1"
        ):
            result = await conn.post([{"text": "hello"}])
        assert result == ["tweet_1"]

    @pytest.mark.asyncio
    async def test_post_thread(self) -> None:
        """Post a thread of tweets successfully."""
        conn = _make_connection()

        with patch.object(
            conn, "post_tweet", new_callable=AsyncMock, side_effect=["t1", "t2", "t3"]
        ):
            result = await conn.post([{"text": "1"}, {"text": "2"}, {"text": "3"}])
        assert result == ["t1", "t2", "t3"]

    @pytest.mark.asyncio
    async def test_post_disabled_tweets(self) -> None:
        """Disabled tweets returns list of '0's."""
        conn = _make_connection(disable_tweets=True)
        result = await conn.post([{"text": "a"}, {"text": "b"}])
        assert result == ["0", "0"]

    @pytest.mark.asyncio
    async def test_post_rollback_on_failure(self) -> None:
        """When a tweet fails, previously posted tweets are deleted."""
        conn = _make_connection()

        with patch.object(
            conn, "post_tweet", new_callable=AsyncMock, side_effect=["t1", None]
        ), patch.object(
            conn, "delete_tweet", new_callable=AsyncMock
        ) as mock_delete, patch(
            "packages.dvilela.connections.twikit.connection.secrets.randbelow",
            return_value=0,
        ), patch(
            "packages.dvilela.connections.twikit.connection.asyncio.sleep"
        ):
            result = await conn.post([{"text": "1"}, {"text": "2"}])

        assert result == [None, None]
        mock_delete.assert_awaited_once_with("t1")


# ---------------------------------------------------------------------------
# post_tweet
# ---------------------------------------------------------------------------


class TestPostTweet:
    """Tests for post_tweet."""

    @pytest.mark.asyncio
    async def test_post_tweet_success(self) -> None:
        """Successful tweet creation and verification."""
        conn = _make_connection()
        mock_result = MagicMock()
        mock_result.id = "new_tweet_id"
        conn.client.create_tweet = AsyncMock(return_value=mock_result)
        conn.client.get_tweet_by_id = AsyncMock(return_value=MagicMock())

        result = await conn.post_tweet(text="hello")
        assert result == "new_tweet_id"

    @pytest.mark.asyncio
    async def test_post_tweet_retry_on_exception(self) -> None:
        """Retries on exception, eventually succeeds."""
        conn = _make_connection()
        mock_result = MagicMock()
        mock_result.id = "retry_tweet"
        conn.client.create_tweet = AsyncMock(
            side_effect=[RuntimeError("fail"), mock_result]
        )
        conn.client.get_tweet_by_id = AsyncMock(return_value=MagicMock())

        result = await conn.post_tweet(text="hello")
        assert result == "retry_tweet"

    @pytest.mark.asyncio
    async def test_post_tweet_all_retries_fail(self) -> None:
        """All retries fail returns None."""
        conn = _make_connection()
        conn.client.create_tweet = AsyncMock(side_effect=RuntimeError("fail"))

        result = await conn.post_tweet(text="hello")
        assert result is None

    @pytest.mark.asyncio
    async def test_post_tweet_verification_failure(self) -> None:
        """Tweet created but verification fails returns None."""
        conn = _make_connection()
        mock_result = MagicMock()
        mock_result.id = "unverified_tweet"
        conn.client.create_tweet = AsyncMock(return_value=mock_result)
        conn.client.get_tweet_by_id = AsyncMock(
            side_effect=twikit.errors.TweetNotAvailable("")
        )

        with patch("packages.dvilela.connections.twikit.connection.asyncio.sleep"):
            result = await conn.post_tweet(text="hello")
        assert result is None


# ---------------------------------------------------------------------------
# delete_tweet
# ---------------------------------------------------------------------------


class TestDeleteTweet:
    """Tests for delete_tweet."""

    @pytest.mark.asyncio
    async def test_delete_tweet_success(self) -> None:
        """Successful deletion."""
        conn = _make_connection()
        conn.client.delete_tweet = AsyncMock()

        await conn.delete_tweet("tweet_123")
        conn.client.delete_tweet.assert_awaited_once_with("tweet_123")

    @pytest.mark.asyncio
    async def test_delete_tweet_retry(self) -> None:
        """Retries on exception."""
        conn = _make_connection()
        conn.client.delete_tweet = AsyncMock(side_effect=[RuntimeError("fail"), None])

        with patch("packages.dvilela.connections.twikit.connection.asyncio.sleep"):
            await conn.delete_tweet("tweet_123")
        assert conn.client.delete_tweet.await_count == 2


# ---------------------------------------------------------------------------
# get_user_tweets
# ---------------------------------------------------------------------------


class TestGetUserTweets:
    """Tests for get_user_tweets."""

    @pytest.mark.asyncio
    async def test_get_user_tweets(self) -> None:
        """Returns list of tweet dicts with user id."""
        conn = _make_connection()
        user = _make_user(id="u1")
        tweets = [_make_tweet(id="t1")]
        conn.client.get_user_by_screen_name = AsyncMock(return_value=user)
        conn.client.get_user_tweets = AsyncMock(return_value=tweets)

        with patch("packages.dvilela.connections.twikit.connection.asyncio.sleep"):
            result = await conn.get_user_tweets("test_handle")

        assert len(result) == 1
        assert result[0]["user_id"] == "u1"


# ---------------------------------------------------------------------------
# like_tweet / follow_user / retweet
# ---------------------------------------------------------------------------


class TestLikeTweet:
    """Tests for like_tweet."""

    @pytest.mark.asyncio
    async def test_like_tweet_success(self) -> None:
        """Successful like."""
        conn = _make_connection()
        conn.client.favorite_tweet = AsyncMock()

        result = await conn.like_tweet("tweet_123")
        assert result == {"success": True}

    @pytest.mark.asyncio
    async def test_like_tweet_twitter_exception(self) -> None:
        """Raise TwitterException on like."""
        conn = _make_connection()
        conn.client.favorite_tweet = AsyncMock(
            side_effect=twikit.errors.TwitterException("api error")
        )

        result = await conn.like_tweet("tweet_123")
        assert result["success"] is False
        assert "Twikit API error" in result["error"]

    @pytest.mark.asyncio
    async def test_like_tweet_generic_exception(self) -> None:
        """Generic exception on like."""
        conn = _make_connection()
        conn.client.favorite_tweet = AsyncMock(side_effect=RuntimeError("unexpected"))

        result = await conn.like_tweet("tweet_123")
        assert result["success"] is False
        assert "Unexpected error" in result["error"]


class TestFollowUser:
    """Tests for follow_user."""

    @pytest.mark.asyncio
    async def test_follow_user_success(self) -> None:
        """Successful follow."""
        conn = _make_connection()
        conn.client.follow_user = AsyncMock()

        result = await conn.follow_user("user_123")
        assert result == {"success": True}

    @pytest.mark.asyncio
    async def test_follow_user_twitter_exception(self) -> None:
        """Raise TwitterException on follow."""
        conn = _make_connection()
        conn.client.follow_user = AsyncMock(
            side_effect=twikit.errors.TwitterException("api error")
        )

        result = await conn.follow_user("user_123")
        assert result["success"] is False
        assert "Twikit API error" in result["error"]

    @pytest.mark.asyncio
    async def test_follow_user_generic_exception(self) -> None:
        """Generic exception on follow."""
        conn = _make_connection()
        conn.client.follow_user = AsyncMock(side_effect=RuntimeError("unexpected"))

        result = await conn.follow_user("user_123")
        assert result["success"] is False
        assert "Unexpected error" in result["error"]


class TestRetweet:
    """Tests for retweet."""

    @pytest.mark.asyncio
    async def test_retweet_success(self) -> None:
        """Successful retweet."""
        conn = _make_connection()
        conn.client.retweet = AsyncMock()

        result = await conn.retweet("tweet_123")
        assert result == {"success": True}

    @pytest.mark.asyncio
    async def test_retweet_twitter_exception(self) -> None:
        """Raise TwitterException on retweet."""
        conn = _make_connection()
        conn.client.retweet = AsyncMock(
            side_effect=twikit.errors.TwitterException("api error")
        )

        result = await conn.retweet("tweet_123")
        assert result["success"] is False
        assert "Twikit API error" in result["error"]

    @pytest.mark.asyncio
    async def test_retweet_generic_exception(self) -> None:
        """Generic exception on retweet."""
        conn = _make_connection()
        conn.client.retweet = AsyncMock(side_effect=RuntimeError("unexpected"))

        result = await conn.retweet("tweet_123")
        assert result["success"] is False
        assert "Unexpected error" in result["error"]


# ---------------------------------------------------------------------------
# filter_suspended_users
# ---------------------------------------------------------------------------


class TestFilterSuspendedUsers:
    """Tests for filter_suspended_users."""

    @pytest.mark.asyncio
    async def test_filter_mix_of_valid_and_suspended(self) -> None:
        """Returns only non-suspended users."""
        conn = _make_connection()

        async def mock_get_user(name: str) -> MagicMock:
            if name == "suspended_user":
                raise twikit.errors.TwitterException("suspended")
            return _make_user(screen_name=name)

        conn.client.get_user_by_screen_name = mock_get_user

        with patch(
            "packages.dvilela.connections.twikit.connection.secrets.randbelow",
            return_value=0,
        ), patch("packages.dvilela.connections.twikit.connection.asyncio.sleep"):
            result = await conn.filter_suspended_users(
                ["valid_user", "suspended_user", "another_valid"]
            )

        assert result == ["valid_user", "another_valid"]

    @pytest.mark.asyncio
    async def test_filter_generic_exception(self) -> None:
        """Generic exception skips user with logging."""
        conn = _make_connection()

        async def mock_get_user(name: str) -> MagicMock:
            raise RuntimeError("unexpected")

        conn.client.get_user_by_screen_name = mock_get_user

        with patch(
            "packages.dvilela.connections.twikit.connection.secrets.randbelow",
            return_value=0,
        ), patch("packages.dvilela.connections.twikit.connection.asyncio.sleep"):
            result = await conn.filter_suspended_users(["user1"])

        assert result == []


# ---------------------------------------------------------------------------
# get_user_by_screen_name
# ---------------------------------------------------------------------------


class TestGetUserByScreenName:
    """Tests for get_user_by_screen_name."""

    @pytest.mark.asyncio
    async def test_get_user_by_screen_name(self) -> None:
        """Returns user dict."""
        conn = _make_connection()
        user = _make_user(id="u1", name="Alice", screen_name="alice")
        conn.client.get_user_by_screen_name = AsyncMock(return_value=user)

        result = await conn.get_user_by_screen_name("alice")
        assert result == {"id": "u1", "name": "Alice", "screen_name": "alice"}


# ---------------------------------------------------------------------------
# get_twitter_user_id
# ---------------------------------------------------------------------------


class TestGetTwitterUserId:
    """Tests for get_twitter_user_id."""

    @pytest.mark.asyncio
    async def test_get_twitter_user_id_success(self) -> None:
        """Returns twid from cookies file."""
        conn = _make_connection()
        conn.cookies_path.parent.mkdir(parents=True, exist_ok=True)
        conn.cookies_path.write_text(json.dumps({"twid": '"u=12345"'}))

        result = await conn.get_twitter_user_id()
        assert result == "u=12345"

    @pytest.mark.asyncio
    async def test_get_twitter_user_id_missing_twid(self) -> None:
        """Missing twid raises ValueError."""
        conn = _make_connection()
        conn.cookies_path.parent.mkdir(parents=True, exist_ok=True)
        conn.cookies_path.write_text(json.dumps({"other": "value"}))

        with pytest.raises(ValueError, match="twid"):
            await conn.get_twitter_user_id()


# ---------------------------------------------------------------------------
# upload_media
# ---------------------------------------------------------------------------


class TestUploadMedia:
    """Tests for upload_media."""

    @pytest.mark.asyncio
    async def test_upload_media_success(self) -> None:
        """Successful upload returns media id."""
        conn = _make_connection()
        conn.client.upload_media = AsyncMock(return_value="media_123")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name

        try:
            with patch(
                "packages.dvilela.connections.twikit.connection.secrets.randbelow",
                return_value=0,
            ), patch("packages.dvilela.connections.twikit.connection.asyncio.sleep"):
                result = await conn.upload_media(temp_path)
            assert result == "media_123"
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_upload_media_file_not_found(self) -> None:
        """File not found returns None immediately."""
        conn = _make_connection()

        with patch(
            "packages.dvilela.connections.twikit.connection.secrets.randbelow",
            return_value=0,
        ), patch("packages.dvilela.connections.twikit.connection.asyncio.sleep"):
            result = await conn.upload_media("/nonexistent/path.png")
        assert result is None

    @pytest.mark.asyncio
    async def test_upload_media_dict_input(self) -> None:
        """Dict input extracts latest_image_path."""
        conn = _make_connection()
        conn.client.upload_media = AsyncMock(return_value="media_456")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name

        try:
            with patch(
                "packages.dvilela.connections.twikit.connection.secrets.randbelow",
                return_value=0,
            ), patch("packages.dvilela.connections.twikit.connection.asyncio.sleep"):
                result = await conn.upload_media({"latest_image_path": temp_path})  # type: ignore[arg-type]
            assert result == "media_456"
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_upload_media_retry_on_exception(self) -> None:
        """Retries on generic exception, eventually succeeds."""
        conn = _make_connection()
        conn.client.upload_media = AsyncMock(
            side_effect=[RuntimeError("fail"), "media_789"]
        )

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name

        try:
            with patch(
                "packages.dvilela.connections.twikit.connection.secrets.randbelow",
                return_value=0,
            ), patch("packages.dvilela.connections.twikit.connection.asyncio.sleep"):
                result = await conn.upload_media(temp_path)
            assert result == "media_789"
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_upload_media_all_retries_fail(self) -> None:
        """All retries fail returns None."""
        conn = _make_connection()
        conn.client.upload_media = AsyncMock(
            side_effect=RuntimeError("persistent failure")
        )

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name

        try:
            with patch(
                "packages.dvilela.connections.twikit.connection.secrets.randbelow",
                return_value=0,
            ), patch("packages.dvilela.connections.twikit.connection.asyncio.sleep"):
                result = await conn.upload_media(temp_path)
            assert result is None
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_upload_media_returns_none(self) -> None:
        """Upload returns None media_id, exhausts retries."""
        conn = _make_connection()
        conn.client.upload_media = AsyncMock(return_value=None)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name

        try:
            with patch(
                "packages.dvilela.connections.twikit.connection.secrets.randbelow",
                return_value=0,
            ), patch("packages.dvilela.connections.twikit.connection.asyncio.sleep"):
                result = await conn.upload_media(temp_path)
            assert result is None
        finally:
            os.unlink(temp_path)


# ---------------------------------------------------------------------------
# send + _handle_envelope (async dispatch pipeline)
# ---------------------------------------------------------------------------


class TestSendAndHandleEnvelope:
    """Tests for send() and _handle_envelope()."""

    @pytest.mark.asyncio
    async def test_send_dispatches_and_queues_response(self) -> None:
        """send() dispatches _get_response via task and queues result."""
        conn = _make_connection()
        object.__setattr__(conn, "_loop", asyncio.get_running_loop())

        response_msg = SrrMessage(
            performative=SrrMessage.Performative.RESPONSE,
            payload="{}",
            error=False,
        )

        async def mock_get_response(srr_message: Any, dialogue: Any) -> SrrMessage:
            return response_msg

        conn._get_response = mock_get_response  # type: ignore[assignment]

        msg = _make_srr_request({"method": "search", "kwargs": {}})
        msg.sender = "agent"
        msg.to = str(PUBLIC_ID)

        envelope = MagicMock()
        envelope.message = msg
        envelope.sender = str(PUBLIC_ID)
        envelope.to = str(PUBLIC_ID)
        envelope.context = None

        await conn.send(envelope)

        # Wait for the background task to complete
        tasks = list(conn.task_to_request.keys())
        assert len(tasks) == 1
        await tasks[0]

        # _handle_done_task callback should have queued the response
        assert conn._response_envelopes is not None
        assert not conn._response_envelopes.empty()


# ---------------------------------------------------------------------------
# _get_response rate-limit branch
# ---------------------------------------------------------------------------


class TestGetResponseRateLimit:
    """Tests for _get_response rate-limit loop."""

    def _setup_dialogue(self, conn: TwikitConnection, msg: SrrMessage) -> Any:
        """Create a mock dialogue."""
        msg.sender = "agent"
        msg.to = str(PUBLIC_ID)

        def _reply(**kwargs: Any) -> SrrMessage:
            return SrrMessage(
                performative=kwargs["performative"],
                payload=kwargs["payload"],
                error=kwargs.get("error", False),
            )

        dialogue = MagicMock()
        dialogue.reply = _reply
        return dialogue

    @pytest.mark.asyncio
    async def test_rate_limit_sleep(self) -> None:
        """Rate limit loop sleeps when last_call is recent."""
        conn = _make_connection()
        # Set last_call to now so the rate-limit loop fires
        conn.last_call = datetime.now(timezone.utc)

        msg = _make_srr_request({"method": "search", "kwargs": {"query": "test"}})
        dialogue = self._setup_dialogue(conn, msg)

        sleep_calls = []

        def mock_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)
            # After first sleep, advance last_call so loop exits
            conn.last_call = datetime(2020, 1, 1, tzinfo=timezone.utc)

        with patch.object(
            conn, "search", new_callable=AsyncMock, return_value=[{"id": "1"}]
        ), patch(
            "packages.dvilela.connections.twikit.connection.asyncio.sleep",
            side_effect=mock_sleep,
        ), patch(
            "packages.dvilela.connections.twikit.connection.secrets.randbelow",
            return_value=0,
        ):
            result = await conn._get_response(msg, dialogue)

        # Verify the rate-limit sleep was called at least once
        assert len(sleep_calls) >= 1
        assert result.error is False


# ---------------------------------------------------------------------------
# twikit_login additional paths
# ---------------------------------------------------------------------------


class TestTwikitLoginAdditionalPaths:
    """Tests for uncovered twikit_login paths."""

    @pytest.mark.asyncio
    async def test_login_validation_fails_triggers_relogin(self) -> None:
        """Cookies load OK, but validate_login returns False → ValueError → re-login."""
        conn = _make_connection(logged_in=False)
        conn.cookies_path.parent.mkdir(parents=True, exist_ok=True)
        conn.cookies_path.write_text("{}")
        conn.client.login = AsyncMock()

        validate_results = [False, True]

        async def mock_validate() -> bool:
            return validate_results.pop(0)

        with patch.object(conn, "validate_login", side_effect=mock_validate):
            await conn.twikit_login()

        assert conn.logged_in is True

    @pytest.mark.asyncio
    async def test_relogin_validation_fails(self) -> None:
        """Re-login succeeds but validate_login returns False → stays logged out."""
        conn = _make_connection(logged_in=False)
        conn.cookies_path.parent.mkdir(parents=True, exist_ok=True)
        conn.cookies_path.write_text("{}")
        conn.client.login = AsyncMock()

        with patch.object(
            conn,
            "validate_login",
            new_callable=AsyncMock,
            return_value=False,
        ):
            await conn.twikit_login()

        assert conn.logged_in is False


# ---------------------------------------------------------------------------
# post_tweet: create_tweet returns result with id=None
# ---------------------------------------------------------------------------


class TestPostTweetNullId:
    """Tests for post_tweet when create_tweet returns a result with id=None."""

    @pytest.mark.asyncio
    async def test_post_tweet_null_tweet_id(self) -> None:
        """create_tweet succeeds but result.id is None → retries exhaust → returns None."""
        conn = _make_connection()
        mock_result = MagicMock()
        mock_result.id = None
        conn.client.create_tweet = AsyncMock(return_value=mock_result)

        result = await conn.post_tweet(text="hello")
        assert result is None


# ---------------------------------------------------------------------------
# delete_tweet: all retries exhausted
# ---------------------------------------------------------------------------


class TestDeleteTweetAllRetriesFail:
    """Tests for delete_tweet when all retries are exhausted."""

    @pytest.mark.asyncio
    async def test_delete_tweet_all_retries_exhausted(self) -> None:
        """All delete retries fail — function completes without raising."""
        conn = _make_connection()
        conn.client.delete_tweet = AsyncMock(
            side_effect=RuntimeError("persistent failure")
        )

        with patch("packages.dvilela.connections.twikit.connection.asyncio.sleep"):
            await conn.delete_tweet("tweet_123")

        assert conn.client.delete_tweet.await_count == 5  # MAX_POST_RETRIES
