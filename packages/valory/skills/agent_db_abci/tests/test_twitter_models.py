# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023-2026 Valory AG
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

"""Tests for twitter_models.py."""

import json
from datetime import datetime, timezone

import pytest

from packages.valory.skills.agent_db_abci.twitter_models import (
    TwitterAction,
    TwitterFollow,
    TwitterLike,
    TwitterPost,
    TwitterRewtweet,
)

TIMESTAMP = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
TIMESTAMP_ISO_Z = "2025-01-15T12:00:00Z"
TIMESTAMP_ISO_OFFSET = "2025-01-15T12:00:00+00:00"


class TestTwitterAction:
    """Tests for TwitterAction."""

    def test_construction(self) -> None:
        """Test basic construction."""
        action = TwitterAction(action="test", timestamp=TIMESTAMP)
        assert action.action == "test"
        assert action.timestamp == TIMESTAMP

    def test_to_json_structure(self) -> None:
        """Test to_json output structure."""
        action = TwitterAction(action="test", timestamp=TIMESTAMP)
        result = action.to_json()
        assert "action" in result
        assert "timestamp" in result
        assert "details" in result
        assert result["action"] == "test"

    def test_to_json_timestamp_z_suffix(self) -> None:
        """Test that to_json replaces +00:00 with Z in timestamp."""
        action = TwitterAction(action="test", timestamp=TIMESTAMP)
        result = action.to_json()
        assert result["timestamp"].endswith("Z")
        assert "+00:00" not in result["timestamp"]


class TestTwitterPost:
    """Tests for TwitterPost."""

    def test_construction(self) -> None:
        """Test basic construction."""
        post = TwitterPost(tweet_id="123", text="Hello world", timestamp=TIMESTAMP)
        assert post.action == "post"
        assert post.tweet_id == "123"
        assert post.text == "Hello world"
        assert post.reply_to_tweet_id is None
        assert post.quote_url is None

    def test_construction_with_optional_fields(self) -> None:
        """Test construction with optional fields."""
        post = TwitterPost(
            tweet_id="123",
            text="Reply",
            timestamp=TIMESTAMP,
            reply_to_tweet_id="456",
            quote_url="https://twitter.com/user/status/789",
        )
        assert post.reply_to_tweet_id == "456"
        assert post.quote_url == "https://twitter.com/user/status/789"

    def test_to_json(self) -> None:
        """Test to_json serialization."""
        post = TwitterPost(tweet_id="123", text="Hello", timestamp=TIMESTAMP)
        result = post.to_json()
        assert result["action"] == "post"
        assert result["details"]["tweet_id"] == "123"
        assert result["details"]["text"] == "Hello"

    def test_from_nested_json_with_dict_details(self) -> None:
        """Test from_nested_json when details is a dict."""
        data = {
            "action": "post",
            "timestamp": TIMESTAMP_ISO_Z,
            "details": {
                "tweet_id": "123",
                "text": "Hello",
                "reply_to_tweet_id": None,
                "quote_url": None,
            },
        }
        post = TwitterPost.from_nested_json(data)
        assert post.tweet_id == "123"
        assert post.text == "Hello"
        assert post.action == "post"
        assert post.timestamp == TIMESTAMP

    def test_from_nested_json_with_string_details(self) -> None:
        """Test from_nested_json when details is a JSON string."""
        data = {
            "action": "post",
            "timestamp": TIMESTAMP_ISO_Z,
            "details": json.dumps(
                {
                    "tweet_id": "123",
                    "text": "Hello",
                }
            ),
        }
        post = TwitterPost.from_nested_json(data)
        assert post.tweet_id == "123"
        assert post.text == "Hello"

    def test_from_nested_json_with_optional_fields(self) -> None:
        """Test from_nested_json with reply_to_tweet_id and quote_url."""
        data = {
            "action": "post",
            "timestamp": TIMESTAMP_ISO_Z,
            "details": {
                "tweet_id": "123",
                "text": "Reply text",
                "reply_to_tweet_id": "456",
                "quote_url": "https://example.com",
            },
        }
        post = TwitterPost.from_nested_json(data)
        assert post.reply_to_tweet_id == "456"
        assert post.quote_url == "https://example.com"

    def test_roundtrip(self) -> None:
        """Test to_json -> from_nested_json roundtrip."""
        original = TwitterPost(
            tweet_id="123",
            text="Hello",
            timestamp=TIMESTAMP,
            reply_to_tweet_id="456",
        )
        json_data = original.to_json()
        restored = TwitterPost.from_nested_json(json_data)
        assert restored.tweet_id == original.tweet_id
        assert restored.text == original.text
        assert restored.reply_to_tweet_id == original.reply_to_tweet_id
        assert restored.timestamp == original.timestamp


class TestTwitterRewtweet:
    """Tests for TwitterRewtweet."""

    def test_construction(self) -> None:
        """Test basic construction."""
        retweet = TwitterRewtweet(tweet_id="123", timestamp=TIMESTAMP)
        assert retweet.action == "retweet"
        assert retweet.tweet_id == "123"

    def test_to_json(self) -> None:
        """Test to_json serialization."""
        retweet = TwitterRewtweet(tweet_id="123", timestamp=TIMESTAMP)
        result = retweet.to_json()
        assert result["action"] == "retweet"
        assert result["details"]["tweet_id"] == "123"

    def test_from_nested_json_with_dict_details(self) -> None:
        """Test from_nested_json when details is a dict."""
        data = {
            "action": "retweet",
            "timestamp": TIMESTAMP_ISO_Z,
            "details": {"tweet_id": "123"},
        }
        retweet = TwitterRewtweet.from_nested_json(data)
        assert retweet.tweet_id == "123"
        assert retweet.action == "retweet"

    def test_from_nested_json_with_string_details(self) -> None:
        """Test from_nested_json when details is a JSON string."""
        data = {
            "action": "retweet",
            "timestamp": TIMESTAMP_ISO_Z,
            "details": json.dumps({"tweet_id": "123"}),
        }
        retweet = TwitterRewtweet.from_nested_json(data)
        assert retweet.tweet_id == "123"

    def test_roundtrip(self) -> None:
        """Test to_json -> from_nested_json roundtrip."""
        original = TwitterRewtweet(tweet_id="999", timestamp=TIMESTAMP)
        restored = TwitterRewtweet.from_nested_json(original.to_json())
        assert restored.tweet_id == original.tweet_id
        assert restored.timestamp == original.timestamp


class TestTwitterFollow:
    """Tests for TwitterFollow."""

    def test_construction(self) -> None:
        """Test basic construction."""
        follow = TwitterFollow(username="alice", timestamp=TIMESTAMP)
        assert follow.action == "follow"
        assert follow.username == "alice"

    def test_to_json(self) -> None:
        """Test to_json serialization."""
        follow = TwitterFollow(username="alice", timestamp=TIMESTAMP)
        result = follow.to_json()
        assert result["action"] == "follow"
        assert result["details"]["username"] == "alice"

    def test_from_nested_json_with_dict_details(self) -> None:
        """Test from_nested_json when details is a dict."""
        data = {
            "action": "follow",
            "timestamp": TIMESTAMP_ISO_Z,
            "details": {"username": "alice"},
        }
        follow = TwitterFollow.from_nested_json(data)
        assert follow.username == "alice"
        assert follow.action == "follow"

    def test_from_nested_json_with_string_details(self) -> None:
        """Test from_nested_json when details is a JSON string."""
        data = {
            "action": "follow",
            "timestamp": TIMESTAMP_ISO_Z,
            "details": json.dumps({"username": "alice"}),
        }
        follow = TwitterFollow.from_nested_json(data)
        assert follow.username == "alice"

    def test_roundtrip(self) -> None:
        """Test to_json -> from_nested_json roundtrip."""
        original = TwitterFollow(username="bob", timestamp=TIMESTAMP)
        restored = TwitterFollow.from_nested_json(original.to_json())
        assert restored.username == original.username
        assert restored.timestamp == original.timestamp


class TestTwitterLike:
    """Tests for TwitterLike."""

    def test_construction(self) -> None:
        """Test basic construction."""
        like = TwitterLike(tweet_id="123", timestamp=TIMESTAMP)
        assert like.action == "like"
        assert like.tweet_id == "123"

    def test_to_json(self) -> None:
        """Test to_json serialization."""
        like = TwitterLike(tweet_id="123", timestamp=TIMESTAMP)
        result = like.to_json()
        assert result["action"] == "like"
        assert result["details"]["tweet_id"] == "123"

    def test_from_nested_json_with_dict_details(self) -> None:
        """Test from_nested_json when details is a dict."""
        data = {
            "action": "like",
            "timestamp": TIMESTAMP_ISO_Z,
            "details": {"tweet_id": "123"},
        }
        like = TwitterLike.from_nested_json(data)
        assert like.tweet_id == "123"
        assert like.action == "like"

    def test_from_nested_json_with_string_details(self) -> None:
        """Test from_nested_json when details is a JSON string."""
        data = {
            "action": "like",
            "timestamp": TIMESTAMP_ISO_Z,
            "details": json.dumps({"tweet_id": "123"}),
        }
        like = TwitterLike.from_nested_json(data)
        assert like.tweet_id == "123"

    def test_roundtrip(self) -> None:
        """Test to_json -> from_nested_json roundtrip."""
        original = TwitterLike(tweet_id="777", timestamp=TIMESTAMP)
        restored = TwitterLike.from_nested_json(original.to_json())
        assert restored.tweet_id == original.tweet_id
        assert restored.timestamp == original.timestamp


class TestTimestampHandling:
    """Tests for timestamp parsing across all models."""

    @pytest.mark.parametrize(
        "timestamp_str",
        [
            "2025-01-15T12:00:00Z",
            "2025-01-15T12:00:00+00:00",
        ],
    )
    def test_post_timestamp_formats(self, timestamp_str: str) -> None:
        """Test that both Z and +00:00 timestamp formats are parsed."""
        data = {
            "action": "post",
            "timestamp": timestamp_str,
            "details": {"tweet_id": "1", "text": "t"},
        }
        post = TwitterPost.from_nested_json(data)
        assert post.timestamp == TIMESTAMP

    @pytest.mark.parametrize(
        "timestamp_str",
        [
            "2025-01-15T12:00:00Z",
            "2025-01-15T12:00:00+00:00",
        ],
    )
    def test_like_timestamp_formats(self, timestamp_str: str) -> None:
        """Test that both Z and +00:00 timestamp formats are parsed for likes."""
        data = {
            "action": "like",
            "timestamp": timestamp_str,
            "details": {"tweet_id": "1"},
        }
        like = TwitterLike.from_nested_json(data)
        assert like.timestamp == TIMESTAMP

    @pytest.mark.parametrize(
        "timestamp_str",
        [
            "2025-01-15T12:00:00Z",
            "2025-01-15T12:00:00+00:00",
        ],
    )
    def test_retweet_timestamp_formats(self, timestamp_str: str) -> None:
        """Test that both Z and +00:00 timestamp formats are parsed for retweets."""
        data = {
            "action": "retweet",
            "timestamp": timestamp_str,
            "details": {"tweet_id": "1"},
        }
        retweet = TwitterRewtweet.from_nested_json(data)
        assert retweet.timestamp == TIMESTAMP

    @pytest.mark.parametrize(
        "timestamp_str",
        [
            "2025-01-15T12:00:00Z",
            "2025-01-15T12:00:00+00:00",
        ],
    )
    def test_follow_timestamp_formats(self, timestamp_str: str) -> None:
        """Test that both Z and +00:00 timestamp formats are parsed for follows."""
        data = {
            "action": "follow",
            "timestamp": timestamp_str,
            "details": {"username": "alice"},
        }
        follow = TwitterFollow.from_nested_json(data)
        assert follow.timestamp == TIMESTAMP
