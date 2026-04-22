# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024-2026 Valory AG
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

"""Tests for twitter behaviour classes."""

# pylint: disable=protected-access,unused-argument

import json
from collections.abc import Generator
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

from packages.valory.skills.agent_db_abci.twitter_models import (
    TwitterFollow,
    TwitterLike,
    TwitterRewtweet,
)
from packages.valory.skills.memeooorr_abci.behaviour_classes.twitter import (
    ActionTweetBehaviour,
    BaseTweetBehaviour,
    CollectFeedbackBehaviour,
    EngageTwitterBehaviour,
    InteractionContext,
    is_tweet_valid,
)
from packages.valory.skills.memeooorr_abci.rounds import Event
from packages.valory.skills.memeooorr_abci.tests.conftest import (
    make_mock_context,
    make_mock_my_agent,
    make_mock_params,
    make_mock_synchronized_data,
    make_twitter_post,
)

# ============================================================================
# Helpers for driving generators that use yield from sub-generators
# ============================================================================


def _drive_gen_with_side_effects(
    gen: Generator[Any, None, Any],
    side_effects_map: dict[str, Any] | None = None,
) -> Any:
    """Drive a generator, providing return values via side effects on the behaviour mock.

    For generators that use `yield from self.some_method(...)`, we need the mock
    methods to return proper generators. This helper just drives the top-level
    generator, sending None for each yield, and returns the StopIteration value.

    :param gen: the generator to drive.
    :param side_effects_map: optional mapping of side effects (unused, kept for API compat).
    :return: the StopIteration value from the generator.
    """
    result = None
    try:
        next(gen)
        while True:
            gen.send(None)
    except StopIteration as e:
        result = e.value
    return result


def _make_gen_return(value: Any) -> Any:
    """Create a generator that immediately returns a value (for mocking yield from calls)."""

    def _gen(*args: Any, **kwargs: Any) -> Generator[Any, None, Any]:
        if False:  # pylint: disable=using-constant-test  # pragma: no cover
            yield  # Make this a generator
        return value

    return _gen


def _make_gen_return_none() -> Any:
    """Create a generator that yields once then returns None."""

    def _gen(*args: Any, **kwargs: Any) -> Generator[Any, None, None]:
        if False:  # pylint: disable=using-constant-test  # pragma: no cover
            yield

    return _gen


# ============================================================================
# Tests for is_tweet_valid
# ============================================================================


class TestIsTweetValid:
    """Tests for the is_tweet_valid function."""

    def test_valid_short_tweet(self) -> None:
        """Test that a short tweet is valid."""
        assert is_tweet_valid("Hello world") is True

    def test_valid_max_length_tweet(self) -> None:
        """Test that a tweet at max length is valid."""
        tweet = "a" * 280
        assert is_tweet_valid(tweet) is True

    def test_invalid_too_long_tweet(self) -> None:
        """Test that a tweet over max length is invalid."""
        tweet = "a" * 281
        assert is_tweet_valid(tweet) is False

    def test_empty_tweet(self) -> None:
        """Test that an empty tweet is valid."""
        assert is_tweet_valid("") is True


# ============================================================================
# Tests for InteractionContext
# ============================================================================


class TestInteractionContext:  # pylint: disable=too-few-public-methods
    """Tests for InteractionContext dataclass."""

    def test_creation_with_none_previous_tweets(self) -> None:
        """Test creating context with None previous_tweets."""
        ctx = InteractionContext(
            pending_tweets={},
            previous_tweets=None,
            persona="persona",
            new_interacted_tweet_ids=[],
        )
        assert ctx.previous_tweets is None


# ============================================================================
# Tests for BaseTweetBehaviour
# ============================================================================


class TestBaseTweetBehaviour:  # pylint: disable=too-many-public-methods
    """Tests for BaseTweetBehaviour methods."""

    def _make_behaviour(self, **kwargs: Any) -> MagicMock:
        """Create a mock behaviour with BaseTweetBehaviour spec."""
        behaviour = MagicMock(spec=BaseTweetBehaviour)
        behaviour.context = make_mock_context(**kwargs)
        behaviour.params = make_mock_params()
        behaviour.synchronized_data = make_mock_synchronized_data()
        return behaviour

    # ----- _format_previous_tweets_str -----

    def test_format_previous_tweets_str_empty(self) -> None:
        """Test formatting with empty list."""
        result = BaseTweetBehaviour._format_previous_tweets_str(None)
        assert result == ""

    def test_format_previous_tweets_str_empty_list(self) -> None:
        """Test formatting with empty list."""
        result = BaseTweetBehaviour._format_previous_tweets_str([])
        assert result == ""

    def test_format_previous_tweets_str_twitter_post_objects(self) -> None:
        """Test formatting with TwitterPost objects."""
        posts = [
            make_twitter_post(
                text="Hello", timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc)
            ),
            make_twitter_post(
                text="World", timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc)
            ),
        ]
        result = BaseTweetBehaviour._format_previous_tweets_str(posts)
        assert "Hello" in result
        assert "World" in result

    def test_format_previous_tweets_str_dict_objects(self) -> None:
        """Test formatting with dict objects."""
        posts = [
            {"text": "Hello", "timestamp": "2024-01-01"},
            {"text": "World", "timestamp": "2024-01-02"},
        ]
        result = BaseTweetBehaviour._format_previous_tweets_str(posts)
        assert "Hello" in result
        assert "World" in result

    # ----- _write_tweet_to_kv_store -----

    def test_write_tweet_to_kv_store_single_dict(self) -> None:
        """Test writing a single tweet dict to KV store."""
        behaviour = self._make_behaviour()
        behaviour.get_tweets_from_db = _make_gen_return([])
        behaviour._write_kv = _make_gen_return(True)

        gen = BaseTweetBehaviour._write_tweet_to_kv_store(behaviour, {"text": "hello"})
        result = _drive_gen_with_side_effects(gen)
        assert result is True

    def test_write_tweet_to_kv_store_list(self) -> None:
        """Test writing a list of tweets to KV store."""
        behaviour = self._make_behaviour()
        behaviour.get_tweets_from_db = _make_gen_return([])
        behaviour._write_kv = _make_gen_return(True)

        gen = BaseTweetBehaviour._write_tweet_to_kv_store(
            behaviour, [{"text": "a"}, {"text": "b"}]
        )
        result = _drive_gen_with_side_effects(gen)
        assert result is True

    # ----- _handle_simple_twitter_action -----

    def test_handle_simple_twitter_action_success_like(self) -> None:
        """Test successful like action."""
        behaviour = self._make_behaviour()
        mock_db_result = MagicMock()
        mock_db_result.attribute_id = "attr_123"

        behaviour._call_tweepy = _make_gen_return({"success": True})
        behaviour.context.agents_fun_db.my_agent.add_interaction = _make_gen_return(
            mock_db_result
        )
        behaviour._store_agent_action = _make_gen_return(None)

        gen = BaseTweetBehaviour._handle_simple_twitter_action(
            behaviour,
            action_description="Liking tweet",
            tweepy_method_name="like_tweet",
            tweepy_kwargs={"tweet_id": "123"},
            action_class=TwitterLike,
            action_constructor_kwargs={"tweet_id": "123"},
        )
        result = _drive_gen_with_side_effects(gen)
        assert result is True

    def test_handle_simple_twitter_action_success_retweet(self) -> None:
        """Test successful retweet action."""
        behaviour = self._make_behaviour()
        mock_db_result = MagicMock()
        mock_db_result.attribute_id = "attr_456"

        behaviour._call_tweepy = _make_gen_return({"success": True})
        behaviour.context.agents_fun_db.my_agent.add_interaction = _make_gen_return(
            mock_db_result
        )
        behaviour._store_agent_action = _make_gen_return(None)

        gen = BaseTweetBehaviour._handle_simple_twitter_action(
            behaviour,
            action_description="Retweeting tweet",
            tweepy_method_name="retweet",
            tweepy_kwargs={"tweet_id": "456"},
            action_class=TwitterRewtweet,
            action_constructor_kwargs={"tweet_id": "456"},
        )
        result = _drive_gen_with_side_effects(gen)
        assert result is True

    def test_handle_simple_twitter_action_success_follow(self) -> None:
        """Test successful follow action."""
        behaviour = self._make_behaviour()
        mock_db_result = MagicMock()
        mock_db_result.attribute_id = "attr_789"

        behaviour._call_tweepy = _make_gen_return({"success": True})
        behaviour.context.agents_fun_db.my_agent.add_interaction = _make_gen_return(
            mock_db_result
        )
        behaviour._store_agent_action = _make_gen_return(None)

        gen = BaseTweetBehaviour._handle_simple_twitter_action(
            behaviour,
            action_description="Following user",
            tweepy_method_name="follow_by_username",
            tweepy_kwargs={"username": "user1"},
            action_class=TwitterFollow,
            action_constructor_kwargs={"username": "user1"},
        )
        result = _drive_gen_with_side_effects(gen)
        assert result is True

    def test_handle_simple_twitter_action_tweepy_returns_none(self) -> None:
        """Test when tweepy call returns None."""
        behaviour = self._make_behaviour()
        behaviour._call_tweepy = _make_gen_return(None)

        gen = BaseTweetBehaviour._handle_simple_twitter_action(
            behaviour,
            action_description="Liking tweet",
            tweepy_method_name="like_tweet",
            tweepy_kwargs={"tweet_id": "123"},
            action_class=TwitterLike,
            action_constructor_kwargs={"tweet_id": "123"},
        )
        result = _drive_gen_with_side_effects(gen)
        assert result is False

    def test_handle_simple_twitter_action_tweepy_returns_error(self) -> None:
        """Test when tweepy call returns success=False."""
        behaviour = self._make_behaviour()
        behaviour._call_tweepy = _make_gen_return(
            {"success": False, "error": "Rate limited"}
        )

        gen = BaseTweetBehaviour._handle_simple_twitter_action(
            behaviour,
            action_description="Liking tweet",
            tweepy_method_name="like_tweet",
            tweepy_kwargs={"tweet_id": "123"},
            action_class=TwitterLike,
            action_constructor_kwargs={"tweet_id": "123"},
        )
        result = _drive_gen_with_side_effects(gen)
        assert result is False

    def test_handle_simple_twitter_action_db_add_fails(self) -> None:
        """Test when DB add_interaction returns None."""
        behaviour = self._make_behaviour()
        behaviour._call_tweepy = _make_gen_return({"success": True})
        behaviour.context.agents_fun_db.my_agent.add_interaction = _make_gen_return(
            None
        )
        behaviour._store_agent_action = _make_gen_return(None)

        gen = BaseTweetBehaviour._handle_simple_twitter_action(
            behaviour,
            action_description="Liking tweet",
            tweepy_method_name="like_tweet",
            tweepy_kwargs={"tweet_id": "123"},
            action_class=TwitterLike,
            action_constructor_kwargs={"tweet_id": "123"},
        )
        result = _drive_gen_with_side_effects(gen)
        assert result is False

    # ----- _create_twitter_content -----

    def test_create_twitter_content_success_plain_tweet(self) -> None:
        """Test posting a plain tweet successfully."""
        behaviour = self._make_behaviour()
        mock_db_result = MagicMock()
        mock_db_result.attribute_id = "attr_1"

        behaviour._call_tweepy = _make_gen_return(["12345"])
        behaviour.context.agents_fun_db.my_agent.add_interaction = _make_gen_return(
            mock_db_result
        )
        behaviour._store_agent_action = _make_gen_return(None)

        gen = BaseTweetBehaviour._create_twitter_content(
            behaviour,
            log_message_prefix="Posting tweet",
            tweet_payload={"text": "Hello world"},
            action_text="Hello world",
        )
        result = _drive_gen_with_side_effects(gen)
        assert result is True

    def test_create_twitter_content_success_reply(self) -> None:
        """Test posting a reply successfully."""
        behaviour = self._make_behaviour()
        mock_db_result = MagicMock()
        mock_db_result.attribute_id = "attr_2"

        behaviour._call_tweepy = _make_gen_return(["12345"])
        behaviour.context.agents_fun_db.my_agent.add_interaction = _make_gen_return(
            mock_db_result
        )
        behaviour._store_agent_action = _make_gen_return(None)

        gen = BaseTweetBehaviour._create_twitter_content(
            behaviour,
            log_message_prefix="Replying to tweet",
            tweet_payload={"text": "Reply text"},
            action_text="Reply text",
            original_tweet_id_for_reply="999",
        )
        result = _drive_gen_with_side_effects(gen)
        assert result is True

    def test_create_twitter_content_success_quote(self) -> None:
        """Test posting a quote successfully."""
        behaviour = self._make_behaviour()
        mock_db_result = MagicMock()
        mock_db_result.attribute_id = "attr_3"

        behaviour._call_tweepy = _make_gen_return(["12345"])
        behaviour.context.agents_fun_db.my_agent.add_interaction = _make_gen_return(
            mock_db_result
        )
        behaviour._store_agent_action = _make_gen_return(None)

        gen = BaseTweetBehaviour._create_twitter_content(
            behaviour,
            log_message_prefix="Quoting tweet",
            tweet_payload={"text": "Quote text"},
            action_text="Quote text",
            quote_tweet_id="888",
        )
        result = _drive_gen_with_side_effects(gen)
        assert result is True

    def test_create_twitter_content_success_with_media(self) -> None:
        """Test posting a tweet with media successfully."""
        behaviour = self._make_behaviour()
        mock_db_result = MagicMock()
        mock_db_result.attribute_id = "attr_4"

        behaviour._call_tweepy = _make_gen_return(["12345"])
        behaviour.context.agents_fun_db.my_agent.add_interaction = _make_gen_return(
            mock_db_result
        )
        behaviour._store_agent_action = _make_gen_return(None)

        gen = BaseTweetBehaviour._create_twitter_content(
            behaviour,
            log_message_prefix="Posting tweet",
            tweet_payload={
                "text": "With media",
                "image_paths": ["/tmp/img.png"],
                "image_ipfs_hashes": ["QmHash"],
            },
            action_text="With media",
            media_type="image",
        )
        result = _drive_gen_with_side_effects(gen)
        assert result is True

    def test_create_twitter_content_with_media_missing_paths(self) -> None:
        """Test posting a tweet with media type but missing image_paths."""
        behaviour = self._make_behaviour()
        mock_db_result = MagicMock()
        mock_db_result.attribute_id = "attr_5"

        behaviour._call_tweepy = _make_gen_return(["12345"])
        behaviour.context.agents_fun_db.my_agent.add_interaction = _make_gen_return(
            mock_db_result
        )
        behaviour._store_agent_action = _make_gen_return(None)

        gen = BaseTweetBehaviour._create_twitter_content(
            behaviour,
            log_message_prefix="Posting tweet",
            tweet_payload={"text": "Without media paths"},
            action_text="Without media paths",
            media_type="image",
        )
        result = _drive_gen_with_side_effects(gen)
        assert result is True

    def test_create_twitter_content_tweepy_fails_main_post(self) -> None:
        """Test when tweepy call fails for a main post (returns None)."""
        behaviour = self._make_behaviour()
        behaviour._call_tweepy = _make_gen_return(None)

        gen = BaseTweetBehaviour._create_twitter_content(
            behaviour,
            log_message_prefix="Posting tweet",
            tweet_payload={"text": "test"},
            action_text="test",
        )
        result = _drive_gen_with_side_effects(gen)
        assert result is None

    def test_create_twitter_content_tweepy_fails_reply(self) -> None:
        """Test when tweepy call fails for a reply (returns False)."""
        behaviour = self._make_behaviour()
        behaviour._call_tweepy = _make_gen_return(None)

        gen = BaseTweetBehaviour._create_twitter_content(
            behaviour,
            log_message_prefix="Replying to tweet",
            tweet_payload={"text": "reply"},
            action_text="reply",
            original_tweet_id_for_reply="123",
        )
        result = _drive_gen_with_side_effects(gen)
        assert result is False

    def test_create_twitter_content_tweepy_returns_empty_list(self) -> None:
        """Test when tweepy returns empty list."""
        behaviour = self._make_behaviour()
        behaviour._call_tweepy = _make_gen_return([])

        gen = BaseTweetBehaviour._create_twitter_content(
            behaviour,
            log_message_prefix="Posting tweet",
            tweet_payload={"text": "test"},
            action_text="test",
        )
        result = _drive_gen_with_side_effects(gen)
        assert result is None

    def test_create_twitter_content_tweepy_returns_dict_error(self) -> None:
        """Test when tweepy returns a dict instead of list."""
        behaviour = self._make_behaviour()
        behaviour._call_tweepy = _make_gen_return({"error": "fail"})

        gen = BaseTweetBehaviour._create_twitter_content(
            behaviour,
            log_message_prefix="Posting tweet",
            tweet_payload={"text": "test"},
            action_text="test",
        )
        result = _drive_gen_with_side_effects(gen)
        assert result is None

    def test_create_twitter_content_tweepy_returns_list_with_none(self) -> None:
        """Test when tweepy returns [None]."""
        behaviour = self._make_behaviour()
        behaviour._call_tweepy = _make_gen_return([None])

        gen = BaseTweetBehaviour._create_twitter_content(
            behaviour,
            log_message_prefix="Posting tweet",
            tweet_payload={"text": "test"},
            action_text="test",
        )
        result = _drive_gen_with_side_effects(gen)
        assert result is None

    def test_create_twitter_content_db_add_fails(self) -> None:
        """Test when DB add_interaction returns None for _create_twitter_content."""
        behaviour = self._make_behaviour()
        behaviour._call_tweepy = _make_gen_return(["12345"])
        behaviour.context.agents_fun_db.my_agent.add_interaction = _make_gen_return(
            None
        )
        behaviour._store_agent_action = _make_gen_return(None)

        gen = BaseTweetBehaviour._create_twitter_content(
            behaviour,
            log_message_prefix="Posting tweet",
            tweet_payload={"text": "test"},
            action_text="test",
        )
        result = _drive_gen_with_side_effects(gen)
        # Still returns True because the tweet was posted
        assert result is True

    # ----- post_tweet -----

    def test_post_tweet_text_string(self) -> None:
        """Test post_tweet with a string text."""
        behaviour = self._make_behaviour()
        behaviour._create_twitter_content = _make_gen_return(True)

        gen = BaseTweetBehaviour.post_tweet(behaviour, text="Hello")
        result = _drive_gen_with_side_effects(gen)
        assert result is True

    def test_post_tweet_text_list(self) -> None:
        """Test post_tweet with a list text (takes first element)."""
        behaviour = self._make_behaviour()
        behaviour._create_twitter_content = _make_gen_return(True)

        gen = BaseTweetBehaviour.post_tweet(behaviour, text=["Hello", "World"])
        result = _drive_gen_with_side_effects(gen)
        assert result is True

    def test_post_tweet_with_images(self) -> None:
        """Test post_tweet with image paths."""
        behaviour = self._make_behaviour()
        behaviour._create_twitter_content = _make_gen_return(True)

        gen = BaseTweetBehaviour.post_tweet(
            behaviour,
            text="With image",
            image_paths=["/tmp/img.png"],
            image_ipfs_hashes=["QmHash"],
            media_type="image",
        )
        result = _drive_gen_with_side_effects(gen)
        assert result is True

    # ----- respond_tweet -----

    def test_respond_tweet_reply(self) -> None:
        """Test respond_tweet for a reply."""
        behaviour = self._make_behaviour()
        behaviour._create_twitter_content = _make_gen_return(True)

        gen = BaseTweetBehaviour.respond_tweet(behaviour, tweet_id="123", text="Reply")
        result = _drive_gen_with_side_effects(gen)
        assert result is True

    def test_respond_tweet_quote(self) -> None:
        """Test respond_tweet for a quote."""
        behaviour = self._make_behaviour()
        behaviour._create_twitter_content = _make_gen_return(True)

        gen = BaseTweetBehaviour.respond_tweet(
            behaviour, tweet_id="123", text="Quote text", quote=True, user_name="user1"
        )
        result = _drive_gen_with_side_effects(gen)
        assert result is True

    def test_respond_tweet_quote_no_username(self) -> None:
        """Test respond_tweet quote without username logs error but continues."""
        behaviour = self._make_behaviour()
        behaviour._create_twitter_content = _make_gen_return(True)

        gen = BaseTweetBehaviour.respond_tweet(
            behaviour, tweet_id="123", text="Quote", quote=True
        )
        result = _drive_gen_with_side_effects(gen)
        assert result is True

    def test_respond_tweet_returns_false(self) -> None:
        """Test respond_tweet when _create_twitter_content returns None/False."""
        behaviour = self._make_behaviour()
        behaviour._create_twitter_content = _make_gen_return(None)

        gen = BaseTweetBehaviour.respond_tweet(behaviour, tweet_id="123", text="Reply")
        result = _drive_gen_with_side_effects(gen)
        assert result is False

    # ----- like_tweet, retweet, follow_user -----

    def test_like_tweet(self) -> None:
        """Test like_tweet delegates to _handle_simple_twitter_action."""
        behaviour = self._make_behaviour()
        behaviour._handle_simple_twitter_action = _make_gen_return(True)

        gen = BaseTweetBehaviour.like_tweet(behaviour, tweet_id="123")
        result = _drive_gen_with_side_effects(gen)
        assert result is True

    def test_retweet(self) -> None:
        """Test retweet delegates to _handle_simple_twitter_action."""
        behaviour = self._make_behaviour()
        behaviour._handle_simple_twitter_action = _make_gen_return(True)

        gen = BaseTweetBehaviour.retweet(behaviour, tweet_id="123")
        result = _drive_gen_with_side_effects(gen)
        assert result is True

    def test_follow_user(self) -> None:
        """Test follow_user delegates to _handle_simple_twitter_action."""
        behaviour = self._make_behaviour()
        behaviour._handle_simple_twitter_action = _make_gen_return(True)

        gen = BaseTweetBehaviour.follow_user(behaviour, user_name="user1")
        result = _drive_gen_with_side_effects(gen)
        assert result is True


# ============================================================================
# Tests for CollectFeedbackBehaviour
# ============================================================================


class TestCollectFeedbackBehaviour:
    """Tests for CollectFeedbackBehaviour."""

    def _make_behaviour(self, posts: Any = None, **kwargs: Any) -> MagicMock:
        """Create a mock CollectFeedbackBehaviour."""
        behaviour = MagicMock(spec=CollectFeedbackBehaviour)
        my_agent = make_mock_my_agent(posts=posts)
        behaviour.context = make_mock_context(my_agent=my_agent)
        behaviour.params = make_mock_params()
        behaviour.synchronized_data = make_mock_synchronized_data(**kwargs)
        behaviour.behaviour_id = "collect_feedback"
        return behaviour

    # ----- _twitter_post_to_dict -----

    def test_twitter_post_to_dict(self) -> None:
        """Test converting TwitterPost to dict."""
        ts = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        post = make_twitter_post(
            tweet_id="123", text="Hello", timestamp=ts, reply_to_tweet_id="456"
        )
        result = CollectFeedbackBehaviour._twitter_post_to_dict(post)
        assert result["tweet_id"] == "123"
        assert result["text"] == "Hello"
        assert result["reply_to_tweet_id"] == "456"
        assert "2024-01-01" in result["timestamp"]

    def test_twitter_post_to_dict_no_reply(self) -> None:
        """Test converting TwitterPost to dict without reply_to."""
        post = make_twitter_post(tweet_id="123", text="Hello")
        result = CollectFeedbackBehaviour._twitter_post_to_dict(post)
        assert result["reply_to_tweet_id"] is None

    # ----- _process_raw_replies -----

    def test_process_raw_replies_twitter_post(self) -> None:
        """Test processing replies that are TwitterPost objects."""
        behaviour = self._make_behaviour()
        post = make_twitter_post(text="reply1")
        result = CollectFeedbackBehaviour._process_raw_replies(behaviour, [post])
        assert len(result) == 1
        assert result[0]["text"] == "reply1"

    def test_process_raw_replies_dict_with_datetime(self) -> None:
        """Test processing replies that are dicts with datetime timestamps."""
        behaviour = self._make_behaviour()
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        result = CollectFeedbackBehaviour._process_raw_replies(
            behaviour, [{"text": "reply", "timestamp": ts}]
        )
        assert len(result) == 1
        assert isinstance(result[0]["timestamp"], str)

    def test_process_raw_replies_dict_without_datetime(self) -> None:
        """Test processing replies that are dicts with string timestamps."""
        behaviour = self._make_behaviour()
        result = CollectFeedbackBehaviour._process_raw_replies(
            behaviour, [{"text": "reply", "timestamp": "2024-01-01"}]
        )
        assert len(result) == 1
        assert result[0]["timestamp"] == "2024-01-01"

    def test_process_raw_replies_not_a_list(self) -> None:
        """Test processing when raw_replies_list is not a list."""
        behaviour = self._make_behaviour()
        result = CollectFeedbackBehaviour._process_raw_replies(behaviour, "not_a_list")
        assert not result

    def test_process_raw_replies_unexpected_type(self) -> None:
        """Test processing replies with unexpected item types."""
        behaviour = self._make_behaviour()
        result = CollectFeedbackBehaviour._process_raw_replies(
            behaviour, [42, "string"]
        )
        assert not result

    # ----- get_feedback -----

    def test_get_feedback_no_posts(self) -> None:
        """Test get_feedback when there are no posts."""
        behaviour = self._make_behaviour(posts=[])
        gen = CollectFeedbackBehaviour.get_feedback(behaviour)
        result = _drive_gen_with_side_effects(gen)
        assert result == {"likes": 0, "retweets": 0, "replies": []}

    def test_get_feedback_last_tweet_none(self) -> None:
        """Test get_feedback when last tweet is None."""
        behaviour = self._make_behaviour(posts=[None])
        gen = CollectFeedbackBehaviour.get_feedback(behaviour)
        result = _drive_gen_with_side_effects(gen)
        assert result == {"likes": 0, "retweets": 0, "replies": []}

    def test_get_feedback_db_returns_none(self) -> None:
        """Test get_feedback when DB returns None feedback."""
        post = make_twitter_post(tweet_id="123", text="Hello")
        behaviour = self._make_behaviour(posts=[post])
        behaviour.context.agents_fun_db.get_tweet_feedback = _make_gen_return(None)

        gen = CollectFeedbackBehaviour.get_feedback(behaviour)
        result = _drive_gen_with_side_effects(gen)
        assert result == {"likes": 0, "retweets": 0, "replies": []}

    def test_get_feedback_success(self) -> None:
        """Test get_feedback with successful DB response."""
        post = make_twitter_post(tweet_id="123", text="Hello")
        behaviour = self._make_behaviour(posts=[post])
        behaviour.context.agents_fun_db.get_tweet_feedback = _make_gen_return(
            {
                "likes": 5,
                "retweets": 2,
                "replies": [{"text": "reply1", "timestamp": "2024-01-01"}],
            }
        )
        behaviour._process_raw_replies = CollectFeedbackBehaviour._process_raw_replies.__get__(  # pylint: disable=no-value-for-parameter
            behaviour
        )

        gen = CollectFeedbackBehaviour.get_feedback(behaviour)
        result = _drive_gen_with_side_effects(gen)
        assert result["likes"] == 5
        assert result["retweets"] == 2
        assert len(result["replies"]) == 1

    # ----- async_act -----

    def test_async_act(self) -> None:
        """Test CollectFeedbackBehaviour.async_act."""
        behaviour = self._make_behaviour()
        behaviour.get_feedback = _make_gen_return(
            {"likes": 0, "retweets": 0, "replies": []}
        )
        behaviour.send_a2a_transaction = _make_gen_return(None)
        behaviour.wait_until_round_end = _make_gen_return(None)
        behaviour.set_done = MagicMock()

        gen = CollectFeedbackBehaviour.async_act(behaviour)
        _drive_gen_with_side_effects(gen)
        behaviour.set_done.assert_called_once()

    def test_async_act_feedback_is_none(self) -> None:
        """Test async_act when feedback is None."""
        behaviour = self._make_behaviour()
        behaviour.get_feedback = _make_gen_return(None)
        behaviour.send_a2a_transaction = _make_gen_return(None)
        behaviour.wait_until_round_end = _make_gen_return(None)
        behaviour.set_done = MagicMock()

        gen = CollectFeedbackBehaviour.async_act(behaviour)
        _drive_gen_with_side_effects(gen)
        behaviour.set_done.assert_called_once()


# ============================================================================
# Tests for EngageTwitterBehaviour
# ============================================================================


class TestEngageTwitterBehaviour:  # pylint: disable=too-many-public-methods
    """Tests for EngageTwitterBehaviour."""

    def _make_behaviour(
        self, posts: Any = None, active_agents: Any = None, **sync_kwargs: Any
    ) -> MagicMock:
        """Create a mock EngageTwitterBehaviour."""
        behaviour = MagicMock(spec=EngageTwitterBehaviour)
        my_agent = make_mock_my_agent(posts=posts)
        behaviour.context = make_mock_context(
            my_agent=my_agent, active_agents=active_agents
        )
        behaviour.params = make_mock_params()
        behaviour.synchronized_data = make_mock_synchronized_data(**sync_kwargs)
        behaviour.behaviour_id = "engage_twitter"
        behaviour.matching_round = MagicMock()
        behaviour.matching_round.auto_round_id.return_value = "engage_twitter_round"
        return behaviour

    # ----- async_act -----

    def test_async_act_invalid_auth(self) -> None:
        """Test async_act when twitter auth fails."""
        behaviour = self._make_behaviour()
        behaviour._call_tweepy = _make_gen_return({"error": "auth failed"})
        behaviour.send_a2a_transaction = _make_gen_return(None)
        behaviour.wait_until_round_end = _make_gen_return(None)
        behaviour.set_done = MagicMock()

        gen = EngageTwitterBehaviour.async_act(behaviour)
        _drive_gen_with_side_effects(gen)
        behaviour.set_done.assert_called_once()

    def test_async_act_auth_returns_none(self) -> None:
        """Test async_act when twitter auth returns None."""
        behaviour = self._make_behaviour()
        behaviour._call_tweepy = _make_gen_return(None)
        behaviour.send_a2a_transaction = _make_gen_return(None)
        behaviour.wait_until_round_end = _make_gen_return(None)
        behaviour.set_done = MagicMock()

        gen = EngageTwitterBehaviour.async_act(behaviour)
        _drive_gen_with_side_effects(gen)
        behaviour.set_done.assert_called_once()

    def test_async_act_auth_ok_done(self) -> None:
        """Test async_act when auth is OK and event is DONE."""
        behaviour = self._make_behaviour()
        behaviour._call_tweepy = _make_gen_return({"user": "me"})
        behaviour.get_event = _make_gen_return((Event.DONE.value, []))
        behaviour.send_a2a_transaction = _make_gen_return(None)
        behaviour.wait_until_round_end = _make_gen_return(None)
        behaviour.set_done = MagicMock()

        gen = EngageTwitterBehaviour.async_act(behaviour)
        _drive_gen_with_side_effects(gen)
        behaviour.set_done.assert_called_once()

    def test_async_act_auth_ok_mech_event(self) -> None:
        """Test async_act when auth is OK and event is MECH."""
        behaviour = self._make_behaviour()
        behaviour._call_tweepy = _make_gen_return({"user": "me"})
        mech_request = [{"nonce": "1", "tool": "t", "prompt": "p"}]
        behaviour.get_event = _make_gen_return((Event.MECH.value, mech_request))
        behaviour.send_a2a_transaction = _make_gen_return(None)
        behaviour.wait_until_round_end = _make_gen_return(None)
        behaviour.set_done = MagicMock()

        gen = EngageTwitterBehaviour.async_act(behaviour)
        _drive_gen_with_side_effects(gen)
        behaviour.set_done.assert_called_once()

    # ----- get_event -----

    def test_get_event_mech_for_twitter(self) -> None:
        """Test get_event when mech_for_twitter is True."""
        behaviour = self._make_behaviour(mech_for_twitter=True)
        behaviour._handle_mech_for_twitter = _make_gen_return(
            ({"123": {"text": "test", "user_name": "user1"}}, [])
        )
        behaviour.interact_twitter = _make_gen_return((Event.DONE.value, [123], []))
        behaviour._update_interacted_tweets = _make_gen_return(None)

        gen = EngageTwitterBehaviour.get_event(behaviour)
        result = _drive_gen_with_side_effects(gen)
        assert result[0] == Event.DONE.value

    def test_get_event_skip_engagement(self) -> None:
        """Test get_event when skip_engagement is True."""
        behaviour = self._make_behaviour()
        behaviour.params.skip_engagement = True
        behaviour.synchronized_data.mech_for_twitter = False
        behaviour.interact_twitter = _make_gen_return((Event.DONE.value, [], []))
        behaviour._update_interacted_tweets = _make_gen_return(None)

        gen = EngageTwitterBehaviour.get_event(behaviour)
        result = _drive_gen_with_side_effects(gen)
        assert result[0] == Event.DONE.value

    def test_get_event_regular_engagement(self) -> None:
        """Test get_event with regular engagement."""
        behaviour = self._make_behaviour(mech_for_twitter=False)
        behaviour.params.skip_engagement = False
        behaviour._handle_regular_engagement = _make_gen_return(({}, []))
        behaviour.interact_twitter = _make_gen_return((Event.DONE.value, [], []))
        behaviour._update_interacted_tweets = _make_gen_return(None)

        gen = EngageTwitterBehaviour.get_event(behaviour)
        result = _drive_gen_with_side_effects(gen)
        assert result[0] == Event.DONE.value

    def test_get_event_mech_event(self) -> None:
        """Test get_event returns MECH event."""
        behaviour = self._make_behaviour(mech_for_twitter=False)
        behaviour.params.skip_engagement = False
        behaviour._handle_regular_engagement = _make_gen_return(({}, []))
        mech_requests = [{"nonce": "1", "tool": "t", "prompt": "p"}]
        behaviour.interact_twitter = _make_gen_return(
            (Event.MECH.value, [], mech_requests)
        )

        gen = EngageTwitterBehaviour.get_event(behaviour)
        result = _drive_gen_with_side_effects(gen)
        assert result[0] == Event.MECH.value
        assert result[1] == mech_requests

    # ----- _handle_mech_for_twitter -----

    def test_handle_mech_for_twitter_with_data(self) -> None:
        """Test _handle_mech_for_twitter with stored data."""
        behaviour = self._make_behaviour(mech_for_twitter=True)
        pending = {"123": {"text": "test", "user_name": "user1"}}
        interacted = [100, 200]
        behaviour._read_json_from_kv = MagicMock(
            side_effect=[
                _make_gen_return(pending)(),
                _make_gen_return(interacted)(),
            ]
        )

        gen = EngageTwitterBehaviour._handle_mech_for_twitter(behaviour)
        result = _drive_gen_with_side_effects(gen)
        assert result[0] == pending
        assert result[1] == interacted

    def test_handle_mech_for_twitter_empty_data(self) -> None:
        """Test _handle_mech_for_twitter with empty stored data."""
        behaviour = self._make_behaviour(mech_for_twitter=True)
        behaviour._read_json_from_kv = MagicMock(
            side_effect=[
                _make_gen_return({})(),
                _make_gen_return([])(),
            ]
        )

        gen = EngageTwitterBehaviour._handle_mech_for_twitter(behaviour)
        result = _drive_gen_with_side_effects(gen)
        assert result[0] == {}
        assert not result[1]

    # ----- _handle_regular_engagement -----

    def test_handle_regular_engagement_no_agents(self) -> None:
        """Test _handle_regular_engagement when no active agents found."""
        behaviour = self._make_behaviour()
        behaviour.get_agent_handles = _make_gen_return([])

        gen = EngageTwitterBehaviour._handle_regular_engagement(behaviour)
        result = _drive_gen_with_side_effects(gen)
        assert result == ({}, [])

    def test_handle_regular_engagement_with_agents(self) -> None:
        """Test _handle_regular_engagement with active agents."""
        behaviour = self._make_behaviour()
        mock_agents = [MagicMock()]
        behaviour.get_agent_handles = _make_gen_return(mock_agents)
        behaviour._get_interacted_tweet_ids = _make_gen_return([])
        behaviour._collect_pending_tweets = _make_gen_return(
            {"123": {"text": "test", "user_name": "user1"}}
        )
        behaviour._store_engagement_data = _make_gen_return(None)

        gen = EngageTwitterBehaviour._handle_regular_engagement(behaviour)
        result = _drive_gen_with_side_effects(gen)
        assert "123" in result[0]

    # ----- _get_interacted_tweet_ids -----

    def test_get_interacted_tweet_ids(self) -> None:
        """Test _get_interacted_tweet_ids reads from KV."""
        behaviour = self._make_behaviour()
        behaviour._read_json_from_kv = _make_gen_return([1, 2, 3])

        gen = EngageTwitterBehaviour._get_interacted_tweet_ids(behaviour)
        result = _drive_gen_with_side_effects(gen)
        assert result == [1, 2, 3]

    # ----- _collect_pending_tweets -----

    def test_collect_pending_tweets_no_agents(self) -> None:
        """Test _collect_pending_tweets with empty agent list."""
        behaviour = self._make_behaviour()

        gen = EngageTwitterBehaviour._collect_pending_tweets(behaviour, [], set())
        result = _drive_gen_with_side_effects(gen)
        assert result == {}

    def test_collect_pending_tweets_agent_not_loaded(self) -> None:
        """Test _collect_pending_tweets with unloaded agent."""
        behaviour = self._make_behaviour()
        agent = MagicMock()
        agent.loaded = False
        agent.agent_instance.agent_id = "agent1"
        agent.load = _make_gen_return(None)
        agent.posts = [make_twitter_post(tweet_id="123", text="hello")]
        agent.twitter_username = "user1"

        gen = EngageTwitterBehaviour._collect_pending_tweets(behaviour, [agent], set())
        result = _drive_gen_with_side_effects(gen)
        assert "123" in result

    def test_collect_pending_tweets_no_posts(self) -> None:
        """Test _collect_pending_tweets when agent has no posts."""
        behaviour = self._make_behaviour()
        agent = MagicMock()
        agent.loaded = True
        agent.posts = []
        agent.twitter_username = "user1"

        gen = EngageTwitterBehaviour._collect_pending_tweets(behaviour, [agent], set())
        result = _drive_gen_with_side_effects(gen)
        assert result == {}

    def test_collect_pending_tweets_already_interacted(self) -> None:
        """Test _collect_pending_tweets skips already interacted tweets."""
        behaviour = self._make_behaviour()
        agent = MagicMock()
        agent.loaded = True
        post = MagicMock()
        post.tweet_id = 123
        post.text = "hello"
        agent.posts = [post]
        agent.twitter_username = "user1"

        gen = EngageTwitterBehaviour._collect_pending_tweets(behaviour, [agent], {123})
        result = _drive_gen_with_side_effects(gen)
        assert result == {}

    def test_collect_pending_tweets_no_username(self) -> None:
        """Test _collect_pending_tweets when agent has no twitter_username."""
        behaviour = self._make_behaviour()
        agent = MagicMock()
        agent.loaded = True
        post = MagicMock()
        post.tweet_id = 123
        post.text = "hello"
        agent.posts = [post]
        agent.twitter_username = None
        agent.agent_instance.agent_id = "agent1"

        gen = EngageTwitterBehaviour._collect_pending_tweets(behaviour, [agent], set())
        result = _drive_gen_with_side_effects(gen)
        assert result == {}

    def test_collect_pending_tweets_success(self) -> None:
        """Test _collect_pending_tweets with valid agent and posts."""
        behaviour = self._make_behaviour()
        agent = MagicMock()
        agent.loaded = True
        post = MagicMock()
        post.tweet_id = 456
        post.text = "hello world"
        agent.posts = [post]
        agent.twitter_username = "user2"

        gen = EngageTwitterBehaviour._collect_pending_tweets(behaviour, [agent], set())
        result = _drive_gen_with_side_effects(gen)
        assert "456" in result
        assert result["456"]["text"] == "hello world"
        assert result["456"]["user_name"] == "user2"

    # ----- _store_engagement_data -----

    def test_store_engagement_data(self) -> None:
        """Test _store_engagement_data writes to KV store."""
        behaviour = self._make_behaviour()
        behaviour._write_kv = _make_gen_return(True)

        gen = EngageTwitterBehaviour._store_engagement_data(
            behaviour, [1, 2], {"123": {"text": "test", "user_name": "user1"}}
        )
        _drive_gen_with_side_effects(gen)

    # ----- _update_interacted_tweets -----

    def test_update_interacted_tweets(self) -> None:
        """Test _update_interacted_tweets extends list and writes to KV."""
        behaviour = self._make_behaviour()
        behaviour._write_kv = _make_gen_return(True)
        existing = [1, 2]

        gen = EngageTwitterBehaviour._update_interacted_tweets(
            behaviour, existing, [3, 4]
        )
        _drive_gen_with_side_effects(gen)
        assert existing == [1, 2, 3, 4]

    # ----- interact_twitter -----

    def test_interact_twitter_llm_returns_none(self) -> None:
        """Test interact_twitter when LLM returns None."""
        behaviour = self._make_behaviour()
        behaviour.get_persona = _make_gen_return("test persona")
        behaviour._read_value_from_kv = _make_gen_return(None)
        behaviour._prepare_prompt_data = _make_gen_return(("prompt", None))
        behaviour._get_llm_decision = _make_gen_return(None)

        gen = EngageTwitterBehaviour.interact_twitter(behaviour, {})
        result = _drive_gen_with_side_effects(gen)
        assert result[0] == Event.ERROR.value

    def test_interact_twitter_json_decode_error_then_fails(self) -> None:
        """Test interact_twitter when LLM returns invalid JSON 3 times."""
        behaviour = self._make_behaviour()
        behaviour.get_persona = _make_gen_return("test persona")
        behaviour._read_value_from_kv = _make_gen_return(None)
        behaviour._prepare_prompt_data = _make_gen_return(("prompt", None))
        # Return invalid JSON 3 times
        call_count = [0]

        def _get_llm_decision_gen(
            *args: Any, **kwargs: Any
        ) -> Generator[Any, None, Any]:
            call_count[0] += 1
            if False:  # pylint: disable=using-constant-test  # pragma: no cover
                yield
            return "not json {"

        behaviour._get_llm_decision = _get_llm_decision_gen

        gen = EngageTwitterBehaviour.interact_twitter(behaviour, {})
        result = _drive_gen_with_side_effects(gen)
        assert result[0] == Event.ERROR.value

    def test_interact_twitter_invalid_format_retries(self) -> None:
        """Test interact_twitter retries on invalid format."""
        behaviour = self._make_behaviour()
        behaviour.get_persona = _make_gen_return("test persona")
        behaviour._read_value_from_kv = _make_gen_return("stored prompt")
        behaviour._prepare_prompt_data = _make_gen_return(("prompt", None))
        behaviour._validate_llm_response = MagicMock(return_value=False)

        # Return valid JSON but invalid format 3 times
        def _get_llm_decision_gen(
            *args: Any, **kwargs: Any
        ) -> Generator[Any, None, Any]:
            if False:  # pylint: disable=using-constant-test  # pragma: no cover
                yield
            return json.dumps({"bad": "format"})

        behaviour._get_llm_decision = _get_llm_decision_gen

        gen = EngageTwitterBehaviour.interact_twitter(behaviour, {})
        result = _drive_gen_with_side_effects(gen)
        assert result[0] == Event.ERROR.value

    def test_interact_twitter_tool_action(self) -> None:
        """Test interact_twitter with tool_action in response."""
        behaviour = self._make_behaviour()
        behaviour.get_persona = _make_gen_return("test persona")
        behaviour._read_value_from_kv = _make_gen_return(None)
        behaviour._prepare_prompt_data = _make_gen_return(("prompt", None))
        behaviour._validate_llm_response = MagicMock(return_value=True)
        behaviour._get_llm_decision = _make_gen_return(
            json.dumps({"tool_action": {"tool_name": "gen_image", "tool_input": "cat"}})
        )
        behaviour._handle_tool_action = _make_gen_return(
            (Event.MECH.value, [], [{"nonce": "1"}])
        )
        behaviour.synchronized_data.mech_for_twitter = False

        gen = EngageTwitterBehaviour.interact_twitter(behaviour, {})
        result = _drive_gen_with_side_effects(gen)
        assert result[0] == Event.MECH.value

    def test_interact_twitter_tool_action_when_mech_for_twitter(self) -> None:
        """Test interact_twitter with tool_action when mech_for_twitter is True (error)."""
        behaviour = self._make_behaviour(mech_for_twitter=True)
        behaviour.get_persona = _make_gen_return("test persona")
        behaviour._read_value_from_kv = _make_gen_return(None)
        behaviour._prepare_prompt_data = _make_gen_return(("prompt", None))
        behaviour._validate_llm_response = MagicMock(return_value=True)
        behaviour._get_llm_decision = _make_gen_return(
            json.dumps({"tool_action": {"tool_name": "gen_image", "tool_input": "cat"}})
        )

        gen = EngageTwitterBehaviour.interact_twitter(behaviour, {})
        result = _drive_gen_with_side_effects(gen)
        assert result[0] == Event.ERROR.value

    def test_interact_twitter_tweet_action(self) -> None:
        """Test interact_twitter with tweet_action in response."""
        behaviour = self._make_behaviour()
        behaviour.get_persona = _make_gen_return("test persona")
        behaviour._read_value_from_kv = _make_gen_return(None)
        behaviour._prepare_prompt_data = _make_gen_return(("prompt", []))
        behaviour._validate_llm_response = MagicMock(return_value=True)
        behaviour._get_llm_decision = _make_gen_return(
            json.dumps({"tweet_action": {"action": "tweet", "text": "Hello"}})
        )
        behaviour._handle_tweet_actions = _make_gen_return(
            (Event.DONE.value, [123], [])
        )

        gen = EngageTwitterBehaviour.interact_twitter(behaviour, {})
        result = _drive_gen_with_side_effects(gen)
        assert result[0] == Event.DONE.value

    def test_interact_twitter_no_valid_action(self) -> None:
        """Test interact_twitter when response has neither tool_action nor tweet_action."""
        behaviour = self._make_behaviour()
        behaviour.get_persona = _make_gen_return("test persona")
        behaviour._read_value_from_kv = _make_gen_return(None)
        behaviour._prepare_prompt_data = _make_gen_return(("prompt", None))
        behaviour._validate_llm_response = MagicMock(return_value=True)
        behaviour._get_llm_decision = _make_gen_return(
            json.dumps({"something_else": "value"})
        )

        gen = EngageTwitterBehaviour.interact_twitter(behaviour, {})
        result = _drive_gen_with_side_effects(gen)
        assert result[0] == Event.ERROR.value

    # ----- _prepare_prompt_data -----

    def test_prepare_prompt_data_mech(self) -> None:
        """Test _prepare_prompt_data when mech_for_twitter is True."""
        behaviour = self._make_behaviour(mech_for_twitter=True)
        behaviour._prepare_mech_prompt_data = _make_gen_return(("mech prompt", []))
        behaviour._write_kv = _make_gen_return(True)

        gen = EngageTwitterBehaviour._prepare_prompt_data(behaviour, {}, "persona")
        result = _drive_gen_with_side_effects(gen)
        assert result[0] == "mech prompt"

    def test_prepare_prompt_data_standard(self) -> None:
        """Test _prepare_prompt_data when mech_for_twitter is False."""
        behaviour = self._make_behaviour(mech_for_twitter=False)
        behaviour._prepare_standard_prompt_data = _make_gen_return(
            ("standard prompt", None)
        )
        behaviour._write_kv = _make_gen_return(True)

        gen = EngageTwitterBehaviour._prepare_prompt_data(behaviour, {}, "persona")
        result = _drive_gen_with_side_effects(gen)
        assert result[0] == "standard prompt"

    # ----- _prepare_standard_prompt_data -----

    def test_prepare_standard_prompt_data_no_pending(self) -> None:
        """Test _prepare_standard_prompt_data with no pending tweets."""
        behaviour = self._make_behaviour(
            mech_for_twitter=False, is_staking_kpi_met=True
        )
        behaviour.context.agents_fun_db.my_agent.posts = []
        behaviour._format_previous_tweets_str = MagicMock(return_value="")
        behaviour.get_sync_time_str = MagicMock(return_value="2024-01-01 12:00:00")
        behaviour.generate_mech_tool_info = MagicMock(return_value="tools_info")
        behaviour.get_latest_agent_actions = _make_gen_return([])
        behaviour._get_shuffled_twitter_actions = MagicMock(
            return_value="- Tweet\n- Like"
        )
        behaviour._save_standard_kv_data = _make_gen_return(None)
        behaviour._read_value_from_kv = _make_gen_return(None)

        gen = EngageTwitterBehaviour._prepare_standard_prompt_data(
            behaviour, {}, "test persona"
        )
        result = _drive_gen_with_side_effects(gen)
        assert isinstance(result[0], str)

    def test_prepare_standard_prompt_data_with_pending_tweets(self) -> None:
        """Test _prepare_standard_prompt_data with pending tweets."""
        behaviour = self._make_behaviour(
            mech_for_twitter=False, is_staking_kpi_met=False
        )
        posts = [make_twitter_post(text="prev tweet")]
        behaviour.context.agents_fun_db.my_agent.posts = posts
        behaviour._format_previous_tweets_str = MagicMock(return_value="prev tweet")
        behaviour.get_sync_time_str = MagicMock(return_value="2024-01-01 12:00:00")
        behaviour.generate_mech_tool_info = MagicMock(return_value="tools_info")
        behaviour.get_latest_agent_actions = _make_gen_return([])
        behaviour._get_shuffled_twitter_actions = MagicMock(return_value="- Tweet")
        behaviour._save_standard_kv_data = _make_gen_return(None)
        behaviour._read_value_from_kv = _make_gen_return(None)

        pending = {"123": {"text": "other tweet", "user_name": "user1"}}
        gen = EngageTwitterBehaviour._prepare_standard_prompt_data(
            behaviour, pending, "persona"
        )
        result = _drive_gen_with_side_effects(gen)
        assert isinstance(result[0], str)

    def test_prepare_standard_prompt_data_with_dict_posts(self) -> None:
        """Test _prepare_standard_prompt_data with dict-type posts."""
        behaviour = self._make_behaviour(
            mech_for_twitter=False, is_staking_kpi_met=True
        )
        behaviour.context.agents_fun_db.my_agent.posts = [
            {"tweet_id": "123", "text": "prev", "timestamp": "2024-01-01"}
        ]
        behaviour._format_previous_tweets_str = MagicMock(return_value="prev")
        behaviour.get_sync_time_str = MagicMock(return_value="2024-01-01 12:00:00")
        behaviour.generate_mech_tool_info = MagicMock(return_value="")
        behaviour.get_latest_agent_actions = _make_gen_return([])
        behaviour._get_shuffled_twitter_actions = MagicMock(return_value="- Tweet")
        behaviour._save_standard_kv_data = _make_gen_return(None)
        behaviour._read_value_from_kv = _make_gen_return(None)

        gen = EngageTwitterBehaviour._prepare_standard_prompt_data(
            behaviour, {}, "persona"
        )
        result = _drive_gen_with_side_effects(gen)
        assert isinstance(result[0], str)

    def test_prepare_standard_prompt_data_failed_mech(self) -> None:
        """Test _prepare_standard_prompt_data when failed_mech is True."""
        behaviour = self._make_behaviour(mech_for_twitter=False, failed_mech=True)
        behaviour.context.agents_fun_db.my_agent.posts = []
        behaviour._format_previous_tweets_str = MagicMock(return_value="")
        behaviour.get_sync_time_str = MagicMock(return_value="2024-01-01 12:00:00")
        behaviour.generate_mech_tool_info = MagicMock(return_value="")
        behaviour.get_latest_agent_actions = _make_gen_return([])
        behaviour._get_shuffled_twitter_actions = MagicMock(return_value="- Tweet")
        behaviour._save_standard_kv_data = _make_gen_return(None)
        behaviour._read_value_from_kv = _make_gen_return("previous prompt")

        gen = EngageTwitterBehaviour._prepare_standard_prompt_data(
            behaviour, {}, "persona"
        )
        result = _drive_gen_with_side_effects(gen)
        assert isinstance(result[0], str)

    # ----- _save_standard_kv_data -----

    def test_save_standard_kv_data(self) -> None:
        """Test _save_standard_kv_data."""
        behaviour = self._make_behaviour()
        behaviour._write_kv = _make_gen_return(True)

        gen = EngageTwitterBehaviour._save_standard_kv_data(
            behaviour, [{"tweet_id": "1"}], {"123": {"text": "t", "user_name": "u"}}
        )
        _drive_gen_with_side_effects(gen)

    def test_save_standard_kv_data_none_tweets(self) -> None:
        """Test _save_standard_kv_data with None tweets."""
        behaviour = self._make_behaviour()
        behaviour._write_kv = _make_gen_return(True)

        gen = EngageTwitterBehaviour._save_standard_kv_data(behaviour, None, {})
        _drive_gen_with_side_effects(gen)

    # ----- _prepare_mech_prompt_data -----

    def test_prepare_mech_prompt_data(self) -> None:
        """Test _prepare_mech_prompt_data."""
        behaviour = self._make_behaviour(
            mech_for_twitter=True, mech_responses=[{"response": "data"}]
        )
        behaviour._read_json_from_kv = MagicMock(
            side_effect=[
                _make_gen_return([{"text": "prev"}])(),
                _make_gen_return({"123": {"text": "other", "user_name": "u1"}})(),
            ]
        )
        behaviour._format_previous_tweets_str = MagicMock(return_value="prev tweets")
        behaviour._determine_mech_summary = _make_gen_return("mech summary")
        behaviour.generate_mech_tool_info = MagicMock(return_value="tools")
        behaviour.get_sync_time_str = MagicMock(return_value="2024-01-01 12:00:00")
        behaviour._clear_mech_kv_data = _make_gen_return(None)

        gen = EngageTwitterBehaviour._prepare_mech_prompt_data(behaviour, "persona")
        result = _drive_gen_with_side_effects(gen)
        assert isinstance(result[0], str)

    def test_prepare_mech_prompt_data_no_mech_responses(self) -> None:
        """Test _prepare_mech_prompt_data with no mech responses."""
        behaviour = self._make_behaviour(mech_for_twitter=True, mech_responses=[])
        behaviour._read_json_from_kv = MagicMock(
            side_effect=[
                _make_gen_return([])(),
                _make_gen_return({})(),
            ]
        )
        behaviour._format_previous_tweets_str = MagicMock(return_value="")
        behaviour._determine_mech_summary = _make_gen_return("fallback summary")
        behaviour.generate_mech_tool_info = MagicMock(return_value="")
        behaviour.get_sync_time_str = MagicMock(return_value="2024-01-01 12:00:00")
        behaviour._clear_mech_kv_data = _make_gen_return(None)

        gen = EngageTwitterBehaviour._prepare_mech_prompt_data(behaviour, "persona")
        result = _drive_gen_with_side_effects(gen)
        assert isinstance(result[0], str)

    def test_prepare_mech_prompt_data_other_tweets_not_dict(self) -> None:
        """Test _prepare_mech_prompt_data when other_tweets_data is not dict."""
        behaviour = self._make_behaviour(
            mech_for_twitter=True, mech_responses=[{"response": "data"}]
        )
        behaviour._read_json_from_kv = MagicMock(
            side_effect=[
                _make_gen_return([])(),
                _make_gen_return("not_a_dict")(),
            ]
        )
        behaviour._format_previous_tweets_str = MagicMock(return_value="")
        behaviour._determine_mech_summary = _make_gen_return("summary")
        behaviour.generate_mech_tool_info = MagicMock(return_value="")
        behaviour.get_sync_time_str = MagicMock(return_value="2024-01-01 12:00:00")
        behaviour._clear_mech_kv_data = _make_gen_return(None)

        gen = EngageTwitterBehaviour._prepare_mech_prompt_data(behaviour, "persona")
        result = _drive_gen_with_side_effects(gen)
        assert "No other tweets found" in result[0]

    # ----- _clear_mech_kv_data -----

    def test_clear_mech_kv_data(self) -> None:
        """Test _clear_mech_kv_data writes empty strings."""
        behaviour = self._make_behaviour()
        behaviour._write_kv = _make_gen_return(True)

        gen = EngageTwitterBehaviour._clear_mech_kv_data(behaviour)
        _drive_gen_with_side_effects(gen)

    # ----- _get_latest_media_info -----

    def test_get_latest_media_info_found(self) -> None:
        """Test _get_latest_media_info with valid media info."""
        behaviour = self._make_behaviour()
        behaviour._read_json_from_kv = _make_gen_return(
            {"type": "image", "path": "/tmp/img.png", "ipfs_hash": "QmHash"}
        )

        gen = EngageTwitterBehaviour._get_latest_media_info(behaviour)
        result = _drive_gen_with_side_effects(gen)
        assert result["type"] == "image"

    def test_get_latest_media_info_not_found(self) -> None:
        """Test _get_latest_media_info when no media info."""
        behaviour = self._make_behaviour()
        behaviour._read_json_from_kv = _make_gen_return(None)

        gen = EngageTwitterBehaviour._get_latest_media_info(behaviour)
        result = _drive_gen_with_side_effects(gen)
        assert result is None

    def test_get_latest_media_info_json_error(self) -> None:
        """Test _get_latest_media_info when JSON decode fails."""
        behaviour = self._make_behaviour()

        def _raises_json(*args: Any, **kwargs: Any) -> Generator[Any, None, None]:
            if False:  # pylint: disable=using-constant-test  # pragma: no cover
                yield
            raise json.JSONDecodeError("test", "doc", 0)

        behaviour._read_json_from_kv = _raises_json

        gen = EngageTwitterBehaviour._get_latest_media_info(behaviour)
        result = _drive_gen_with_side_effects(gen)
        assert result is None

    def test_get_latest_media_info_generic_exception(self) -> None:
        """Test _get_latest_media_info when generic exception occurs."""
        behaviour = self._make_behaviour()

        def _raises(*args: Any, **kwargs: Any) -> Generator[Any, None, None]:
            if False:  # pylint: disable=using-constant-test  # pragma: no cover
                yield
            raise RuntimeError("unexpected error")

        behaviour._read_json_from_kv = _raises

        gen = EngageTwitterBehaviour._get_latest_media_info(behaviour)
        result = _drive_gen_with_side_effects(gen)
        assert result is None

    # ----- _determine_mech_summary -----

    def test_determine_mech_summary_image(self) -> None:
        """Test _determine_mech_summary with image media info."""
        behaviour = self._make_behaviour()
        behaviour._get_latest_media_info = _make_gen_return({"type": "image"})

        gen = EngageTwitterBehaviour._determine_mech_summary(behaviour)
        result = _drive_gen_with_side_effects(gen)
        assert "image" in result

    def test_determine_mech_summary_video(self) -> None:
        """Test _determine_mech_summary with video media info."""
        behaviour = self._make_behaviour()
        behaviour._get_latest_media_info = _make_gen_return({"type": "video"})

        gen = EngageTwitterBehaviour._determine_mech_summary(behaviour)
        result = _drive_gen_with_side_effects(gen)
        assert "video" in result

    def test_determine_mech_summary_unknown_type(self) -> None:
        """Test _determine_mech_summary with unknown media type."""
        behaviour = self._make_behaviour()
        behaviour._get_latest_media_info = _make_gen_return({"type": "audio"})

        gen = EngageTwitterBehaviour._determine_mech_summary(behaviour)
        result = _drive_gen_with_side_effects(gen)
        assert "failed" in result

    def test_determine_mech_summary_no_media(self) -> None:
        """Test _determine_mech_summary with no media info."""
        behaviour = self._make_behaviour()
        behaviour._get_latest_media_info = _make_gen_return(None)

        gen = EngageTwitterBehaviour._determine_mech_summary(behaviour)
        result = _drive_gen_with_side_effects(gen)
        assert "failed" in result

    # ----- _get_llm_decision -----

    def test_get_llm_decision(self) -> None:
        """Test _get_llm_decision calls genai."""
        behaviour = self._make_behaviour()
        behaviour._call_genai = _make_gen_return(
            '{"tweet_action": {"action": "tweet"}}'
        )

        gen = EngageTwitterBehaviour._get_llm_decision(behaviour, "test prompt")
        result = _drive_gen_with_side_effects(gen)
        assert result is not None

    # ----- Validation methods -----

    def test_validate_llm_response_not_dict(self) -> None:
        """Test _validate_llm_response with non-dict input."""
        behaviour = self._make_behaviour()
        result = EngageTwitterBehaviour._validate_llm_response(behaviour, "not a dict")  # type: ignore[arg-type]
        assert result is False

    def test_validate_llm_response_mech_for_twitter(self) -> None:
        """Test _validate_llm_response when mech_for_twitter is True."""
        behaviour = self._make_behaviour(mech_for_twitter=True)
        behaviour._validate_mech_llm_response = EngageTwitterBehaviour._validate_mech_llm_response.__get__(  # pylint: disable=no-value-for-parameter
            behaviour
        )
        result = EngageTwitterBehaviour._validate_llm_response(
            behaviour,
            {"tweet_action": {"action_type": "tweet_with_media", "text": "hello"}},
        )
        assert result is True

    def test_validate_llm_response_non_mech(self) -> None:
        """Test _validate_llm_response when mech_for_twitter is False."""
        behaviour = self._make_behaviour(mech_for_twitter=False)
        behaviour._validate_non_mech_llm_response = EngageTwitterBehaviour._validate_non_mech_llm_response.__get__(  # pylint: disable=no-value-for-parameter
            behaviour
        )
        behaviour._validate_non_mech_tweet_action = EngageTwitterBehaviour._validate_non_mech_tweet_action.__get__(  # pylint: disable=no-value-for-parameter
            behaviour
        )
        result = EngageTwitterBehaviour._validate_llm_response(
            behaviour, {"tweet_action": {"action": "tweet", "text": "hello"}}
        )
        assert result is True

    def test_validate_mech_llm_response_valid(self) -> None:
        """Test _validate_mech_llm_response with valid response."""
        behaviour = self._make_behaviour()
        result = EngageTwitterBehaviour._validate_mech_llm_response(
            behaviour,
            {"tweet_action": {"action_type": "tweet_with_media", "text": "hello"}},
        )
        assert result is True

    def test_validate_mech_llm_response_action_field(self) -> None:
        """Test _validate_mech_llm_response with 'action' instead of 'action_type'."""
        behaviour = self._make_behaviour()
        result = EngageTwitterBehaviour._validate_mech_llm_response(
            behaviour, {"tweet_action": {"action": "tweet_with_media", "text": "hello"}}
        )
        assert result is True

    def test_validate_mech_llm_response_no_tweet_action(self) -> None:
        """Test _validate_mech_llm_response without tweet_action."""
        behaviour = self._make_behaviour()
        result = EngageTwitterBehaviour._validate_mech_llm_response(
            behaviour, {"tool_action": {}}
        )
        assert result is False

    def test_validate_mech_llm_response_wrong_action(self) -> None:
        """Test _validate_mech_llm_response with wrong action type."""
        behaviour = self._make_behaviour()
        result = EngageTwitterBehaviour._validate_mech_llm_response(
            behaviour, {"tweet_action": {"action_type": "tweet", "text": "hello"}}
        )
        assert result is False

    def test_validate_mech_llm_response_no_text(self) -> None:
        """Test _validate_mech_llm_response without text field."""
        behaviour = self._make_behaviour()
        result = EngageTwitterBehaviour._validate_mech_llm_response(
            behaviour, {"tweet_action": {"action_type": "tweet_with_media"}}
        )
        assert result is False

    def test_validate_mech_llm_response_tweet_action_not_dict(self) -> None:
        """Test _validate_mech_llm_response with tweet_action as non-dict."""
        behaviour = self._make_behaviour()
        result = EngageTwitterBehaviour._validate_mech_llm_response(
            behaviour, {"tweet_action": "not_a_dict"}
        )
        assert result is False

    def test_validate_non_mech_tweet_action_valid(self) -> None:
        """Test _validate_non_mech_tweet_action with valid single action."""
        behaviour = self._make_behaviour()
        result = EngageTwitterBehaviour._validate_non_mech_tweet_action(
            behaviour, {"action_type": "tweet", "text": "hello"}
        )
        assert result is True

    def test_validate_non_mech_tweet_action_valid_list(self) -> None:
        """Test _validate_non_mech_tweet_action with valid list of actions."""
        behaviour = self._make_behaviour()
        result = EngageTwitterBehaviour._validate_non_mech_tweet_action(
            behaviour, [{"action_type": "like"}, {"action_type": "reply", "text": "hi"}]
        )
        assert result is True

    def test_validate_non_mech_tweet_action_invalid_media(self) -> None:
        """Test _validate_non_mech_tweet_action rejects tweet_with_media."""
        behaviour = self._make_behaviour()
        result = EngageTwitterBehaviour._validate_non_mech_tweet_action(
            behaviour, {"action_type": "tweet_with_media", "text": "hello"}
        )
        assert result is False

    def test_validate_non_mech_tweet_action_non_dict_item(self) -> None:
        """Test _validate_non_mech_tweet_action with non-dict item in list (continues)."""
        behaviour = self._make_behaviour()
        result = EngageTwitterBehaviour._validate_non_mech_tweet_action(
            behaviour, ["not_a_dict"]  # type: ignore[list-item]
        )
        assert result is True

    def test_validate_non_mech_tool_action_valid(self) -> None:
        """Test _validate_non_mech_tool_action with valid tool action."""
        behaviour = self._make_behaviour()
        result = EngageTwitterBehaviour._validate_non_mech_tool_action(
            behaviour, {"tool_name": "gen_image", "tool_input": "cat"}
        )
        assert result is True

    def test_validate_non_mech_tool_action_not_dict(self) -> None:
        """Test _validate_non_mech_tool_action with non-dict input."""
        behaviour = self._make_behaviour()
        result = EngageTwitterBehaviour._validate_non_mech_tool_action(
            behaviour, "not_a_dict"  # type: ignore[arg-type]
        )
        assert result is False

    def test_validate_non_mech_tool_action_missing_fields(self) -> None:
        """Test _validate_non_mech_tool_action with missing required fields."""
        behaviour = self._make_behaviour()
        result = EngageTwitterBehaviour._validate_non_mech_tool_action(
            behaviour, {"tool_name": "gen_image"}
        )
        assert result is False

    def test_validate_non_mech_llm_response_valid_tweet(self) -> None:
        """Test _validate_non_mech_llm_response with valid tweet_action."""
        behaviour = self._make_behaviour()
        behaviour._validate_non_mech_tweet_action = MagicMock(return_value=True)
        result = EngageTwitterBehaviour._validate_non_mech_llm_response(
            behaviour, {"tweet_action": {"action": "tweet"}}
        )
        assert result is True

    def test_validate_non_mech_llm_response_valid_tool(self) -> None:
        """Test _validate_non_mech_llm_response with valid tool_action."""
        behaviour = self._make_behaviour()
        behaviour._validate_non_mech_tool_action = MagicMock(return_value=True)
        result = EngageTwitterBehaviour._validate_non_mech_llm_response(
            behaviour, {"tool_action": {"tool_name": "t", "tool_input": "i"}}
        )
        assert result is True

    def test_validate_non_mech_llm_response_neither(self) -> None:
        """Test _validate_non_mech_llm_response with neither valid action."""
        behaviour = self._make_behaviour()
        result = EngageTwitterBehaviour._validate_non_mech_llm_response(
            behaviour, {"unknown": "value"}
        )
        assert result is False

    def test_validate_non_mech_llm_response_tweet_action_none(self) -> None:
        """Test _validate_non_mech_llm_response with tweet_action=None then tool_action."""
        behaviour = self._make_behaviour()
        behaviour._validate_non_mech_tool_action = MagicMock(return_value=True)
        result = EngageTwitterBehaviour._validate_non_mech_llm_response(
            behaviour,
            {
                "tweet_action": None,
                "tool_action": {"tool_name": "t", "tool_input": "i"},
            },
        )
        assert result is True

    # ----- _handle_tool_action -----

    def test_handle_tool_action_valid(self) -> None:
        """Test _handle_tool_action with valid tool action."""
        behaviour = self._make_behaviour()
        behaviour._store_agent_action = _make_gen_return(None)

        gen = EngageTwitterBehaviour._handle_tool_action(
            behaviour,
            {"tool_action": {"tool_name": "gen_image", "tool_input": "a cat"}},
        )
        result = _drive_gen_with_side_effects(gen)
        assert result[0] == Event.MECH.value
        assert len(result[2]) == 1

    def test_handle_tool_action_missing_fields(self) -> None:
        """Test _handle_tool_action with missing tool_name/tool_input."""
        behaviour = self._make_behaviour()

        gen = EngageTwitterBehaviour._handle_tool_action(
            behaviour, {"tool_action": {"tool_name": "gen_image"}}
        )
        result = _drive_gen_with_side_effects(gen)
        assert result[0] == Event.ERROR.value

    def test_handle_tool_action_empty_tool_action(self) -> None:
        """Test _handle_tool_action with empty tool_action."""
        behaviour = self._make_behaviour()

        gen = EngageTwitterBehaviour._handle_tool_action(
            behaviour, {"tool_action": None}
        )
        result = _drive_gen_with_side_effects(gen)
        assert result[0] == Event.ERROR.value

    def test_handle_tool_action_no_tool_action_key(self) -> None:
        """Test _handle_tool_action without tool_action key."""
        behaviour = self._make_behaviour()

        gen = EngageTwitterBehaviour._handle_tool_action(behaviour, {})
        result = _drive_gen_with_side_effects(gen)
        assert result[0] == Event.ERROR.value

    # ----- _handle_tweet_actions -----

    def test_handle_tweet_actions_single_action(self) -> None:
        """Test _handle_tweet_actions with single dict action."""
        behaviour = self._make_behaviour()
        behaviour._process_single_interaction = _make_gen_return(None)

        gen = EngageTwitterBehaviour._handle_tweet_actions(
            behaviour,
            {"tweet_action": {"action": "tweet", "text": "hello"}},
            {},
            None,
            "persona",
            [],
        )
        result = _drive_gen_with_side_effects(gen)
        assert result[0] == Event.DONE.value

    def test_handle_tweet_actions_list_of_actions(self) -> None:
        """Test _handle_tweet_actions with list of actions."""
        behaviour = self._make_behaviour()
        behaviour._process_single_interaction = _make_gen_return(None)

        gen = EngageTwitterBehaviour._handle_tweet_actions(
            behaviour,
            {
                "tweet_action": [
                    {"action": "tweet", "text": "hello"},
                    {"action": "like", "selected_tweet_id": "123"},
                ]
            },
            {"123": {"text": "test", "user_name": "user1"}},
            None,
            "persona",
            [],
        )
        result = _drive_gen_with_side_effects(gen)
        assert result[0] == Event.DONE.value

    def test_handle_tweet_actions_string_action(self) -> None:
        """Test _handle_tweet_actions with string action."""
        behaviour = self._make_behaviour()
        behaviour._process_single_interaction = _make_gen_return(None)

        gen = EngageTwitterBehaviour._handle_tweet_actions(
            behaviour,
            {"tweet_action": "tweet"},
            {},
            None,
            "persona",
            [],
        )
        result = _drive_gen_with_side_effects(gen)
        assert result[0] == Event.DONE.value

    # ----- _process_single_interaction -----

    def test_process_single_interaction_invalid_format(self) -> None:
        """Test _process_single_interaction with non-dict interaction."""
        behaviour = self._make_behaviour()
        ctx = InteractionContext(
            pending_tweets={},
            previous_tweets=None,
            persona="persona",
            new_interacted_tweet_ids=[],
        )

        gen = EngageTwitterBehaviour._process_single_interaction(
            behaviour, "not_a_dict", ctx  # type: ignore[arg-type]
        )
        _drive_gen_with_side_effects(gen)

    def test_process_single_interaction_tweet_action(self) -> None:
        """Test _process_single_interaction with tweet action."""
        behaviour = self._make_behaviour()
        behaviour._validate_interaction = MagicMock(return_value=True)
        behaviour.sleep = _make_gen_return(None)
        behaviour._handle_new_tweet = _make_gen_return(None)
        ctx = InteractionContext(
            pending_tweets={},
            previous_tweets=None,
            persona="persona",
            new_interacted_tweet_ids=[],
        )

        gen = EngageTwitterBehaviour._process_single_interaction(
            behaviour,
            {"action": "tweet", "text": "Hello world"},
            ctx,
        )
        _drive_gen_with_side_effects(gen)

    def test_process_single_interaction_tweet_with_media(self) -> None:
        """Test _process_single_interaction with tweet_with_media action."""
        behaviour = self._make_behaviour()
        behaviour._validate_interaction = MagicMock(return_value=True)
        behaviour._get_latest_media_info = _make_gen_return(
            {"type": "image", "path": "/tmp/img.png", "ipfs_hash": "QmHash"}
        )
        behaviour.sleep = _make_gen_return(None)
        behaviour._handle_media_tweet = _make_gen_return(True)
        ctx = InteractionContext(
            pending_tweets={},
            previous_tweets=None,
            persona="persona",
            new_interacted_tweet_ids=[],
        )

        gen = EngageTwitterBehaviour._process_single_interaction(
            behaviour,
            {"action": "tweet_with_media", "text": "With media"},
            ctx,
        )
        _drive_gen_with_side_effects(gen)

    def test_process_single_interaction_tweet_with_media_no_info(self) -> None:
        """Test _process_single_interaction with tweet_with_media but no media info."""
        behaviour = self._make_behaviour()
        behaviour._validate_interaction = MagicMock(return_value=True)
        behaviour._get_latest_media_info = _make_gen_return(None)
        behaviour.sleep = _make_gen_return(None)
        ctx = InteractionContext(
            pending_tweets={},
            previous_tweets=None,
            persona="persona",
            new_interacted_tweet_ids=[],
        )

        gen = EngageTwitterBehaviour._process_single_interaction(
            behaviour,
            {"action": "tweet_with_media", "text": "With media"},
            ctx,
        )
        _drive_gen_with_side_effects(gen)

    def test_process_single_interaction_tweet_with_media_fails(self) -> None:
        """Test _process_single_interaction when media tweet fails."""
        behaviour = self._make_behaviour()
        behaviour._validate_interaction = MagicMock(return_value=True)
        behaviour._get_latest_media_info = _make_gen_return(
            {"type": "image", "path": "/tmp/img.png", "ipfs_hash": "QmHash"}
        )
        behaviour.sleep = _make_gen_return(None)
        behaviour._handle_media_tweet = _make_gen_return(False)
        ctx = InteractionContext(
            pending_tweets={},
            previous_tweets=None,
            persona="persona",
            new_interacted_tweet_ids=[],
        )

        gen = EngageTwitterBehaviour._process_single_interaction(
            behaviour,
            {"action": "tweet_with_media", "text": "With media"},
            ctx,
        )
        _drive_gen_with_side_effects(gen)

    def test_process_single_interaction_like(self) -> None:
        """Test _process_single_interaction with like action."""
        behaviour = self._make_behaviour()
        behaviour._validate_interaction = MagicMock(return_value=True)
        behaviour.sleep = _make_gen_return(None)
        behaviour._handle_tweet_interaction = _make_gen_return(None)
        ctx = InteractionContext(
            pending_tweets={"123": {"text": "t", "user_name": "u"}},
            previous_tweets=None,
            persona="persona",
            new_interacted_tweet_ids=[],
        )

        gen = EngageTwitterBehaviour._process_single_interaction(
            behaviour,
            {"action": "like", "selected_tweet_id": "123"},
            ctx,
        )
        _drive_gen_with_side_effects(gen)

    def test_process_single_interaction_validation_fails(self) -> None:
        """Test _process_single_interaction when validation fails."""
        behaviour = self._make_behaviour()
        behaviour._validate_interaction = MagicMock(return_value=False)
        ctx = InteractionContext(
            pending_tweets={},
            previous_tweets=None,
            persona="persona",
            new_interacted_tweet_ids=[],
        )

        gen = EngageTwitterBehaviour._process_single_interaction(
            behaviour,
            {"action": "like", "selected_tweet_id": "999"},
            ctx,
        )
        _drive_gen_with_side_effects(gen)

    # ----- _handle_media_tweet -----

    def test_handle_media_tweet_success(self) -> None:
        """Test _handle_media_tweet with valid media info."""
        behaviour = self._make_behaviour()
        behaviour._write_kv = _make_gen_return(True)
        behaviour.post_tweet = _make_gen_return(True)

        gen = EngageTwitterBehaviour._handle_media_tweet(
            behaviour,
            "Hello media",
            {"type": "image", "path": "/tmp/img.png", "ipfs_hash": "QmHash"},
        )
        result = _drive_gen_with_side_effects(gen)
        assert result is True

    def test_handle_media_tweet_empty_media_info(self) -> None:
        """Test _handle_media_tweet with empty media_info."""
        behaviour = self._make_behaviour()

        gen = EngageTwitterBehaviour._handle_media_tweet(behaviour, "text", {})
        result = _drive_gen_with_side_effects(gen)
        assert result is False

    def test_handle_media_tweet_missing_path(self) -> None:
        """Test _handle_media_tweet with missing path."""
        behaviour = self._make_behaviour()
        behaviour._write_kv = _make_gen_return(True)

        gen = EngageTwitterBehaviour._handle_media_tweet(
            behaviour, "text", {"type": "image", "ipfs_hash": "QmHash"}
        )
        result = _drive_gen_with_side_effects(gen)
        assert result is False

    def test_handle_media_tweet_missing_type(self) -> None:
        """Test _handle_media_tweet with missing type."""
        behaviour = self._make_behaviour()
        behaviour._write_kv = _make_gen_return(True)

        gen = EngageTwitterBehaviour._handle_media_tweet(
            behaviour, "text", {"path": "/tmp/img.png", "ipfs_hash": "QmHash"}
        )
        result = _drive_gen_with_side_effects(gen)
        assert result is False

    def test_handle_media_tweet_missing_ipfs_hash(self) -> None:
        """Test _handle_media_tweet with missing ipfs_hash."""
        behaviour = self._make_behaviour()
        behaviour._write_kv = _make_gen_return(True)

        gen = EngageTwitterBehaviour._handle_media_tweet(
            behaviour, "text", {"path": "/tmp/img.png", "type": "image"}
        )
        result = _drive_gen_with_side_effects(gen)
        assert result is False

    def test_handle_media_tweet_non_string_path(self) -> None:
        """Test _handle_media_tweet with non-string path."""
        behaviour = self._make_behaviour()
        behaviour._write_kv = _make_gen_return(True)

        gen = EngageTwitterBehaviour._handle_media_tweet(
            behaviour, "text", {"type": "image", "path": 123, "ipfs_hash": "QmHash"}
        )
        result = _drive_gen_with_side_effects(gen)
        assert result is False

    def test_handle_media_tweet_post_fails(self) -> None:
        """Test _handle_media_tweet when post_tweet returns None."""
        behaviour = self._make_behaviour()
        behaviour._write_kv = _make_gen_return(True)
        behaviour.post_tweet = _make_gen_return(None)

        gen = EngageTwitterBehaviour._handle_media_tweet(
            behaviour,
            "Hello media",
            {"type": "image", "path": "/tmp/img.png", "ipfs_hash": "QmHash"},
        )
        result = _drive_gen_with_side_effects(gen)
        assert result is False

    # ----- _validate_interaction -----

    def test_validate_interaction_none_action(self) -> None:
        """Test _validate_interaction with None action."""
        behaviour = self._make_behaviour()
        result = EngageTwitterBehaviour._validate_interaction(
            behaviour, None, None, None, {}
        )
        assert result is False

    def test_validate_interaction_none_string_action(self) -> None:
        """Test _validate_interaction with 'none' string action."""
        behaviour = self._make_behaviour()
        result = EngageTwitterBehaviour._validate_interaction(
            behaviour, "none", None, None, {}
        )
        assert result is False

    def test_validate_interaction_tweet_action(self) -> None:
        """Test _validate_interaction for tweet action (no tweet_id needed)."""
        behaviour = self._make_behaviour()
        result = EngageTwitterBehaviour._validate_interaction(
            behaviour, "tweet", None, None, {}
        )
        assert result is True

    def test_validate_interaction_tweet_with_media(self) -> None:
        """Test _validate_interaction for tweet_with_media action."""
        behaviour = self._make_behaviour()
        result = EngageTwitterBehaviour._validate_interaction(
            behaviour, "tweet_with_media", None, None, {}
        )
        assert result is True

    def test_validate_interaction_like_valid(self) -> None:
        """Test _validate_interaction for like with valid tweet_id."""
        behaviour = self._make_behaviour()
        result = EngageTwitterBehaviour._validate_interaction(
            behaviour, "like", "123", None, {"123": {"text": "t", "user_name": "u"}}
        )
        assert result is True

    def test_validate_interaction_like_invalid_tweet_id(self) -> None:
        """Test _validate_interaction for like with invalid tweet_id."""
        behaviour = self._make_behaviour()
        result = EngageTwitterBehaviour._validate_interaction(
            behaviour, "like", "999", None, {"123": {"text": "t", "user_name": "u"}}
        )
        assert result is False

    def test_validate_interaction_like_no_tweet_id(self) -> None:
        """Test _validate_interaction for like without tweet_id."""
        behaviour = self._make_behaviour()
        result = EngageTwitterBehaviour._validate_interaction(
            behaviour, "like", None, None, {"123": {"text": "t", "user_name": "u"}}
        )
        assert result is False

    def test_validate_interaction_follow_valid(self) -> None:
        """Test _validate_interaction for follow with valid username."""
        behaviour = self._make_behaviour()
        result = EngageTwitterBehaviour._validate_interaction(
            behaviour,
            "follow",
            None,
            "user1",
            {"123": {"text": "t", "user_name": "user1"}},
        )
        assert result is True

    def test_validate_interaction_follow_invalid_username(self) -> None:
        """Test _validate_interaction for follow with invalid username."""
        behaviour = self._make_behaviour()
        result = EngageTwitterBehaviour._validate_interaction(
            behaviour,
            "follow",
            None,
            "unknown",
            {"123": {"text": "t", "user_name": "user1"}},
        )
        assert result is False

    def test_validate_interaction_follow_no_username(self) -> None:
        """Test _validate_interaction for follow without username."""
        behaviour = self._make_behaviour()
        result = EngageTwitterBehaviour._validate_interaction(
            behaviour,
            "follow",
            None,
            None,
            {"123": {"text": "t", "user_name": "user1"}},
        )
        assert result is False

    # ----- _handle_new_tweet -----

    def test_handle_new_tweet_valid(self) -> None:
        """Test _handle_new_tweet with valid tweet text."""
        behaviour = self._make_behaviour()
        behaviour._format_previous_tweets_str = MagicMock(return_value="")
        behaviour.replace_tweet_with_alternative_model = _make_gen_return(None)
        behaviour.post_tweet = _make_gen_return(True)

        gen = EngageTwitterBehaviour._handle_new_tweet(
            behaviour, "Hello world", None, "persona"
        )
        _drive_gen_with_side_effects(gen)

    def test_handle_new_tweet_with_replacement(self) -> None:
        """Test _handle_new_tweet when alternative model provides replacement."""
        behaviour = self._make_behaviour()
        behaviour._format_previous_tweets_str = MagicMock(return_value="")
        behaviour.replace_tweet_with_alternative_model = _make_gen_return(
            "New tweet text"
        )
        behaviour.post_tweet = _make_gen_return(True)

        gen = EngageTwitterBehaviour._handle_new_tweet(
            behaviour, "Original text", None, "persona"
        )
        _drive_gen_with_side_effects(gen)

    def test_handle_new_tweet_too_long(self) -> None:
        """Test _handle_new_tweet when tweet is too long."""
        behaviour = self._make_behaviour()
        behaviour._format_previous_tweets_str = MagicMock(return_value="")
        behaviour.replace_tweet_with_alternative_model = _make_gen_return(None)

        gen = EngageTwitterBehaviour._handle_new_tweet(
            behaviour, "a" * 300, None, "persona"
        )
        _drive_gen_with_side_effects(gen)

    # ----- _handle_tweet_interaction -----

    def test_handle_tweet_interaction_like_success(self) -> None:
        """Test _handle_tweet_interaction for like."""
        behaviour = self._make_behaviour()
        behaviour.like_tweet = _make_gen_return(True)
        new_ids: list[int] = []

        gen = EngageTwitterBehaviour._handle_tweet_interaction(
            behaviour,
            "like",
            "123",
            None,
            None,
            {"123": {"text": "t", "user_name": "u"}},
            new_ids,
        )
        _drive_gen_with_side_effects(gen)
        assert 123 in new_ids

    def test_handle_tweet_interaction_retweet_success(self) -> None:
        """Test _handle_tweet_interaction for retweet."""
        behaviour = self._make_behaviour()
        behaviour.retweet = _make_gen_return(True)
        new_ids: list[int] = []

        gen = EngageTwitterBehaviour._handle_tweet_interaction(
            behaviour,
            "retweet",
            "456",
            None,
            None,
            {"456": {"text": "t", "user_name": "u"}},
            new_ids,
        )
        _drive_gen_with_side_effects(gen)
        assert 456 in new_ids

    def test_handle_tweet_interaction_reply_success(self) -> None:
        """Test _handle_tweet_interaction for reply."""
        behaviour = self._make_behaviour()
        behaviour.respond_tweet = _make_gen_return(True)
        new_ids: list[int] = []

        gen = EngageTwitterBehaviour._handle_tweet_interaction(
            behaviour,
            "reply",
            "789",
            "reply text",
            None,
            {"789": {"text": "t", "user_name": "u"}},
            new_ids,
        )
        _drive_gen_with_side_effects(gen)
        assert 789 in new_ids

    def test_handle_tweet_interaction_quote_success(self) -> None:
        """Test _handle_tweet_interaction for quote."""
        behaviour = self._make_behaviour()
        behaviour.respond_tweet = _make_gen_return(True)
        new_ids: list[int] = []

        gen = EngageTwitterBehaviour._handle_tweet_interaction(
            behaviour,
            "quote",
            "101",
            "quote text",
            None,
            {"101": {"text": "t", "user_name": "quoter"}},
            new_ids,
        )
        _drive_gen_with_side_effects(gen)
        assert 101 in new_ids

    def test_handle_tweet_interaction_quote_no_username(self) -> None:
        """Test _handle_tweet_interaction for quote with missing user_name."""
        behaviour = self._make_behaviour()
        new_ids: list[int] = []

        gen = EngageTwitterBehaviour._handle_tweet_interaction(
            behaviour,
            "quote",
            "101",
            "quote text",
            None,
            {"101": {"text": "t"}},
            new_ids,  # no user_name key
        )
        _drive_gen_with_side_effects(gen)
        assert not new_ids

    def test_handle_tweet_interaction_follow_success(self) -> None:
        """Test _handle_tweet_interaction for follow."""
        behaviour = self._make_behaviour()
        behaviour.follow_user = _make_gen_return(True)
        new_ids: list[int] = []

        gen = EngageTwitterBehaviour._handle_tweet_interaction(
            behaviour,
            "follow",
            None,
            None,
            "user1",
            {"123": {"text": "t", "user_name": "user1"}},
            new_ids,
        )
        _drive_gen_with_side_effects(gen)
        # Follow doesn't add to new_interacted_tweet_ids
        assert not new_ids

    def test_handle_tweet_interaction_follow_no_username(self) -> None:
        """Test _handle_tweet_interaction for follow with no user_name."""
        behaviour = self._make_behaviour()
        new_ids: list[int] = []

        gen = EngageTwitterBehaviour._handle_tweet_interaction(
            behaviour,
            "follow",
            None,
            None,
            None,
            {},
            new_ids,
        )
        _drive_gen_with_side_effects(gen)
        assert not new_ids

    def test_handle_tweet_interaction_no_tweet_id_for_like(self) -> None:
        """Test _handle_tweet_interaction for like with no tweet_id."""
        behaviour = self._make_behaviour()
        new_ids: list[int] = []

        gen = EngageTwitterBehaviour._handle_tweet_interaction(
            behaviour,
            "like",
            None,
            None,
            None,
            {"123": {"text": "t", "user_name": "u"}},
            new_ids,
        )
        _drive_gen_with_side_effects(gen)
        assert not new_ids

    def test_handle_tweet_interaction_tweet_not_in_pending(self) -> None:
        """Test _handle_tweet_interaction when tweet_id not in pending_tweets."""
        behaviour = self._make_behaviour()
        new_ids: list[int] = []

        gen = EngageTwitterBehaviour._handle_tweet_interaction(
            behaviour,
            "like",
            "999",
            None,
            None,
            {"123": {"text": "t", "user_name": "u"}},
            new_ids,
        )
        _drive_gen_with_side_effects(gen)
        assert not new_ids

    def test_handle_tweet_interaction_text_too_long(self) -> None:
        """Test _handle_tweet_interaction with text that is too long."""
        behaviour = self._make_behaviour()
        new_ids: list[int] = []

        gen = EngageTwitterBehaviour._handle_tweet_interaction(
            behaviour,
            "reply",
            "123",
            "a" * 300,
            None,
            {"123": {"text": "t", "user_name": "u"}},
            new_ids,
        )
        _drive_gen_with_side_effects(gen)
        assert not new_ids

    def test_handle_tweet_interaction_like_fails(self) -> None:
        """Test _handle_tweet_interaction for like that fails."""
        behaviour = self._make_behaviour()
        behaviour.like_tweet = _make_gen_return(False)
        new_ids: list[int] = []

        gen = EngageTwitterBehaviour._handle_tweet_interaction(
            behaviour,
            "like",
            "123",
            None,
            None,
            {"123": {"text": "t", "user_name": "u"}},
            new_ids,
        )
        _drive_gen_with_side_effects(gen)
        assert not new_ids

    # ----- generate_mech_tool_info -----

    def test_generate_mech_tool_info(self) -> None:
        """Test generate_mech_tool_info."""
        behaviour = self._make_behaviour()
        behaviour.params.tools_for_mech = {
            "gen_image": "Generate an image",
            "gen_video": "Generate a video",
        }
        result = EngageTwitterBehaviour.generate_mech_tool_info(behaviour)
        assert "gen_image" in result
        assert "gen_video" in result

    # ----- get_agent_handles -----

    def test_get_agent_handles_has_agents(self) -> None:
        """Test get_agent_handles when agents_fun_db has active agents."""
        behaviour = self._make_behaviour()
        mock_agents = [MagicMock(), MagicMock()]
        behaviour.context.agents_fun_db.get_active_agents.return_value = mock_agents

        # get_agent_handles has Generator type hint but no yield, so it's a
        # regular function that returns the list directly
        result = EngageTwitterBehaviour.get_agent_handles(behaviour)
        assert len(result) == 2  # type: ignore[arg-type]

    def test_get_agent_handles_no_agents(self) -> None:
        """Test get_agent_handles when no active agents."""
        behaviour = self._make_behaviour()
        behaviour.context.agents_fun_db.get_active_agents.return_value = []

        result = EngageTwitterBehaviour.get_agent_handles(behaviour)
        assert not result

    # ----- _get_shuffled_twitter_actions -----

    def test_get_shuffled_twitter_actions(self) -> None:
        """Test _get_shuffled_twitter_actions returns all actions."""
        result = EngageTwitterBehaviour._get_shuffled_twitter_actions()
        assert "Tweet" in result
        assert "Reply" in result
        assert "Quote" in result
        assert "Like" in result
        assert "Retweet" in result
        assert "Follow" in result


# ============================================================================
# Tests for ActionTweetBehaviour
# ============================================================================


class TestActionTweetBehaviour:
    """Tests for ActionTweetBehaviour."""

    def _make_behaviour(self, token_action: Any = None, **kwargs: Any) -> MagicMock:
        """Create a mock ActionTweetBehaviour."""
        behaviour = MagicMock(spec=ActionTweetBehaviour)
        behaviour.context = make_mock_context()
        behaviour.params = make_mock_params()
        behaviour.synchronized_data = make_mock_synchronized_data(
            token_action=token_action or {"tweet": "Hello world"}
        )
        behaviour.behaviour_id = "action_tweet"
        behaviour.matching_round = MagicMock()
        behaviour.matching_round.auto_round_id.return_value = "action_tweet_round"
        return behaviour

    # ----- async_act -----

    def test_async_act(self) -> None:
        """Test ActionTweetBehaviour.async_act."""
        behaviour = self._make_behaviour()
        behaviour.get_event = _make_gen_return(Event.DONE.value)
        behaviour.send_a2a_transaction = _make_gen_return(None)
        behaviour.wait_until_round_end = _make_gen_return(None)
        behaviour.set_done = MagicMock()

        gen = ActionTweetBehaviour.async_act(behaviour)
        _drive_gen_with_side_effects(gen)
        behaviour.set_done.assert_called_once()

    # ----- get_event -----

    def test_get_event_missing_tweet(self) -> None:
        """Test get_event when tweet is missing."""
        behaviour = self._make_behaviour(token_action={"tweet": ""})

        gen = ActionTweetBehaviour.get_event(behaviour)
        result = _drive_gen_with_side_effects(gen)
        assert result == Event.MISSING_TWEET.value

    def test_get_event_missing_tweet_key(self) -> None:
        """Test get_event when tweet key is missing (falsy)."""
        behaviour = self._make_behaviour(token_action={"tweet": None})

        gen = ActionTweetBehaviour.get_event(behaviour)
        result = _drive_gen_with_side_effects(gen)
        assert result == Event.MISSING_TWEET.value

    def test_get_event_post_success(self) -> None:
        """Test get_event when post_tweet succeeds."""
        behaviour = self._make_behaviour(token_action={"tweet": "Buy my token!"})
        behaviour.post_tweet = _make_gen_return(True)
        behaviour._read_json_from_kv = _make_gen_return(
            {
                "tweet_action": [
                    {"action_type": "tweet", "action_data": {"tweet_id": "12345"}}
                ]
            }
        )
        behaviour._store_agent_action = _make_gen_return(None)

        gen = ActionTweetBehaviour.get_event(behaviour)
        result = _drive_gen_with_side_effects(gen)
        assert result == Event.DONE.value

    def test_get_event_post_success_non_tweet_action(self) -> None:
        """Test get_event when latest action is not 'tweet' type."""
        behaviour = self._make_behaviour(token_action={"tweet": "Buy my token!"})
        behaviour.post_tweet = _make_gen_return(True)
        behaviour._read_json_from_kv = _make_gen_return(
            {
                "tweet_action": [
                    {"action_type": "retweet", "action_data": {"tweet_id": "12345"}}
                ]
            }
        )
        behaviour._store_agent_action = _make_gen_return(None)

        gen = ActionTweetBehaviour.get_event(behaviour)
        result = _drive_gen_with_side_effects(gen)
        assert result == Event.DONE.value

    def test_get_event_post_fails(self) -> None:
        """Test get_event when post_tweet fails (returns None/falsy)."""
        behaviour = self._make_behaviour(token_action={"tweet": "Buy my token!"})
        behaviour.post_tweet = _make_gen_return(None)
        behaviour._read_json_from_kv = _make_gen_return({})
        behaviour._store_agent_action = _make_gen_return(None)

        gen = ActionTweetBehaviour.get_event(behaviour)
        result = _drive_gen_with_side_effects(gen)
        assert result == Event.ERROR.value


# ============================================================================
# Additional branch coverage tests
# ============================================================================


class TestAdditionalBranches:
    """Tests for covering remaining partial branches."""

    def _make_engage_behaviour(self, **sync_kwargs: Any) -> MagicMock:
        """Create a mock EngageTwitterBehaviour."""
        behaviour = MagicMock(spec=EngageTwitterBehaviour)
        behaviour.context = make_mock_context()
        behaviour.params = make_mock_params()
        behaviour.synchronized_data = make_mock_synchronized_data(**sync_kwargs)
        behaviour.behaviour_id = "engage_twitter"
        behaviour.matching_round = MagicMock()
        return behaviour

    def test_get_event_error_event(self) -> None:
        """Test get_event when interact_twitter returns ERROR (not DONE, not MECH)."""
        behaviour = self._make_engage_behaviour(mech_for_twitter=False)
        behaviour.params.skip_engagement = False
        behaviour._handle_regular_engagement = _make_gen_return(({}, []))
        behaviour.interact_twitter = _make_gen_return((Event.ERROR.value, [], []))

        gen = EngageTwitterBehaviour.get_event(behaviour)
        result = _drive_gen_with_side_effects(gen)
        assert result[0] == Event.ERROR.value

    def test_handle_tweet_interaction_unknown_action(self) -> None:
        """Test _handle_tweet_interaction with unknown action type that falls through all elif."""
        behaviour = self._make_engage_behaviour()
        new_ids: list[int] = []

        gen = EngageTwitterBehaviour._handle_tweet_interaction(
            behaviour,
            "unknown_action",
            "123",
            "text",
            None,
            {"123": {"text": "t", "user_name": "u"}},
            new_ids,
        )
        _drive_gen_with_side_effects(gen)
        # success remains False, no ids appended
        assert not new_ids

    def test_format_previous_tweets_str_neither_post_nor_dict(self) -> None:
        """Test _format_previous_tweets_str with items that are neither TwitterPost nor dict."""
        # Items that are not TwitterPost or dict don't produce output
        posts: list = [42, "string_item"]
        result = BaseTweetBehaviour._format_previous_tweets_str(posts)  # type: ignore[arg-type]
        # Since neither 42 nor "string_item" match TwitterPost or dict isinstance checks,
        # they produce no formatted tweets
        assert result == ""

    def test_prepare_standard_prompt_data_posts_neither_type(self) -> None:
        """Test _prepare_standard_prompt_data with posts that are neither TwitterPost nor dict."""
        behaviour = self._make_engage_behaviour(is_staking_kpi_met=True)
        # Use an object that is neither TwitterPost nor dict
        mock_item = MagicMock()
        # Make isinstance checks fail for TwitterPost and dict
        behaviour.context.agents_fun_db.my_agent.posts = [mock_item]
        behaviour._format_previous_tweets_str = MagicMock(return_value="")
        behaviour.get_sync_time_str = MagicMock(return_value="2024-01-01 12:00:00")
        behaviour.generate_mech_tool_info = MagicMock(return_value="")
        behaviour.get_latest_agent_actions = _make_gen_return([])
        behaviour._get_shuffled_twitter_actions = MagicMock(return_value="- Tweet")
        behaviour._save_standard_kv_data = _make_gen_return(None)
        behaviour._read_value_from_kv = _make_gen_return(None)

        gen = EngageTwitterBehaviour._prepare_standard_prompt_data(
            behaviour, {}, "persona"
        )
        result = _drive_gen_with_side_effects(gen)
        assert isinstance(result[0], str)
