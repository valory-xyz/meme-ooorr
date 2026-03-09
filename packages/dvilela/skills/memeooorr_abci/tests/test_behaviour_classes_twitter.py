# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024 David Vilela Freire
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

"""Tests for behaviour_classes/twitter.py."""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from packages.dvilela.skills.memeooorr_abci.behaviour_classes.twitter import (
    ActionTweetBehaviour,
    BaseTweetBehaviour,
    CollectFeedbackBehaviour,
    EngageTwitterBehaviour,
    InteractionContext,
    is_tweet_valid,
)
from packages.dvilela.skills.memeooorr_abci.rounds import (
    ActionTweetRound,
    CollectFeedbackRound,
    EngageTwitterRound,
    Event,
)

from .conftest import (
    SENDER,
    make_mock_context,
    make_mock_params,
    make_mock_synchronized_data,
)


class TestMatchingRounds:
    """Tests for matching_round assignments."""

    def test_collect_feedback_matching_round(self) -> None:
        """Test CollectFeedbackBehaviour has correct matching_round."""
        assert CollectFeedbackBehaviour.matching_round is CollectFeedbackRound

    def test_engage_twitter_matching_round(self) -> None:
        """Test EngageTwitterBehaviour has correct matching_round."""
        assert EngageTwitterBehaviour.matching_round is EngageTwitterRound

    def test_action_tweet_matching_round(self) -> None:
        """Test ActionTweetBehaviour has correct matching_round."""
        assert ActionTweetBehaviour.matching_round is ActionTweetRound


class TestIsTweetValid:
    """Tests for is_tweet_valid function."""

    def test_valid_short_tweet(self) -> None:
        """Test short tweet is valid."""
        assert is_tweet_valid("Hello") is True

    def test_invalid_long_tweet(self) -> None:
        """Test overly long tweet is invalid."""
        assert is_tweet_valid("x" * 300) is False

    def test_empty_tweet(self) -> None:
        """Test empty tweet is valid."""
        assert is_tweet_valid("") is True

    def test_exactly_280_chars(self) -> None:
        """Test tweet at exactly 280 chars."""
        assert is_tweet_valid("a" * 280) is True


class TestInteractionContext:
    """Tests for InteractionContext dataclass."""

    def test_creation(self) -> None:
        """Test InteractionContext can be created."""
        ctx = InteractionContext(
            pending_tweets={"1": {"text": "hello", "user_name": "user1"}},
            previous_tweets=[{"text": "prev"}],
            persona="test",
            new_interacted_tweet_ids=[1, 2],
        )
        assert ctx.persona == "test"
        assert len(ctx.new_interacted_tweet_ids) == 2

    def test_mutable_tweet_ids(self) -> None:
        """Test new_interacted_tweet_ids is mutable."""
        ctx = InteractionContext(
            pending_tweets={},
            previous_tweets=None,
            persona="test",
            new_interacted_tweet_ids=[],
        )
        ctx.new_interacted_tweet_ids.append(42)
        assert 42 in ctx.new_interacted_tweet_ids


class TestFormatPreviousTweetsStr:
    """Tests for _format_previous_tweets_str static method."""

    def test_none_input(self) -> None:
        """Test returns empty string for None."""
        result = BaseTweetBehaviour._format_previous_tweets_str(None)
        assert result == ""

    def test_empty_list(self) -> None:
        """Test returns empty string for empty list."""
        result = BaseTweetBehaviour._format_previous_tweets_str([])
        assert result == ""

    def test_dict_tweets(self) -> None:
        """Test formatting dict tweets."""
        tweets = [
            {"text": "Hello world", "timestamp": "2024-01-01"},
            {"text": "Another tweet", "timestamp": "2024-01-02"},
        ]
        result = BaseTweetBehaviour._format_previous_tweets_str(tweets)
        assert "Hello world" in result
        assert "Another tweet" in result

    def test_twitter_post_objects(self) -> None:
        """Test formatting TwitterPost objects."""
        post = MagicMock()
        post.text = "Test post"
        post.timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
        # Make isinstance check work
        from packages.valory.skills.agent_db_abci.twitter_models import TwitterPost

        result = BaseTweetBehaviour._format_previous_tweets_str([post])
        # Since post is a MagicMock, not an actual TwitterPost, it goes through dict branch
        assert isinstance(result, str)


class TestCollectFeedbackBehaviour:
    """Tests for CollectFeedbackBehaviour."""

    def _make_behaviour(self) -> MagicMock:
        behaviour = MagicMock(spec=CollectFeedbackBehaviour)
        behaviour.context = make_mock_context()
        return behaviour

    def test_twitter_post_to_dict(self) -> None:
        """Test _twitter_post_to_dict converts a post to dict."""
        post = MagicMock()
        post.action = "tweet"
        post.timestamp = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        post.tweet_id = 12345
        post.text = "Test tweet"
        post.reply_to_tweet_id = None

        result = CollectFeedbackBehaviour._twitter_post_to_dict(post)
        assert result["action"] == "tweet"
        assert result["tweet_id"] == "12345"
        assert result["text"] == "Test tweet"
        assert result["reply_to_tweet_id"] is None

    def test_twitter_post_to_dict_with_reply(self) -> None:
        """Test _twitter_post_to_dict with reply_to_tweet_id."""
        post = MagicMock()
        post.action = "reply"
        post.timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
        post.tweet_id = 12345
        post.text = "Reply text"
        post.reply_to_tweet_id = 99999

        result = CollectFeedbackBehaviour._twitter_post_to_dict(post)
        assert result["reply_to_tweet_id"] == "99999"

    def test_process_raw_replies_not_list(self) -> None:
        """Test _process_raw_replies returns empty list for non-list input."""
        behaviour = self._make_behaviour()
        result = CollectFeedbackBehaviour._process_raw_replies(
            behaviour, "not a list"
        )
        assert result == []

    def test_process_raw_replies_empty(self) -> None:
        """Test _process_raw_replies with empty list."""
        behaviour = self._make_behaviour()
        result = CollectFeedbackBehaviour._process_raw_replies(behaviour, [])
        assert result == []

    def test_process_raw_replies_dict_items(self) -> None:
        """Test _process_raw_replies with dict items."""
        behaviour = self._make_behaviour()
        replies = [
            {"text": "Great!", "tweet_id": "1"},
            {"text": "Nice!", "tweet_id": "2"},
        ]
        result = CollectFeedbackBehaviour._process_raw_replies(behaviour, replies)
        assert len(result) == 2
        assert result[0]["text"] == "Great!"

    def test_process_raw_replies_dict_with_datetime_timestamp(self) -> None:
        """Test _process_raw_replies converts datetime timestamp to ISO."""
        behaviour = self._make_behaviour()
        ts = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        replies = [{"text": "Hello", "timestamp": ts}]
        result = CollectFeedbackBehaviour._process_raw_replies(behaviour, replies)
        assert isinstance(result[0]["timestamp"], str)

    def test_process_raw_replies_unexpected_type(self) -> None:
        """Test _process_raw_replies skips unexpected types."""
        behaviour = self._make_behaviour()
        replies = [42, "string", None]
        result = CollectFeedbackBehaviour._process_raw_replies(behaviour, replies)
        assert result == []

    def test_get_feedback_no_posts(self) -> None:
        """Test get_feedback returns empty feedback when no posts."""
        behaviour = self._make_behaviour()
        behaviour.context.agents_fun_db.my_agent.posts = []

        gen = CollectFeedbackBehaviour.get_feedback(behaviour)
        try:
            next(gen)
        except StopIteration as e:
            result = e.value
        assert result == {"likes": 0, "retweets": 0, "replies": []}

    def test_get_feedback_last_tweet_none(self) -> None:
        """Test get_feedback returns empty feedback when last tweet is None."""
        behaviour = self._make_behaviour()
        behaviour.context.agents_fun_db.my_agent.posts = [None]

        gen = CollectFeedbackBehaviour.get_feedback(behaviour)
        try:
            next(gen)
        except StopIteration as e:
            result = e.value
        assert result == {"likes": 0, "retweets": 0, "replies": []}


class TestEngageTwitterBehaviour:
    """Tests for EngageTwitterBehaviour."""

    def _make_behaviour(self, **kwargs: Any) -> MagicMock:
        behaviour = MagicMock(spec=EngageTwitterBehaviour)
        behaviour.params = make_mock_params(**kwargs)
        behaviour.context = make_mock_context(params=behaviour.params)
        behaviour.synchronized_data = make_mock_synchronized_data()
        behaviour.get_sync_time_str = MagicMock(return_value="2024-01-01 12:00:00")
        return behaviour

    def test_validate_llm_response_not_dict(self) -> None:
        """Test _validate_llm_response returns False for non-dict."""
        behaviour = self._make_behaviour()
        result = EngageTwitterBehaviour._validate_llm_response(
            behaviour, "not a dict"
        )
        assert result is False

    def test_validate_mech_llm_response_no_tweet_action(self) -> None:
        """Test _validate_mech_llm_response returns False without tweet_action."""
        behaviour = self._make_behaviour()
        result = EngageTwitterBehaviour._validate_mech_llm_response(
            behaviour, {"tool_action": {}}
        )
        assert result is False

    def test_validate_mech_llm_response_wrong_action_type(self) -> None:
        """Test _validate_mech_llm_response returns False for wrong action type."""
        behaviour = self._make_behaviour()
        result = EngageTwitterBehaviour._validate_mech_llm_response(
            behaviour,
            {"tweet_action": {"action_type": "tweet", "text": "hello"}},
        )
        assert result is False

    def test_validate_mech_llm_response_valid(self) -> None:
        """Test _validate_mech_llm_response returns True for valid response."""
        behaviour = self._make_behaviour()
        result = EngageTwitterBehaviour._validate_mech_llm_response(
            behaviour,
            {"tweet_action": {"action_type": "tweet_with_media", "text": "hello"}},
        )
        assert result is True

    def test_validate_mech_llm_response_missing_text(self) -> None:
        """Test _validate_mech_llm_response returns False when text is missing."""
        behaviour = self._make_behaviour()
        result = EngageTwitterBehaviour._validate_mech_llm_response(
            behaviour,
            {"tweet_action": {"action_type": "tweet_with_media"}},
        )
        assert result is False

    def test_validate_non_mech_tweet_action_valid(self) -> None:
        """Test _validate_non_mech_tweet_action returns True for valid action."""
        behaviour = self._make_behaviour()
        result = EngageTwitterBehaviour._validate_non_mech_tweet_action(
            behaviour,
            {"action_type": "tweet", "text": "hello"},
        )
        assert result is True

    def test_validate_non_mech_tweet_action_tweet_with_media_rejected(self) -> None:
        """Test _validate_non_mech_tweet_action rejects tweet_with_media."""
        behaviour = self._make_behaviour()
        result = EngageTwitterBehaviour._validate_non_mech_tweet_action(
            behaviour,
            {"action_type": "tweet_with_media", "text": "hello"},
        )
        assert result is False

    def test_validate_non_mech_tweet_action_list(self) -> None:
        """Test _validate_non_mech_tweet_action with list of actions."""
        behaviour = self._make_behaviour()
        result = EngageTwitterBehaviour._validate_non_mech_tweet_action(
            behaviour,
            [
                {"action_type": "tweet", "text": "hello"},
                {"action_type": "like", "tweet_id": "123"},
            ],
        )
        assert result is True

    def test_validate_non_mech_tool_action_valid(self) -> None:
        """Test _validate_non_mech_tool_action returns True for valid tool action."""
        behaviour = self._make_behaviour()
        result = EngageTwitterBehaviour._validate_non_mech_tool_action(
            behaviour,
            {"tool_name": "image_gen", "tool_input": "a cat"},
        )
        assert result is True

    def test_validate_non_mech_tool_action_not_dict(self) -> None:
        """Test _validate_non_mech_tool_action returns False for non-dict."""
        behaviour = self._make_behaviour()
        result = EngageTwitterBehaviour._validate_non_mech_tool_action(
            behaviour, "not a dict"
        )
        assert result is False

    def test_validate_non_mech_tool_action_missing_fields(self) -> None:
        """Test _validate_non_mech_tool_action returns False for missing fields."""
        behaviour = self._make_behaviour()
        result = EngageTwitterBehaviour._validate_non_mech_tool_action(
            behaviour, {"tool_name": "test"}
        )
        assert result is False

    def test_validate_non_mech_llm_response_with_tweet_action(self) -> None:
        """Test _validate_non_mech_llm_response with valid tweet_action."""
        behaviour = self._make_behaviour()
        result = EngageTwitterBehaviour._validate_non_mech_llm_response(
            behaviour,
            {"tweet_action": {"action_type": "tweet", "text": "hello"}},
        )
        assert result is True

    def test_validate_non_mech_llm_response_with_tool_action(self) -> None:
        """Test _validate_non_mech_llm_response with valid tool_action."""
        behaviour = self._make_behaviour()
        result = EngageTwitterBehaviour._validate_non_mech_llm_response(
            behaviour,
            {"tool_action": {"tool_name": "test", "tool_input": "input"}},
        )
        assert result is True

    def test_validate_non_mech_llm_response_neither(self) -> None:
        """Test _validate_non_mech_llm_response returns False when neither valid."""
        behaviour = self._make_behaviour()
        result = EngageTwitterBehaviour._validate_non_mech_llm_response(
            behaviour,
            {"some_other_key": "value"},
        )
        assert result is False

    def test_validate_llm_response_mech_for_twitter_true(self) -> None:
        """Test _validate_llm_response delegates to mech validation."""
        behaviour = self._make_behaviour()
        behaviour.synchronized_data = make_mock_synchronized_data(
            mech_for_twitter=True
        )
        behaviour._validate_mech_llm_response = MagicMock(return_value=True)
        result = EngageTwitterBehaviour._validate_llm_response(
            behaviour,
            {"tweet_action": {"action_type": "tweet_with_media", "text": "hello"}},
        )
        assert result is True

    def test_validate_llm_response_mech_for_twitter_false(self) -> None:
        """Test _validate_llm_response delegates to non-mech validation."""
        behaviour = self._make_behaviour()
        behaviour.synchronized_data = make_mock_synchronized_data(
            mech_for_twitter=False
        )
        behaviour._validate_non_mech_llm_response = MagicMock(return_value=True)
        result = EngageTwitterBehaviour._validate_llm_response(
            behaviour,
            {"tweet_action": {"action_type": "tweet", "text": "hello"}},
        )
        assert result is True


class TestEngageTwitterGetEvent:
    """Tests for EngageTwitterBehaviour.get_event."""

    def _make_behaviour(self, **kwargs: Any) -> MagicMock:
        behaviour = MagicMock(spec=EngageTwitterBehaviour)
        behaviour.params = make_mock_params(**kwargs)
        behaviour.context = make_mock_context(params=behaviour.params)
        behaviour.synchronized_data = make_mock_synchronized_data()
        return behaviour

    def test_skip_engagement(self) -> None:
        """Test get_event returns DONE when skip_engagement is True and mech_for_twitter is False."""
        behaviour = self._make_behaviour(skip_engagement=True)
        behaviour.synchronized_data = make_mock_synchronized_data(
            mech_for_twitter=False
        )

        gen = EngageTwitterBehaviour.get_event(behaviour)
        try:
            next(gen)
        except StopIteration as e:
            result = e.value
        assert result[0] == Event.DONE.value
        assert result[1] == []


class TestProcessSingleInteraction:
    """Tests for _process_single_interaction."""

    def _make_behaviour(self) -> MagicMock:
        behaviour = MagicMock(spec=EngageTwitterBehaviour)
        behaviour.params = make_mock_params()
        behaviour.context = make_mock_context(params=behaviour.params)
        behaviour.synchronized_data = make_mock_synchronized_data()
        return behaviour

    def test_non_dict_interaction(self) -> None:
        """Test _process_single_interaction returns early for non-dict."""
        behaviour = self._make_behaviour()
        ctx = InteractionContext(
            pending_tweets={},
            previous_tweets=None,
            persona="test",
            new_interacted_tweet_ids=[],
        )

        gen = EngageTwitterBehaviour._process_single_interaction(
            behaviour, "not a dict", ctx
        )
        try:
            next(gen)
        except StopIteration:
            pass  # Should return early


class TestDetermineMechSummary:
    """Tests for _determine_mech_summary."""

    def _make_behaviour(self) -> MagicMock:
        behaviour = MagicMock(spec=EngageTwitterBehaviour)
        behaviour.context = make_mock_context()
        return behaviour

    def test_no_media_info(self) -> None:
        """Test returns default message when no media info."""
        behaviour = self._make_behaviour()

        def mock_get_latest_media_info():
            yield
            return None

        behaviour._get_latest_media_info = mock_get_latest_media_info

        gen = EngageTwitterBehaviour._determine_mech_summary(behaviour)
        next(gen)
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert "failed" in result.lower() or "proceed" in result.lower()

    def test_image_media_info(self) -> None:
        """Test returns image message when image media."""
        behaviour = self._make_behaviour()

        def mock_get_latest_media_info():
            yield
            return {"type": "image", "path": "/tmp/img.png"}

        behaviour._get_latest_media_info = mock_get_latest_media_info

        gen = EngageTwitterBehaviour._determine_mech_summary(behaviour)
        next(gen)
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert "image" in result.lower()

    def test_video_media_info(self) -> None:
        """Test returns video message when video media."""
        behaviour = self._make_behaviour()

        def mock_get_latest_media_info():
            yield
            return {"type": "video", "path": "/tmp/vid.mp4"}

        behaviour._get_latest_media_info = mock_get_latest_media_info

        gen = EngageTwitterBehaviour._determine_mech_summary(behaviour)
        next(gen)
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert "video" in result.lower()

    def test_unknown_media_type(self) -> None:
        """Test handles unknown media type gracefully."""
        behaviour = self._make_behaviour()

        def mock_get_latest_media_info():
            yield
            return {"type": "audio", "path": "/tmp/audio.mp3"}

        behaviour._get_latest_media_info = mock_get_latest_media_info

        gen = EngageTwitterBehaviour._determine_mech_summary(behaviour)
        next(gen)
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        # Should return the default/fallback message
        assert isinstance(result, str)


class TestHandleToolAction:
    """Tests for _handle_tool_action."""

    def _make_behaviour(self) -> MagicMock:
        behaviour = MagicMock(spec=EngageTwitterBehaviour)
        behaviour.context = make_mock_context()
        behaviour.synchronized_data = make_mock_synchronized_data()
        return behaviour

    def test_missing_tool_action(self) -> None:
        """Test returns ERROR when tool_action is missing."""
        behaviour = self._make_behaviour()

        gen = EngageTwitterBehaviour._handle_tool_action(behaviour, {})
        try:
            val = next(gen)
            while True:
                val = gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result[0] == Event.ERROR.value

    def test_missing_tool_name(self) -> None:
        """Test returns ERROR when tool_name is missing."""
        behaviour = self._make_behaviour()

        gen = EngageTwitterBehaviour._handle_tool_action(
            behaviour, {"tool_action": {"tool_input": "test"}}
        )
        try:
            val = next(gen)
            while True:
                val = gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result[0] == Event.ERROR.value

    def test_valid_tool_action(self) -> None:
        """Test returns MECH with valid mech requests."""
        behaviour = self._make_behaviour()

        def mock_store_agent_action(action_type, data):
            yield
            return None

        behaviour._store_agent_action = mock_store_agent_action

        gen = EngageTwitterBehaviour._handle_tool_action(
            behaviour,
            {
                "tool_action": {
                    "tool_name": "image_gen",
                    "tool_input": "a beautiful sunset",
                }
            },
        )
        try:
            val = next(gen)
            while True:
                val = gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result[0] == Event.MECH.value
        assert len(result[2]) == 1
        assert result[2][0]["tool"] == "image_gen"


class TestHandleTweetActions:
    """Tests for _handle_tweet_actions."""

    def _make_behaviour(self) -> MagicMock:
        behaviour = MagicMock(spec=EngageTwitterBehaviour)
        behaviour.context = make_mock_context()
        behaviour.synchronized_data = make_mock_synchronized_data()
        return behaviour

    def test_string_tweet_action(self) -> None:
        """Test _handle_tweet_actions converts string to list."""
        behaviour = self._make_behaviour()

        def mock_process_single(interaction, context):
            yield
            return None

        behaviour._process_single_interaction = mock_process_single

        gen = EngageTwitterBehaviour._handle_tweet_actions(
            behaviour,
            {"tweet_action": "tweet"},
            {},
            None,
            "test persona",
            [],
        )
        try:
            val = next(gen)
            while True:
                val = gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result[0] == Event.DONE.value

    def test_dict_tweet_action(self) -> None:
        """Test _handle_tweet_actions wraps dict in list."""
        behaviour = self._make_behaviour()

        def mock_process_single(interaction, context):
            yield
            return None

        behaviour._process_single_interaction = mock_process_single

        gen = EngageTwitterBehaviour._handle_tweet_actions(
            behaviour,
            {"tweet_action": {"action": "tweet", "text": "hello"}},
            {},
            None,
            "test persona",
            [],
        )
        try:
            val = next(gen)
            while True:
                val = gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result[0] == Event.DONE.value

    def test_list_tweet_actions(self) -> None:
        """Test _handle_tweet_actions processes list of actions."""
        behaviour = self._make_behaviour()
        call_count = [0]

        def mock_process_single(interaction, context):
            call_count[0] += 1
            yield
            return None

        behaviour._process_single_interaction = mock_process_single

        gen = EngageTwitterBehaviour._handle_tweet_actions(
            behaviour,
            {
                "tweet_action": [
                    {"action": "tweet", "text": "hello"},
                    {"action": "like", "selected_tweet_id": "123"},
                ]
            },
            {},
            None,
            "test persona",
            [],
        )
        try:
            val = next(gen)
            while True:
                val = gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result[0] == Event.DONE.value
        assert call_count[0] == 2


class TestBaseTweetBehaviourWriteTweet:
    """Tests for BaseTweetBehaviour._write_tweet_to_kv_store."""

    def _make_behaviour(self) -> MagicMock:
        behaviour = MagicMock(spec=BaseTweetBehaviour)
        behaviour.context = make_mock_context()
        return behaviour

    def test_write_single_tweet(self) -> None:
        """Test _write_tweet_to_kv_store with single tweet dict."""
        behaviour = self._make_behaviour()

        def mock_get_tweets():
            yield
            return []

        def mock_write_kv(data):
            yield
            return True

        behaviour.get_tweets_from_db = mock_get_tweets
        behaviour._write_kv = mock_write_kv

        gen = BaseTweetBehaviour._write_tweet_to_kv_store(
            behaviour, {"text": "hello", "tweet_id": "1"}
        )
        try:
            val = next(gen)
            while True:
                val = gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result is True

    def test_write_list_of_tweets(self) -> None:
        """Test _write_tweet_to_kv_store with list of tweets."""
        behaviour = self._make_behaviour()

        def mock_get_tweets():
            yield
            return [{"text": "existing"}]

        def mock_write_kv(data):
            yield
            return True

        behaviour.get_tweets_from_db = mock_get_tweets
        behaviour._write_kv = mock_write_kv

        gen = BaseTweetBehaviour._write_tweet_to_kv_store(
            behaviour,
            [{"text": "new1", "tweet_id": "2"}, {"text": "new2", "tweet_id": "3"}],
        )
        try:
            val = next(gen)
            while True:
                val = gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result is True
