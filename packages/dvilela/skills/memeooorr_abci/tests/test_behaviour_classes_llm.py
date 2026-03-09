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

"""Tests for behaviour_classes/llm.py."""

# pylint: disable=protected-access,too-few-public-methods,unsubscriptable-object,unused-argument,used-before-assignment,useless-return

import json
from typing import List, Optional
from unittest.mock import MagicMock

from packages.dvilela.skills.memeooorr_abci.behaviour_classes.llm import (
    ActionDecisionBehaviour,
)
from packages.dvilela.skills.memeooorr_abci.rounds import ActionDecisionRound, Event

from .conftest import make_mock_context, make_mock_params, make_mock_synchronized_data


class TestActionDecisionBehaviourMatchingRound:
    """Tests for ActionDecisionBehaviour matching round."""

    def test_matching_round(self) -> None:
        """Test that matching_round is ActionDecisionRound."""
        assert ActionDecisionBehaviour.matching_round is ActionDecisionRound


class TestParseFeedbackData:
    """Tests for _parse_feedback_data method."""

    def _make_behaviour(self) -> MagicMock:
        behaviour = MagicMock(spec=ActionDecisionBehaviour)
        behaviour.context = make_mock_context()
        return behaviour

    def test_dict_input(self) -> None:
        """Test _parse_feedback_data with dict input."""
        behaviour = self._make_behaviour()
        data = {"likes": 5, "replies": []}
        result = ActionDecisionBehaviour._parse_feedback_data(behaviour, data)
        assert result == data

    def test_json_string_input(self) -> None:
        """Test _parse_feedback_data with valid JSON string."""
        behaviour = self._make_behaviour()
        data = json.dumps({"likes": 5, "replies": []})
        result = ActionDecisionBehaviour._parse_feedback_data(behaviour, data)
        assert result == {"likes": 5, "replies": []}

    def test_invalid_json_string(self) -> None:
        """Test _parse_feedback_data with invalid JSON string."""
        behaviour = self._make_behaviour()
        result = ActionDecisionBehaviour._parse_feedback_data(
            behaviour, "not-valid-json{"
        )
        assert result is None

    def test_none_input(self) -> None:
        """Test _parse_feedback_data with None input."""
        behaviour = self._make_behaviour()
        result = ActionDecisionBehaviour._parse_feedback_data(behaviour, None)
        assert result == {}

    def test_empty_string_input(self) -> None:
        """Test _parse_feedback_data with empty string."""
        behaviour = self._make_behaviour()
        result = ActionDecisionBehaviour._parse_feedback_data(behaviour, "")
        assert result == {}

    def test_other_type_input(self) -> None:
        """Test _parse_feedback_data with unexpected type."""
        behaviour = self._make_behaviour()
        result = ActionDecisionBehaviour._parse_feedback_data(behaviour, 42)
        assert result is None


class TestFormatReplies:
    """Tests for _format_replies method."""

    def _make_behaviour(self) -> MagicMock:
        behaviour = MagicMock(spec=ActionDecisionBehaviour)
        behaviour.context = make_mock_context()
        return behaviour

    def test_none_input(self) -> None:
        """Test _format_replies with None input."""
        behaviour = self._make_behaviour()
        result = ActionDecisionBehaviour._format_replies(behaviour, None)
        assert result == ""

    def test_empty_replies(self) -> None:
        """Test _format_replies with empty dict."""
        behaviour = self._make_behaviour()
        result = ActionDecisionBehaviour._format_replies(behaviour, {"replies": []})
        assert result == ""

    def test_no_replies_key(self) -> None:
        """Test _format_replies with dict without replies key."""
        behaviour = self._make_behaviour()
        result = ActionDecisionBehaviour._format_replies(behaviour, {})
        assert result == ""

    def test_replies_not_list(self) -> None:
        """Test _format_replies when replies is not a list."""
        behaviour = self._make_behaviour()
        result = ActionDecisionBehaviour._format_replies(
            behaviour, {"replies": "not a list"}
        )
        assert result == ""

    def test_dict_replies(self) -> None:
        """Test _format_replies with dict reply items."""
        behaviour = self._make_behaviour()
        replies = [
            {"text": "Great post!", "tweet_id": "123"},
            {"text": "Nice one!", "tweet_id": "456"},
        ]
        result = ActionDecisionBehaviour._format_replies(
            behaviour, {"replies": replies}
        )
        assert "Great post!" in result
        assert "Nice one!" in result
        assert "123" in result

    def test_non_dict_reply_items_skipped(self) -> None:
        """Test _format_replies skips non-dict items."""
        behaviour = self._make_behaviour()
        replies = [{"text": "Good", "tweet_id": "1"}, "not a dict", 42]
        result = ActionDecisionBehaviour._format_replies(
            behaviour, {"replies": replies}
        )
        assert "Good" in result


class TestGetTokenDetails:
    """Tests for _get_token_details method."""

    def _make_behaviour(self, meme_coins: Optional[List] = None) -> MagicMock:
        behaviour = MagicMock(spec=ActionDecisionBehaviour)
        behaviour.context = make_mock_context()
        behaviour.synchronized_data = make_mock_synchronized_data(
            meme_coins=meme_coins or []
        )
        return behaviour

    def test_summon_action(self) -> None:
        """Test _get_token_details for summon action."""
        behaviour = self._make_behaviour()
        action = {
            "token_name": "TestToken",
            "token_ticker": "TT",
            "token_supply": 1000000,
        }
        name, ticker, supply, address = ActionDecisionBehaviour._get_token_details(
            behaviour, "summon", action, None
        )
        assert name == "TestToken"
        assert ticker == "TT"
        assert supply == 1000000
        assert address is None

    def test_existing_token_action(self) -> None:
        """Test _get_token_details finds token by nonce."""
        meme_coins = [
            {
                "token_nonce": 5,
                "token_name": "MyCoin",
                "token_ticker": "MC",
                "token_address": "0xabc",
            }
        ]
        behaviour = self._make_behaviour(meme_coins=meme_coins)
        name, ticker, supply, address = ActionDecisionBehaviour._get_token_details(
            behaviour, "heart", {}, 5
        )
        assert name == "MyCoin"
        assert ticker == "MC"
        assert address == "0xabc"
        assert supply is None

    def test_token_not_found(self) -> None:
        """Test _get_token_details when token nonce not found."""
        meme_coins = [
            {
                "token_nonce": 5,
                "token_name": "MyCoin",
                "token_ticker": "MC",
                "token_address": "0xabc",
            }
        ]
        behaviour = self._make_behaviour(meme_coins=meme_coins)
        name, ticker, supply, address = ActionDecisionBehaviour._get_token_details(
            behaviour, "heart", {}, 99
        )
        assert name is None
        assert ticker is None
        assert supply is None
        assert address is None

    def test_none_token_nonce(self) -> None:
        """Test _get_token_details with None token_nonce for non-summon action."""
        behaviour = self._make_behaviour()
        name, ticker, supply, address = ActionDecisionBehaviour._get_token_details(
            behaviour, "heart", {}, None
        )
        assert name is None
        assert ticker is None
        assert supply is None
        assert address is None


class TestGetEventMemecoinDisabled:
    """Tests for get_event when memecoin logic is disabled."""

    def _make_behaviour(self) -> MagicMock:
        behaviour = MagicMock(spec=ActionDecisionBehaviour)
        behaviour.params = make_mock_params(is_memecoin_logic_enabled=False)
        behaviour.context = make_mock_context(params=behaviour.params)
        behaviour.synchronized_data = make_mock_synchronized_data()
        behaviour.get_sync_timestamp = MagicMock(return_value=1700000000.0)
        return behaviour

    def test_memecoin_disabled_no_llm_response(self) -> None:
        """Test get_event returns WAIT when memecoin disabled and LLM returns None."""
        behaviour = self._make_behaviour()

        def mock_get_tweets_from_db():  # type: ignore[no-untyped-def]
            yield
            return []

        def mock_get_persona():  # type: ignore[no-untyped-def]
            yield
            return "test persona"

        def mock_call_genai(prompt, schema=None):  # type: ignore[no-untyped-def]
            yield
            return None

        behaviour.get_tweets_from_db = mock_get_tweets_from_db
        behaviour.get_persona = mock_get_persona
        behaviour._call_genai = mock_call_genai

        gen = ActionDecisionBehaviour.get_event(behaviour)
        result = None
        try:
            _ = next(gen)
            while True:
                _ = gen.send(None)
        except StopIteration as e:
            result = e.value

        event = result[0]
        assert event == Event.WAIT.value

    def test_memecoin_disabled_invalid_json_response(self) -> None:
        """Test get_event returns WAIT when memecoin disabled and LLM returns invalid JSON."""
        behaviour = self._make_behaviour()

        def mock_get_tweets_from_db():  # type: ignore[no-untyped-def]
            yield
            return []

        def mock_get_persona():  # type: ignore[no-untyped-def]
            yield
            return "test persona"

        def mock_call_genai(prompt, schema=None):  # type: ignore[no-untyped-def]
            yield
            return "not valid json{"

        behaviour.get_tweets_from_db = mock_get_tweets_from_db
        behaviour.get_persona = mock_get_persona
        behaviour._call_genai = mock_call_genai

        gen = ActionDecisionBehaviour.get_event(behaviour)
        result = None
        try:
            _ = next(gen)
            while True:
                _ = gen.send(None)
        except StopIteration as e:
            result = e.value

        event = result[0]
        assert event == Event.WAIT.value

    def test_memecoin_disabled_valid_response_no_persona(self) -> None:
        """Test get_event returns SKIP when memecoin disabled and valid response without persona."""
        behaviour = self._make_behaviour()

        def mock_get_tweets_from_db():  # type: ignore[no-untyped-def]
            yield
            return []

        def mock_get_persona():  # type: ignore[no-untyped-def]
            yield
            return "test persona"

        def mock_call_genai(prompt, schema=None):  # type: ignore[no-untyped-def]
            yield
            return json.dumps({"action": "none"})

        behaviour.get_tweets_from_db = mock_get_tweets_from_db
        behaviour.get_persona = mock_get_persona
        behaviour._call_genai = mock_call_genai

        gen = ActionDecisionBehaviour.get_event(behaviour)
        result = None
        try:
            _ = next(gen)
            while True:
                _ = gen.send(None)
        except StopIteration as e:
            result = e.value

        event = result[0]
        assert event == Event.SKIP.value
        new_persona = result[9]
        assert new_persona is None

    def test_memecoin_disabled_valid_response_with_persona(self) -> None:
        """Test get_event returns SKIP with new_persona when provided."""
        behaviour = self._make_behaviour()

        def mock_get_tweets_from_db():  # type: ignore[no-untyped-def]
            yield
            return []

        def mock_get_persona():  # type: ignore[no-untyped-def]
            yield
            return "test persona"

        def mock_call_genai(prompt, schema=None):  # type: ignore[no-untyped-def]
            yield
            return json.dumps({"new_persona": "new personality"})

        def mock_write_kv(data):  # type: ignore[no-untyped-def]
            yield
            return True

        behaviour.get_tweets_from_db = mock_get_tweets_from_db
        behaviour.get_persona = mock_get_persona
        behaviour._call_genai = mock_call_genai
        behaviour._write_kv = mock_write_kv
        behaviour.shared_state = MagicMock()

        gen = ActionDecisionBehaviour.get_event(behaviour)
        result = None
        try:
            _ = next(gen)
            while True:
                _ = gen.send(None)
        except StopIteration as e:
            result = e.value

        event = result[0]
        assert event == Event.SKIP.value
        new_persona = result[9]
        assert new_persona == "new personality"


class TestGetEventMemecoinEnabled:
    """Tests for get_event when memecoin logic is enabled."""

    def _make_behaviour(self, meme_coins: Optional[List] = None) -> MagicMock:
        behaviour = MagicMock(spec=ActionDecisionBehaviour)
        behaviour.params = make_mock_params(is_memecoin_logic_enabled=True)
        behaviour.context = make_mock_context(params=behaviour.params)
        coins = meme_coins or []
        behaviour.synchronized_data = make_mock_synchronized_data(meme_coins=coins)
        behaviour.get_sync_timestamp = MagicMock(return_value=1700000000.0)
        behaviour.get_chain_id = MagicMock(return_value="base")
        behaviour.get_native_ticker = MagicMock(return_value="ETH")
        return behaviour

    def test_llm_returns_none(self) -> None:
        """Test get_event returns WAIT when LLM returns None."""
        behaviour = self._make_behaviour()

        def mock_get_tweets_from_db():  # type: ignore[no-untyped-def]
            yield
            return []

        def mock_get_persona():  # type: ignore[no-untyped-def]
            yield
            return "test"

        def mock_get_native_balance():  # type: ignore[no-untyped-def]
            yield
            return {"safe": 1.0, "agent": 1.0}

        def mock_get_summon_cooldown():  # type: ignore[no-untyped-def]
            yield
            return 86400

        def mock_read_kv(keys):  # type: ignore[no-untyped-def]
            yield
            return {"last_summon_timestamp": str(1700000000.0)}

        def mock_call_genai(prompt, schema=None):  # type: ignore[no-untyped-def]
            yield
            return None

        behaviour.get_tweets_from_db = mock_get_tweets_from_db
        behaviour.get_persona = mock_get_persona
        behaviour.get_native_balance = mock_get_native_balance
        behaviour.get_summon_cooldown_seconds = mock_get_summon_cooldown
        behaviour._read_kv = mock_read_kv
        behaviour._call_genai = mock_call_genai

        gen = ActionDecisionBehaviour.get_event(behaviour)
        result = None
        try:
            _ = next(gen)
            while True:
                _ = gen.send(None)
        except StopIteration as e:
            result = e.value

        assert result[0] == Event.WAIT.value

    def test_llm_returns_invalid_json(self) -> None:
        """Test get_event returns WAIT when LLM returns invalid JSON."""
        behaviour = self._make_behaviour()

        def mock_get_tweets_from_db():  # type: ignore[no-untyped-def]
            yield
            return []

        def mock_get_persona():  # type: ignore[no-untyped-def]
            yield
            return "test"

        def mock_get_native_balance():  # type: ignore[no-untyped-def]
            yield
            return {"safe": 1.0, "agent": 1.0}

        def mock_get_summon_cooldown():  # type: ignore[no-untyped-def]
            yield
            return 86400

        def mock_read_kv(keys):  # type: ignore[no-untyped-def]
            yield
            return {"last_summon_timestamp": str(1700000000.0)}

        def mock_call_genai(prompt, schema=None):  # type: ignore[no-untyped-def]
            yield
            return "not valid json{"

        behaviour.get_tweets_from_db = mock_get_tweets_from_db
        behaviour.get_persona = mock_get_persona
        behaviour.get_native_balance = mock_get_native_balance
        behaviour.get_summon_cooldown_seconds = mock_get_summon_cooldown
        behaviour._read_kv = mock_read_kv
        behaviour._call_genai = mock_call_genai

        gen = ActionDecisionBehaviour.get_event(behaviour)
        result = None
        try:
            _ = next(gen)
            while True:
                _ = gen.send(None)
        except StopIteration as e:
            result = e.value

        assert result[0] == Event.WAIT.value

    def test_llm_returns_action_none(self) -> None:
        """Test get_event returns WAIT when LLM decides on action 'none'."""
        behaviour = self._make_behaviour()

        def mock_get_tweets_from_db():  # type: ignore[no-untyped-def]
            yield
            return []

        def mock_get_persona():  # type: ignore[no-untyped-def]
            yield
            return "test"

        def mock_get_native_balance():  # type: ignore[no-untyped-def]
            yield
            return {"safe": 1.0, "agent": 1.0}

        def mock_get_summon_cooldown():  # type: ignore[no-untyped-def]
            yield
            return 86400

        def mock_read_kv(keys):  # type: ignore[no-untyped-def]
            yield
            return {"last_summon_timestamp": str(1700000000.0)}

        def mock_call_genai(prompt, schema=None):  # type: ignore[no-untyped-def]
            yield
            return json.dumps({"action_name": "none"})

        def mock_replace_tweet(prompt):  # type: ignore[no-untyped-def]
            yield
            return None

        behaviour.get_tweets_from_db = mock_get_tweets_from_db
        behaviour.get_persona = mock_get_persona
        behaviour.get_native_balance = mock_get_native_balance
        behaviour.get_summon_cooldown_seconds = mock_get_summon_cooldown
        behaviour._read_kv = mock_read_kv
        behaviour._call_genai = mock_call_genai
        behaviour.replace_tweet_with_alternative_model = mock_replace_tweet

        gen = ActionDecisionBehaviour.get_event(behaviour)
        result = None
        try:
            _ = next(gen)
            while True:
                _ = gen.send(None)
        except StopIteration as e:
            result = e.value

        assert result[0] == Event.WAIT.value

    def test_heart_invalid_nonce(self) -> None:
        """Test get_event returns WAIT when heart action has invalid nonce."""
        meme_coins = [
            {
                "token_nonce": 5,
                "token_name": "Test",
                "token_ticker": "TST",
                "token_address": "0xabc",
                "heart_count": 0,
                "available_actions": ["heart"],
            }
        ]
        behaviour = self._make_behaviour(meme_coins=meme_coins)

        def mock_get_tweets_from_db():  # type: ignore[no-untyped-def]
            yield
            return []

        def mock_get_persona():  # type: ignore[no-untyped-def]
            yield
            return "test"

        def mock_get_native_balance():  # type: ignore[no-untyped-def]
            yield
            return {"safe": 1.0, "agent": 1.0}

        def mock_get_summon_cooldown():  # type: ignore[no-untyped-def]
            yield
            return 86400

        def mock_read_kv(keys):  # type: ignore[no-untyped-def]
            yield
            return {"last_summon_timestamp": str(1700000000.0)}

        def mock_call_genai(prompt, schema=None):  # type: ignore[no-untyped-def]
            yield
            return json.dumps(
                {
                    "action_name": "heart",
                    "heart": {"token_nonce": 999, "amount": 100},
                    "action_tweet": "hearting!",
                }
            )

        def mock_replace_tweet(prompt):  # type: ignore[no-untyped-def]
            yield
            return None

        behaviour.get_tweets_from_db = mock_get_tweets_from_db
        behaviour.get_persona = mock_get_persona
        behaviour.get_native_balance = mock_get_native_balance
        behaviour.get_summon_cooldown_seconds = mock_get_summon_cooldown
        behaviour._read_kv = mock_read_kv
        behaviour._call_genai = mock_call_genai
        behaviour.replace_tweet_with_alternative_model = mock_replace_tweet

        gen = ActionDecisionBehaviour.get_event(behaviour)
        result = None
        try:
            _ = next(gen)
            while True:
                _ = gen.send(None)
        except StopIteration as e:
            result = e.value

        assert result[0] == Event.WAIT.value

    def test_action_not_in_available_actions(self) -> None:
        """Test get_event returns WAIT when action is not available for token."""
        meme_coins = [
            {
                "token_nonce": 5,
                "token_name": "Test",
                "token_ticker": "TST",
                "token_address": "0xabc",
                "heart_count": 0,
                "available_actions": ["collect"],  # heart not available
            }
        ]
        behaviour = self._make_behaviour(meme_coins=meme_coins)

        def mock_get_tweets_from_db():  # type: ignore[no-untyped-def]
            yield
            return []

        def mock_get_persona():  # type: ignore[no-untyped-def]
            yield
            return "test"

        def mock_get_native_balance():  # type: ignore[no-untyped-def]
            yield
            return {"safe": 1.0, "agent": 1.0}

        def mock_get_summon_cooldown():  # type: ignore[no-untyped-def]
            yield
            return 86400

        def mock_read_kv(keys):  # type: ignore[no-untyped-def]
            yield
            return {"last_summon_timestamp": str(1700000000.0)}

        def mock_call_genai(prompt, schema=None):  # type: ignore[no-untyped-def]
            yield
            return json.dumps(
                {
                    "action_name": "heart",
                    "heart": {"token_nonce": 5, "amount": 100},
                    "action_tweet": "hearting!",
                }
            )

        def mock_replace_tweet(prompt):  # type: ignore[no-untyped-def]
            yield
            return None

        behaviour.get_tweets_from_db = mock_get_tweets_from_db
        behaviour.get_persona = mock_get_persona
        behaviour.get_native_balance = mock_get_native_balance
        behaviour.get_summon_cooldown_seconds = mock_get_summon_cooldown
        behaviour._read_kv = mock_read_kv
        behaviour._call_genai = mock_call_genai
        behaviour.replace_tweet_with_alternative_model = mock_replace_tweet

        gen = ActionDecisionBehaviour.get_event(behaviour)
        result = None
        try:
            _ = next(gen)
            while True:
                _ = gen.send(None)
        except StopIteration as e:
            result = e.value

        assert result[0] == Event.WAIT.value

    def test_summon_not_available_returns_wait(self) -> None:
        """Test summon action returns WAIT when cooldown not met."""
        behaviour = self._make_behaviour()

        def mock_get_tweets_from_db():  # type: ignore[no-untyped-def]
            yield
            return []

        def mock_get_persona():  # type: ignore[no-untyped-def]
            yield
            return "test"

        def mock_get_native_balance():  # type: ignore[no-untyped-def]
            yield
            return {"safe": 1.0, "agent": 1.0}

        def mock_get_summon_cooldown():  # type: ignore[no-untyped-def]
            yield
            return 86400

        # last_summon_timestamp is very recent (current time)
        def mock_read_kv(keys):  # type: ignore[no-untyped-def]
            yield
            return {"last_summon_timestamp": str(1700000000.0)}

        def mock_call_genai(prompt, schema=None):  # type: ignore[no-untyped-def]
            yield
            return json.dumps(
                {
                    "action_name": "summon",
                    "summon": {
                        "token_name": "NewToken",
                        "token_ticker": "NT",
                        "token_supply": 1000000,
                        "amount": 100,
                    },
                    "action_tweet": "summoning!",
                }
            )

        def mock_replace_tweet(prompt):  # type: ignore[no-untyped-def]
            yield
            return None

        behaviour.get_tweets_from_db = mock_get_tweets_from_db
        behaviour.get_persona = mock_get_persona
        behaviour.get_native_balance = mock_get_native_balance
        behaviour.get_summon_cooldown_seconds = mock_get_summon_cooldown
        behaviour._read_kv = mock_read_kv
        behaviour._call_genai = mock_call_genai
        behaviour.replace_tweet_with_alternative_model = mock_replace_tweet

        gen = ActionDecisionBehaviour.get_event(behaviour)
        try:
            _ = next(gen)
            while True:
                _ = gen.send(None)
        except StopIteration as e:
            result = e.value

        # Since last_summon = current_timestamp, seconds_since = 0 < 86400
        assert result[0] == Event.WAIT.value
