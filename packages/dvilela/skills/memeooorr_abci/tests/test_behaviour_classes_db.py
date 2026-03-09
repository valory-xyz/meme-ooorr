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

"""Tests for behaviour_classes/db.py."""

import json
from typing import Any
from unittest.mock import MagicMock, PropertyMock

import pytest

from packages.dvilela.skills.memeooorr_abci.behaviour_classes.base import (
    HOUR_TO_SECONDS,
)
from packages.dvilela.skills.memeooorr_abci.behaviour_classes.db import (
    LoadDatabaseBehaviour,
)
from packages.dvilela.skills.memeooorr_abci.rounds import LoadDatabaseRound

from .conftest import (
    SAFE_ADDRESS,
    SENDER,
    make_mock_context,
    make_mock_params,
    make_mock_synchronized_data,
)


class TestLoadDatabaseBehaviourMatchingRound:
    """Tests for LoadDatabaseBehaviour matching round."""

    def test_matching_round(self) -> None:
        """Test that matching_round is LoadDatabaseRound."""
        assert LoadDatabaseBehaviour.matching_round is LoadDatabaseRound


class TestGatherAgentDetails:
    """Tests for gather_agent_details method."""

    def _make_behaviour(self) -> MagicMock:
        behaviour = MagicMock(spec=LoadDatabaseBehaviour)
        behaviour.params = make_mock_params()
        behaviour.context = make_mock_context(params=behaviour.params)
        behaviour.synchronized_data = make_mock_synchronized_data()
        return behaviour

    def test_gather_agent_details_returns_json_string(self) -> None:
        """Test gather_agent_details returns valid JSON string."""
        behaviour = self._make_behaviour()
        result = LoadDatabaseBehaviour.gather_agent_details(behaviour, "test persona")
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_gather_agent_details_contains_twitter_username(self) -> None:
        """Test gather_agent_details includes twitter_username."""
        behaviour = self._make_behaviour()
        result = LoadDatabaseBehaviour.gather_agent_details(behaviour, "test persona")
        parsed = json.loads(result)
        assert "twitter_username" in parsed
        assert parsed["twitter_username"] == "test_user"

    def test_gather_agent_details_contains_twitter_user_id(self) -> None:
        """Test gather_agent_details includes twitter_user_id."""
        behaviour = self._make_behaviour()
        result = LoadDatabaseBehaviour.gather_agent_details(behaviour, "test persona")
        parsed = json.loads(result)
        assert "twitter_user_id" in parsed
        assert parsed["twitter_user_id"] == "12345"

    def test_gather_agent_details_contains_safe_address(self) -> None:
        """Test gather_agent_details includes safe_address."""
        behaviour = self._make_behaviour()
        result = LoadDatabaseBehaviour.gather_agent_details(behaviour, "test persona")
        parsed = json.loads(result)
        assert "safe_address" in parsed
        assert parsed["safe_address"] == SAFE_ADDRESS

    def test_gather_agent_details_contains_persona(self) -> None:
        """Test gather_agent_details includes persona."""
        behaviour = self._make_behaviour()
        result = LoadDatabaseBehaviour.gather_agent_details(
            behaviour, "custom persona"
        )
        parsed = json.loads(result)
        assert parsed["persona"] == "custom persona"

    def test_gather_agent_details_contains_twitter_display_name(self) -> None:
        """Test gather_agent_details includes twitter_display_name."""
        behaviour = self._make_behaviour()
        result = LoadDatabaseBehaviour.gather_agent_details(behaviour, "test persona")
        parsed = json.loads(result)
        assert "twitter_display_name" in parsed
        assert parsed["twitter_display_name"] == "Test User"


class TestPopulateKeysInKv:
    """Tests for populate_keys_in_kv method."""

    def _make_behaviour(self, **kwargs: Any) -> MagicMock:
        behaviour = MagicMock(spec=LoadDatabaseBehaviour)
        behaviour.params = make_mock_params(**kwargs)
        behaviour.context = make_mock_context(params=behaviour.params)
        behaviour.synchronized_data = make_mock_synchronized_data()
        behaviour.get_sync_timestamp = MagicMock(return_value=1700000000.0)
        return behaviour

    def test_populate_keys_writes_initial_keys(self) -> None:
        """Test populate_keys_in_kv writes all initial keys."""
        behaviour = self._make_behaviour()
        write_calls = []

        def mock_write_kv(data):
            write_calls.append(data)
            yield
            return True

        def mock_read_kv(keys):
            yield
            return None

        behaviour._write_kv = mock_write_kv
        behaviour._read_kv = mock_read_kv

        gen = LoadDatabaseBehaviour.populate_keys_in_kv(behaviour)
        try:
            val = next(gen)
            while True:
                val = gen.send(None)
        except StopIteration:
            pass

        # Should write: previous_tweets, other_tweets, interacted_tweet_ids,
        # pending_tweets, last_summon_timestamp, last_heart_timestamp
        written_keys = set()
        for call in write_calls:
            written_keys.update(call.keys())

        assert "previous_tweets_for_tw_mech" in written_keys
        assert "other_tweets_for_tw_mech" in written_keys
        assert "interacted_tweet_ids_for_tw_mech" in written_keys
        assert "pending_tweets_for_tw_mech" in written_keys
        assert "last_summon_timestamp" in written_keys
        assert "last_heart_timestamp" in written_keys

    def test_populate_keys_skips_last_summon_if_exists(self) -> None:
        """Test populate_keys_in_kv skips last_summon_timestamp if already set."""
        behaviour = self._make_behaviour()
        write_calls = []

        def mock_write_kv(data):
            write_calls.append(data)
            yield
            return True

        # Counter to track which read call we're on
        read_count = [0]

        def mock_read_kv(keys):
            read_count[0] += 1
            yield
            if "last_summon_timestamp" in keys:
                return {"last_summon_timestamp": "1700000000.0"}
            if "last_heart_timestamp" in keys:
                return {"last_heart_timestamp": "1700000000.0"}
            return None

        behaviour._write_kv = mock_write_kv
        behaviour._read_kv = mock_read_kv

        gen = LoadDatabaseBehaviour.populate_keys_in_kv(behaviour)
        try:
            val = next(gen)
            while True:
                val = gen.send(None)
        except StopIteration:
            pass

        # Should NOT write last_summon_timestamp or last_heart_timestamp
        written_keys = set()
        for call in write_calls:
            written_keys.update(call.keys())

        assert "last_summon_timestamp" not in written_keys
        assert "last_heart_timestamp" not in written_keys


class TestLoadDb:
    """Tests for load_db method."""

    def _make_behaviour(self) -> MagicMock:
        behaviour = MagicMock(spec=LoadDatabaseBehaviour)
        behaviour.params = make_mock_params()
        behaviour.context = make_mock_context(params=behaviour.params)
        behaviour.synchronized_data = make_mock_synchronized_data()
        return behaviour

    def test_load_db_returns_tuple(self) -> None:
        """Test load_db returns a 3-tuple."""
        behaviour = self._make_behaviour()

        def mock_get_persona():
            yield
            return "test persona"

        def mock_get_heart_cooldown():
            yield
            return 24

        def mock_get_summon_cooldown():
            yield
            return 86400

        behaviour.get_persona = mock_get_persona
        behaviour.get_heart_cooldown_hours = mock_get_heart_cooldown
        behaviour.get_summon_cooldown_seconds = mock_get_summon_cooldown

        gen = LoadDatabaseBehaviour.load_db(behaviour)
        try:
            val = next(gen)
            while True:
                val = gen.send(None)
        except StopIteration as e:
            result = e.value

        assert isinstance(result, tuple)
        assert len(result) == 3
        persona, heart_cd, summon_cd = result
        assert persona == "test persona"
        assert heart_cd == 24
        assert summon_cd == 86400
