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

"""Tests for behaviour_classes/db.py."""

# pylint: disable=protected-access,too-few-public-methods,unpacking-non-sequence,unused-argument,used-before-assignment,useless-return

import json
from typing import Any
from unittest.mock import MagicMock

from packages.valory.skills.memeooorr_abci.behaviour_classes.db import (
    LoadDatabaseBehaviour,
)
from packages.valory.skills.memeooorr_abci.tests.conftest import (
    SAFE_ADDRESS,
    make_mock_context,
    make_mock_params,
    make_mock_synchronized_data,
)


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
        result = LoadDatabaseBehaviour.gather_agent_details(behaviour, "custom persona")
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

        def mock_write_kv(data):  # type: ignore[no-untyped-def]
            write_calls.append(data)
            yield
            return True

        def mock_read_kv(keys):  # type: ignore[no-untyped-def]
            yield
            return None

        behaviour._write_kv = mock_write_kv
        behaviour._read_kv = mock_read_kv

        gen = LoadDatabaseBehaviour.populate_keys_in_kv(behaviour)
        try:
            next(gen)
            while True:
                gen.send(None)
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

        def mock_write_kv(data):  # type: ignore[no-untyped-def]
            write_calls.append(data)
            yield
            return True

        # Counter to track which read call we're on
        read_count = [0]

        def mock_read_kv(keys):  # type: ignore[no-untyped-def]
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
            next(gen)
            while True:
                gen.send(None)
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

    def test_load_db_delegates_and_calls_agents_fun_db_load(self) -> None:
        """Test load_db aggregates sub-loaders and triggers agents_fun_db.load()."""
        behaviour = self._make_behaviour()
        load_called = [False]

        def mock_get_persona():  # type: ignore[no-untyped-def]
            yield
            return "test persona"

        def mock_get_heart_cooldown():  # type: ignore[no-untyped-def]
            yield
            return 24

        def mock_get_summon_cooldown():  # type: ignore[no-untyped-def]
            yield
            return 86400

        def mock_agents_fun_db_load():  # type: ignore[no-untyped-def]
            load_called[0] = True
            yield

        behaviour.get_persona = mock_get_persona
        behaviour.get_heart_cooldown_hours = mock_get_heart_cooldown
        behaviour.get_summon_cooldown_seconds = mock_get_summon_cooldown
        behaviour.context.agents_fun_db.load = mock_agents_fun_db_load

        gen = LoadDatabaseBehaviour.load_db(behaviour)
        result = None
        try:
            next(gen)
            while True:
                gen.send(None)
        except StopIteration as e:
            result = e.value

        assert isinstance(result, tuple)
        assert len(result) == 3
        # Verify agents_fun_db.load() was called (the non-obvious side effect)
        assert load_called[0], "agents_fun_db.load() must be called by load_db"


class TestAsyncAct:
    """Tests for LoadDatabaseBehaviour.async_act."""

    def _make_behaviour(self) -> MagicMock:
        behaviour = MagicMock(spec=LoadDatabaseBehaviour)
        behaviour.params = make_mock_params()
        behaviour.context = make_mock_context(params=behaviour.params)
        behaviour.synchronized_data = make_mock_synchronized_data()
        behaviour.behaviour_id = "test_behaviour"
        return behaviour

    def test_async_act_builds_correct_payload_and_writes_kv(self) -> None:
        """Test async_act builds LoadDatabasePayload with correct fields and writes agent_details to KV."""
        behaviour = self._make_behaviour()
        payloads_sent: list = []
        kv_writes: list = []

        def mock_load_db():  # type: ignore[no-untyped-def]
            yield
            return ("test persona", 24, 86400)

        def mock_populate_keys_in_kv():  # type: ignore[no-untyped-def]
            yield
            return None

        def mock_init_own_twitter_details():  # type: ignore[no-untyped-def]
            yield
            return None

        def mock_gather_agent_details(persona):  # type: ignore[no-untyped-def]
            return json.dumps({"persona": persona, "twitter_username": "test_user"})

        def mock_write_kv(data):  # type: ignore[no-untyped-def]
            kv_writes.append(data)
            yield
            return True

        def mock_send_a2a_transaction(payload):  # type: ignore[no-untyped-def]
            payloads_sent.append(payload)
            yield
            return None

        def mock_wait_until_round_end():  # type: ignore[no-untyped-def]
            yield
            return None

        behaviour.load_db = mock_load_db
        behaviour.populate_keys_in_kv = mock_populate_keys_in_kv
        behaviour.init_own_twitter_details = mock_init_own_twitter_details
        behaviour.gather_agent_details = mock_gather_agent_details
        behaviour._write_kv = mock_write_kv
        behaviour.send_a2a_transaction = mock_send_a2a_transaction
        behaviour.wait_until_round_end = mock_wait_until_round_end
        behaviour.set_done = MagicMock()

        gen = LoadDatabaseBehaviour.async_act(behaviour)
        try:
            _ = next(gen)
            while True:
                _ = gen.send(None)
        except StopIteration:
            pass

        behaviour.set_done.assert_called_once()

        # Verify payload was constructed with correct fields from load_db output
        assert len(payloads_sent) == 1
        payload = payloads_sent[0]
        assert payload.persona == "test persona"
        assert payload.heart_cooldown_hours == 24
        assert payload.summon_cooldown_seconds == 86400

        # Verify agent_details in payload matches gather_agent_details output
        agent_details = json.loads(payload.agent_details)
        assert agent_details["persona"] == "test persona"
        assert agent_details["twitter_username"] == "test_user"

        # Verify agent_details was written to KV store
        assert any("agent_details" in w for w in kv_writes)
