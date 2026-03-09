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

"""Tests for behaviour_classes/base.py."""

# pylint: disable=protected-access,unused-argument,used-before-assignment,use-implicit-booleaness-not-comparison,useless-return

import abc
import json
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

from packages.dvilela.skills.memeooorr_abci.behaviour_classes.base import (
    BASE_CHAIN_ID,
    CELO_CHAIN_ID,
    MemeooorrBaseBehaviour,
    is_tweet_valid,
)

from .conftest import (
    MEME_FACTORY_ADDRESS_BASE,
    MEME_FACTORY_ADDRESS_CELO,
    OLAS_TOKEN_ADDRESS_BASE,
    OLAS_TOKEN_ADDRESS_CELO,
    SERVICE_REGISTRY_ADDRESS_BASE,
    SERVICE_REGISTRY_ADDRESS_CELO,
    make_mock_context,
    make_mock_params,
    make_mock_synchronized_data,
)


class TestIsTweetValid:
    """Tests for is_tweet_valid function."""

    def test_short_tweet_is_valid(self) -> None:
        """Test that a short tweet is valid."""
        assert is_tweet_valid("Hello world") is True

    def test_empty_tweet_is_valid(self) -> None:
        """Test that an empty tweet is valid."""
        assert is_tweet_valid("") is True

    def test_max_length_tweet_is_valid(self) -> None:
        """Test that a tweet at max length is valid."""
        tweet = "a" * 280
        assert is_tweet_valid(tweet) is True

    def test_over_max_length_tweet_is_invalid(self) -> None:
        """Test that a tweet over max length is invalid."""
        tweet = "a" * 281
        assert is_tweet_valid(tweet) is False


class TestMemeooorrBaseBehaviour:
    """Tests for MemeooorrBaseBehaviour."""

    def test_is_abstract(self) -> None:
        """Test that MemeooorrBaseBehaviour is abstract."""
        assert abc.ABC in MemeooorrBaseBehaviour.__mro__

    def test_get_chain_id_base(self) -> None:
        """Test get_chain_id returns 'base' when home_chain_id is 'base'."""
        behaviour = MagicMock(spec=MemeooorrBaseBehaviour)
        behaviour.params = make_mock_params(home_chain_id="base")
        result = MemeooorrBaseBehaviour.get_chain_id(behaviour)
        assert result == BASE_CHAIN_ID

    def test_get_chain_id_celo(self) -> None:
        """Test get_chain_id returns 'celo' when home_chain_id is 'celo'."""
        behaviour = MagicMock(spec=MemeooorrBaseBehaviour)
        behaviour.params = make_mock_params(home_chain_id="celo")
        result = MemeooorrBaseBehaviour.get_chain_id(behaviour)
        assert result == CELO_CHAIN_ID

    def test_get_chain_id_base_uppercase(self) -> None:
        """Test get_chain_id handles uppercase 'BASE'."""
        behaviour = MagicMock(spec=MemeooorrBaseBehaviour)
        behaviour.params = make_mock_params(home_chain_id="BASE")
        result = MemeooorrBaseBehaviour.get_chain_id(behaviour)
        assert result == BASE_CHAIN_ID

    def test_get_chain_id_unknown(self) -> None:
        """Test get_chain_id returns empty string for unknown chain."""
        behaviour = MagicMock(spec=MemeooorrBaseBehaviour)
        behaviour.params = make_mock_params(home_chain_id="unknown")
        result = MemeooorrBaseBehaviour.get_chain_id(behaviour)
        assert result == ""

    def test_get_native_ticker_base(self) -> None:
        """Test get_native_ticker returns 'ETH' for base chain."""
        behaviour = MagicMock(spec=MemeooorrBaseBehaviour)
        behaviour.params = make_mock_params(home_chain_id="base")
        result = MemeooorrBaseBehaviour.get_native_ticker(behaviour)
        assert result == "ETH"

    def test_get_native_ticker_celo(self) -> None:
        """Test get_native_ticker returns 'CELO' for celo chain."""
        behaviour = MagicMock(spec=MemeooorrBaseBehaviour)
        behaviour.params = make_mock_params(home_chain_id="celo")
        result = MemeooorrBaseBehaviour.get_native_ticker(behaviour)
        assert result == "CELO"

    def test_get_native_ticker_unknown(self) -> None:
        """Test get_native_ticker returns empty string for unknown chain."""
        behaviour = MagicMock(spec=MemeooorrBaseBehaviour)
        behaviour.params = make_mock_params(home_chain_id="unknown")
        result = MemeooorrBaseBehaviour.get_native_ticker(behaviour)
        assert result == ""

    def test_get_meme_factory_address_base(self) -> None:
        """Test get_meme_factory_address returns base address."""
        behaviour = MagicMock(spec=MemeooorrBaseBehaviour)
        behaviour.params = make_mock_params(home_chain_id="base")
        behaviour.get_chain_id = lambda: "base"
        result = MemeooorrBaseBehaviour.get_meme_factory_address(behaviour)
        assert result == MEME_FACTORY_ADDRESS_BASE

    def test_get_meme_factory_address_celo(self) -> None:
        """Test get_meme_factory_address returns celo address."""
        behaviour = MagicMock(spec=MemeooorrBaseBehaviour)
        behaviour.params = make_mock_params(home_chain_id="celo")
        behaviour.get_chain_id = lambda: "celo"
        result = MemeooorrBaseBehaviour.get_meme_factory_address(behaviour)
        assert result == MEME_FACTORY_ADDRESS_CELO

    def test_get_service_registry_address_base(self) -> None:
        """Test get_service_registry_address returns base address."""
        behaviour = MagicMock(spec=MemeooorrBaseBehaviour)
        behaviour.params = make_mock_params(home_chain_id="base")
        behaviour.get_chain_id = lambda: "base"
        result = MemeooorrBaseBehaviour.get_service_registry_address(behaviour)
        assert result == SERVICE_REGISTRY_ADDRESS_BASE

    def test_get_service_registry_address_celo(self) -> None:
        """Test get_service_registry_address returns celo address."""
        behaviour = MagicMock(spec=MemeooorrBaseBehaviour)
        behaviour.params = make_mock_params(home_chain_id="celo")
        behaviour.get_chain_id = lambda: "celo"
        result = MemeooorrBaseBehaviour.get_service_registry_address(behaviour)
        assert result == SERVICE_REGISTRY_ADDRESS_CELO

    def test_get_olas_address_base(self) -> None:
        """Test get_olas_address returns base address."""
        behaviour = MagicMock(spec=MemeooorrBaseBehaviour)
        behaviour.params = make_mock_params(home_chain_id="base")
        behaviour.get_chain_id = lambda: "base"
        result = MemeooorrBaseBehaviour.get_olas_address(behaviour)
        assert result == OLAS_TOKEN_ADDRESS_BASE

    def test_get_olas_address_celo(self) -> None:
        """Test get_olas_address returns celo address."""
        behaviour = MagicMock(spec=MemeooorrBaseBehaviour)
        behaviour.params = make_mock_params(home_chain_id="celo")
        behaviour.get_chain_id = lambda: "celo"
        result = MemeooorrBaseBehaviour.get_olas_address(behaviour)
        assert result == OLAS_TOKEN_ADDRESS_CELO

    def test_get_min_deploy_value_base(self) -> None:
        """Test get_min_deploy_value for base chain."""
        behaviour = MagicMock(spec=MemeooorrBaseBehaviour)
        behaviour.get_chain_id = lambda: "base"
        result = MemeooorrBaseBehaviour.get_min_deploy_value(behaviour)
        assert result == int(0.01 * 1e18)

    def test_get_min_deploy_value_celo(self) -> None:
        """Test get_min_deploy_value for celo chain."""
        behaviour = MagicMock(spec=MemeooorrBaseBehaviour)
        behaviour.get_chain_id = lambda: "celo"
        result = MemeooorrBaseBehaviour.get_min_deploy_value(behaviour)
        assert result == 10

    def test_get_min_deploy_value_unknown(self) -> None:
        """Test get_min_deploy_value for unknown chain returns 0."""
        behaviour = MagicMock(spec=MemeooorrBaseBehaviour)
        behaviour.get_chain_id = lambda: "unknown"
        result = MemeooorrBaseBehaviour.get_min_deploy_value(behaviour)
        assert result == 0

    def test_get_meme_factory_deployment_block_base(self) -> None:
        """Test get_meme_factory_deployment_block for base chain."""
        behaviour = MagicMock(spec=MemeooorrBaseBehaviour)
        behaviour.params = make_mock_params(
            home_chain_id="base",
            meme_factory_deployment_block_base=12345,
        )
        behaviour.get_chain_id = lambda: "base"
        result = MemeooorrBaseBehaviour.get_meme_factory_deployment_block(behaviour)
        assert result == 12345

    def test_get_meme_factory_deployment_block_celo(self) -> None:
        """Test get_meme_factory_deployment_block for celo chain."""
        behaviour = MagicMock(spec=MemeooorrBaseBehaviour)
        behaviour.params = make_mock_params(
            home_chain_id="celo",
            meme_factory_deployment_block_celo=67890,
        )
        behaviour.get_chain_id = lambda: "celo"
        result = MemeooorrBaseBehaviour.get_meme_factory_deployment_block(behaviour)
        assert result == 67890


class TestMemeooorrBaseBehaviourGenerators:
    """Tests for generator-based methods in MemeooorrBaseBehaviour."""

    def _make_behaviour(self, **params_overrides: Any) -> MagicMock:
        """Create a behaviour mock with proper context setup."""
        behaviour = MagicMock(spec=MemeooorrBaseBehaviour)
        behaviour.params = make_mock_params(**params_overrides)
        behaviour.context = make_mock_context(params=behaviour.params)
        behaviour.synchronized_data = make_mock_synchronized_data()
        behaviour.get_chain_id = lambda: "base"
        behaviour.get_meme_factory_address = lambda: MEME_FACTORY_ADDRESS_BASE
        return behaviour

    def test_get_sync_timestamp(self) -> None:
        """Test get_sync_timestamp returns float timestamp."""
        behaviour = self._make_behaviour()
        result = MemeooorrBaseBehaviour.get_sync_timestamp(behaviour)
        assert isinstance(result, float)
        assert result == 1700000000.0

    def test_get_sync_datetime(self) -> None:
        """Test get_sync_datetime returns datetime."""
        behaviour = self._make_behaviour()
        behaviour.get_sync_timestamp = MagicMock(return_value=1700000000.0)
        result = MemeooorrBaseBehaviour.get_sync_datetime(behaviour)
        assert isinstance(result, datetime)

    def test_get_sync_time_str(self) -> None:
        """Test get_sync_time_str returns formatted string."""
        behaviour = self._make_behaviour()
        behaviour.get_sync_timestamp = MagicMock(return_value=1700000000.0)
        behaviour.get_sync_datetime = MagicMock(
            return_value=datetime(2023, 11, 14, 22, 13, 20)
        )
        result = MemeooorrBaseBehaviour.get_sync_time_str(behaviour)
        assert isinstance(result, str)
        assert "2023" in result

    def test_cleanup_temp_file_with_path(self, tmp_path: Any) -> None:
        """Test _cleanup_temp_file removes file when path exists."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")
        behaviour = self._make_behaviour()
        MemeooorrBaseBehaviour._cleanup_temp_file(
            behaviour, str(test_file), "test reason"
        )
        assert not test_file.exists()

    def test_cleanup_temp_file_without_path(self) -> None:
        """Test _cleanup_temp_file logs warning when path is None."""
        behaviour = self._make_behaviour()
        # Should not raise
        MemeooorrBaseBehaviour._cleanup_temp_file(behaviour, None, "test reason")

    def test_get_tweets_from_db_no_data(self) -> None:
        """Test get_tweets_from_db returns empty list when no data."""
        behaviour = self._make_behaviour()

        def mock_read_kv(keys):  # type: ignore[no-untyped-def]
            yield
            return None

        behaviour.read_kv = mock_read_kv

        gen = MemeooorrBaseBehaviour.get_tweets_from_db(behaviour)
        next(gen)  # advance to yield
        result = None
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result == []

    def test_get_tweets_from_db_with_data(self) -> None:
        """Test get_tweets_from_db returns parsed tweets."""
        behaviour = self._make_behaviour()
        tweets = [{"text": "hello", "id": "1"}]

        def mock_read_kv(keys):  # type: ignore[no-untyped-def]
            yield
            return {"tweets": json.dumps(tweets)}

        behaviour.read_kv = mock_read_kv

        gen = MemeooorrBaseBehaviour.get_tweets_from_db(behaviour)
        next(gen)
        result = None
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result == tweets

    def test_read_json_from_kv_no_data(self) -> None:
        """Test _read_json_from_kv returns default when no data."""
        behaviour = self._make_behaviour()

        def mock_read_kv(keys):  # type: ignore[no-untyped-def]
            yield
            return None

        behaviour._read_kv = mock_read_kv

        gen = MemeooorrBaseBehaviour._read_json_from_kv(behaviour, "test_key", {})
        next(gen)
        result = None
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result == {}

    def test_read_json_from_kv_with_data(self) -> None:
        """Test _read_json_from_kv returns parsed JSON."""
        behaviour = self._make_behaviour()
        data = {"key1": "value1"}

        def mock_read_kv(keys):  # type: ignore[no-untyped-def]
            yield
            return {"test_key": json.dumps(data)}

        behaviour._read_kv = mock_read_kv

        gen = MemeooorrBaseBehaviour._read_json_from_kv(behaviour, "test_key", {})
        next(gen)
        result = None
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result == data

    def test_read_json_from_kv_invalid_json(self) -> None:
        """Test _read_json_from_kv returns default for invalid JSON."""
        behaviour = self._make_behaviour()

        def mock_read_kv(keys):  # type: ignore[no-untyped-def]
            yield
            return {"test_key": "not-valid-json{"}

        behaviour._read_kv = mock_read_kv

        gen = MemeooorrBaseBehaviour._read_json_from_kv(behaviour, "test_key", [])
        next(gen)
        result = None
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result == []

    def test_read_value_from_kv_no_data(self) -> None:
        """Test _read_value_from_kv returns default when no data."""
        behaviour = self._make_behaviour()

        def mock_read_kv(keys):  # type: ignore[no-untyped-def]
            yield
            return None

        behaviour._read_kv = mock_read_kv

        gen = MemeooorrBaseBehaviour._read_value_from_kv(
            behaviour, "test_key", "default"
        )
        next(gen)
        result = None
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result == "default"

    def test_read_value_from_kv_with_data(self) -> None:
        """Test _read_value_from_kv returns stored value."""
        behaviour = self._make_behaviour()

        def mock_read_kv(keys):  # type: ignore[no-untyped-def]
            yield
            return {"test_key": "stored_value"}

        behaviour._read_kv = mock_read_kv

        gen = MemeooorrBaseBehaviour._read_value_from_kv(
            behaviour, "test_key", "default"
        )
        next(gen)
        result = None
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result == "stored_value"

    def test_read_media_info_list_empty(self) -> None:
        """Test _read_media_info_list returns empty list when no data."""
        behaviour = self._make_behaviour()

        def mock_read_json(key, default):  # type: ignore[no-untyped-def]
            yield
            return []

        behaviour._read_json_from_kv = mock_read_json

        gen = MemeooorrBaseBehaviour._read_media_info_list(behaviour)
        next(gen)
        result = None
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result == []

    def test_read_media_info_list_not_list(self) -> None:
        """Test _read_media_info_list returns empty list when data is not a list."""
        behaviour = self._make_behaviour()

        def mock_read_json(key, default):  # type: ignore[no-untyped-def]
            yield
            return "not a list"

        behaviour._read_json_from_kv = mock_read_json

        gen = MemeooorrBaseBehaviour._read_media_info_list(behaviour)
        next(gen)
        result = None
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result == []

    def test_store_agent_action(self) -> None:
        """Test _store_agent_action stores action correctly."""
        behaviour = self._make_behaviour()
        behaviour.get_sync_timestamp = MagicMock(return_value=1700000000.0)
        write_calls = []

        def mock_read_json(key, default):  # type: ignore[no-untyped-def]
            yield
            return {}

        def mock_write_kv(data):  # type: ignore[no-untyped-def]
            write_calls.append(data)
            yield
            return True

        behaviour._read_json_from_kv = mock_read_json
        behaviour._write_kv = mock_write_kv

        gen = MemeooorrBaseBehaviour._store_agent_action(
            behaviour, "tweet_action", {"text": "hello"}
        )
        # Step through the generator
        try:
            next(gen)
            while True:
                gen.send(None)
        except StopIteration:
            pass

        assert len(write_calls) == 1
        stored = json.loads(write_calls[0]["agent_actions"])
        assert "tweet_action" in stored
        assert len(stored["tweet_action"]) == 1

    def test_get_latest_agent_actions_empty(self) -> None:
        """Test get_latest_agent_actions returns empty list when no actions."""
        behaviour = self._make_behaviour()

        def mock_read_json(key, default):  # type: ignore[no-untyped-def]
            yield
            return {}

        behaviour._read_json_from_kv = mock_read_json

        gen = MemeooorrBaseBehaviour.get_latest_agent_actions(behaviour, "tweet_action")
        next(gen)
        result = None
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result == []

    def test_get_latest_agent_actions_with_limit(self) -> None:
        """Test get_latest_agent_actions respects limit."""
        behaviour = self._make_behaviour()
        actions = {"tweet_action": [{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}]}

        def mock_read_json(key, default):  # type: ignore[no-untyped-def]
            yield
            return actions

        behaviour._read_json_from_kv = mock_read_json

        gen = MemeooorrBaseBehaviour.get_latest_agent_actions(
            behaviour, "tweet_action", limit=2
        )
        next(gen)
        result = None
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result is not None
        assert len(result) == 2
        assert result == [{"a": 3}, {"a": 4}]

    def test_get_latest_agent_actions_not_list(self) -> None:
        """Test get_latest_agent_actions returns empty list when value is not list."""
        behaviour = self._make_behaviour()

        def mock_read_json(key, default):  # type: ignore[no-untyped-def]
            yield
            return {"tweet_action": "not a list"}

        behaviour._read_json_from_kv = mock_read_json

        gen = MemeooorrBaseBehaviour.get_latest_agent_actions(behaviour, "tweet_action")
        next(gen)
        result = None
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result == []
