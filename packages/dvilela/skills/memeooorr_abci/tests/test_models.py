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

"""Tests for models.py."""

# pylint: disable=import-outside-toplevel,too-few-public-methods

from typing import Any, Dict
from unittest.mock import MagicMock, patch

from packages.dvilela.skills.memeooorr_abci.models import (
    AlternativeModelForTweets,
    Params,
)
from packages.valory.skills.mech_interact_abci.models import MechParams


class TestAlternativeModelForTweets:
    """Tests for AlternativeModelForTweets."""

    def _make_data(self, **overrides: Any) -> Dict[str, Any]:
        """Create a data dict with sensible defaults."""
        data: Dict[str, Any] = {
            "url": "https://api.example.com",
            "api_key": "test_key",
            "model": "gpt-4",
            "max_tokens": 100,
            "top_p": 1,
            "top_k": 50,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "temperature": 0.7,
        }
        data.update(overrides)
        return data

    def test_from_dict_with_valid_key(self) -> None:
        """Test from_dict with a valid API key."""
        data = self._make_data()
        model = AlternativeModelForTweets.from_dict(data)
        assert model.use is True
        assert model.api_key == "test_key"
        assert model.url == "https://api.example.com"
        assert model.model == "gpt-4"
        assert model.max_tokens == 100
        assert model.top_p == 1
        assert model.top_k == 50
        assert model.presence_penalty == 0
        assert model.frequency_penalty == 0
        assert model.temperature == 0.7

    def test_from_dict_with_none_key(self) -> None:
        """Test from_dict with None API key."""
        data = self._make_data(api_key=None)
        model = AlternativeModelForTweets.from_dict(data)
        assert model.use is False
        assert model.api_key is None

    def test_from_dict_with_empty_key(self) -> None:
        """Test from_dict with empty string API key."""
        data = self._make_data(api_key="")
        model = AlternativeModelForTweets.from_dict(data)
        assert model.use is False
        assert model.api_key is None

    def test_from_dict_with_whitespace_key(self) -> None:
        """Test from_dict with whitespace-only API key."""
        data = self._make_data(api_key="   ")
        model = AlternativeModelForTweets.from_dict(data)
        assert model.use is False
        assert model.api_key is None

    def test_from_dict_with_placeholder_key(self) -> None:
        """Test from_dict with placeholder string API key."""
        data = self._make_data(api_key="${str:}")
        model = AlternativeModelForTweets.from_dict(data)
        assert model.use is False
        assert model.api_key is None


class TestParams:
    """Tests for Params.__init__ method."""

    @staticmethod
    def _make_kwargs() -> Dict[str, Any]:
        """Create kwargs dict with all required fields for Params.__init__."""
        mock_skill_context = MagicMock()
        mock_skill_context.skill_id = "test/skill:0.1.0"
        return {
            "skill_context": mock_skill_context,
            "name": "params",
            "service_endpoint": "http://localhost:8000",
            "minimum_gas_balance": 0.001,
            "min_feedback_replies": 1,
            "meme_factory_address_base": "0x" + "d" * 40,
            "meme_factory_address_celo": "0x" + "e" * 40,
            "olas_token_address_base": "0x" + "2" * 40,
            "olas_token_address_celo": "0x" + "3" * 40,
            "service_registry_address_base": "0x" + "f" * 40,
            "service_registry_address_celo": "0x" + "1" * 40,
            "persona": "test persona",
            "home_chain_id": "base",
            "meme_factory_deployment_block_base": 0,
            "meme_factory_deployment_block_celo": 0,
            "meme_subgraph_url": "https://subgraph.example.com",
            "olas_subgraph_url": "https://olas.example.com",
            "skip_engagement": False,
            "min_summon_amount_base": 0.01,
            "max_summon_amount_base": 0.1,
            "max_heart_amount_base": 0.05,
            "min_summon_amount_celo": 10.0,
            "max_summon_amount_celo": 100.0,
            "max_heart_amount_celo": 50.0,
            "staking_token_contract_address": "0x" + "b" * 40,
            "activity_checker_contract_address": "0x" + "c" * 40,
            "fireworks_api_key": "fw_test_key",
            "alternative_model_for_tweets": {
                "url": "https://api.example.com",
                "model": "test-model",
                "max_tokens": 100,
                "top_p": 1,
                "top_k": 1,
                "presence_penalty": 0,
                "frequency_penalty": 0,
                "temperature": 0.7,
            },
            "tx_loop_breaker_count": 3,
            "tools_for_mech": {},
            "summon_cooldown_seconds": 86400,
            "store_path": "/tmp/test",
            "heart_cooldown_hours": 24,
            "is_memecoin_logic_enabled": True,
            "genai_api_key": "test_key",
            "x402_payment_requirements": {},
            "lifi_quote_to_amount_url": "https://lifi.example.com",
            "base_ledger_rpc": "https://rpc.example.com",
            "use_x402": False,
            "stop_posting_if_staking_kpi_met": False,
        }

    def test_init_sets_all_attributes(self) -> None:
        """Test that Params.__init__ sets all expected attributes."""
        kwargs = self._make_kwargs()
        with patch.object(MechParams, "__init__", return_value=None):
            instance = Params.__new__(Params)
            Params.__init__(instance, **kwargs)

            assert instance.service_endpoint == "http://localhost:8000"
            assert instance.minimum_gas_balance == 0.001
            assert instance.min_feedback_replies == 1
            assert instance.persona == "test persona"
            assert instance.home_chain_id == "base"
            assert instance.skip_engagement is False
            assert instance.tx_loop_breaker_count == 3
            assert instance.summon_cooldown_seconds == 86400
            assert instance.heart_cooldown_hours == 24
            assert instance.store_path == "/tmp/test"
            assert instance.is_memecoin_logic_enabled is True
            assert instance.genai_api_key == "test_key"
            assert instance.use_x402 is False
            assert instance.stop_posting_if_staking_kpi_met is False
            assert isinstance(
                instance.alternative_model_for_tweets, AlternativeModelForTweets
            )
            assert instance.alternative_model_for_tweets.api_key == "fw_test_key"
            assert instance.fireworks_api_key == "fw_test_key"

    def test_init_fireworks_key_none(self) -> None:
        """Test that Params.__init__ handles None fireworks_api_key."""
        kwargs = self._make_kwargs()
        kwargs.pop("fireworks_api_key")  # use default (None via kwargs.get)
        with patch.object(MechParams, "__init__", return_value=None):
            instance = Params.__new__(Params)
            Params.__init__(instance, **kwargs)

            assert instance.fireworks_api_key is None
            assert instance.alternative_model_for_tweets.use is False

    def test_init_store_path_default(self) -> None:
        """Test that store_path defaults to empty string when not provided."""
        kwargs = self._make_kwargs()
        kwargs.pop("store_path")
        with patch.object(MechParams, "__init__", return_value=None):
            instance = Params.__new__(Params)
            Params.__init__(instance, **kwargs)
            assert instance.store_path == ""
