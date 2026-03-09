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

"""Shared test fixtures for memeooorr_abci behaviour tests."""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, PropertyMock

import pytest


SENDER = "test_agent_address"
SAFE_ADDRESS = "0x" + "a" * 40
STAKING_TOKEN_ADDRESS = "0x" + "b" * 40
ACTIVITY_CHECKER_ADDRESS = "0x" + "c" * 40
MEME_FACTORY_ADDRESS_BASE = "0x" + "d" * 40
MEME_FACTORY_ADDRESS_CELO = "0x" + "e" * 40
SERVICE_REGISTRY_ADDRESS_BASE = "0x" + "f" * 40
SERVICE_REGISTRY_ADDRESS_CELO = "0x" + "1" * 40
OLAS_TOKEN_ADDRESS_BASE = "0x" + "2" * 40
OLAS_TOKEN_ADDRESS_CELO = "0x" + "3" * 40
MECH_MARKETPLACE_ADDRESS = "0x" + "4" * 40


@dataclass
class MockAlternativeModelForTweets:
    """Mock alternative model configuration."""

    use: bool = False
    url: str = "https://api.example.com"
    api_key: Optional[str] = None
    model: str = "test-model"
    max_tokens: int = 100
    top_p: int = 1
    top_k: int = 1
    presence_penalty: int = 0
    frequency_penalty: int = 0
    temperature: float = 0.7


@dataclass
class MockMechMarketplaceConfig:
    """Mock MechMarketplaceConfig."""

    mech_marketplace_address: str = MECH_MARKETPLACE_ADDRESS


def make_mock_params(**overrides: Any) -> MagicMock:
    """Create a mock Params object with sensible defaults."""
    params = MagicMock()
    defaults = {
        "home_chain_id": "base",
        "minimum_gas_balance": 0.001,
        "min_feedback_replies": 1,
        "meme_factory_address_base": MEME_FACTORY_ADDRESS_BASE,
        "meme_factory_address_celo": MEME_FACTORY_ADDRESS_CELO,
        "olas_token_address_base": OLAS_TOKEN_ADDRESS_BASE,
        "olas_token_address_celo": OLAS_TOKEN_ADDRESS_CELO,
        "service_registry_address_base": SERVICE_REGISTRY_ADDRESS_BASE,
        "service_registry_address_celo": SERVICE_REGISTRY_ADDRESS_CELO,
        "persona": "test persona",
        "staking_token_contract_address": STAKING_TOKEN_ADDRESS,
        "activity_checker_contract_address": ACTIVITY_CHECKER_ADDRESS,
        "on_chain_service_id": 1,
        "meme_subgraph_url": "https://subgraph.example.com",
        "olas_subgraph_url": "https://olas-subgraph.example.com",
        "skip_engagement": False,
        "min_summon_amount_base": 0.01,
        "max_summon_amount_base": 0.1,
        "max_heart_amount_base": 0.05,
        "min_summon_amount_celo": 10,
        "max_summon_amount_celo": 100,
        "max_heart_amount_celo": 50,
        "tx_loop_breaker_count": 3,
        "summon_cooldown_seconds": 86400,
        "heart_cooldown_hours": 24,
        "store_path": "/tmp/test_store",
        "is_memecoin_logic_enabled": True,
        "alternative_model_for_tweets": MockAlternativeModelForTweets(),
        "mech_marketplace_config": MockMechMarketplaceConfig(),
        "tools_for_mech": {},
        "service_endpoint": "http://localhost:8000",
        "meme_factory_deployment_block_base": 0,
        "meme_factory_deployment_block_celo": 0,
        "stop_posting_if_staking_kpi_met": False,
        "genai_api_key": "test_key",
    }
    defaults.update(overrides)
    for key, value in defaults.items():
        setattr(params, key, value)
    return params


def make_mock_synchronized_data(**overrides: Any) -> MagicMock:
    """Create a mock SynchronizedData object with sensible defaults."""
    sync_data = MagicMock()
    defaults: Dict[str, Any] = {
        "safe_contract_address": SAFE_ADDRESS,
        "persona": "test persona",
        "meme_coins": [],
        "pending_tweet": [],
        "feedback": [],
        "token_action": {},
        "final_tx_hash": None,
        "tx_submitter": "",
        "is_staking_kpi_met": None,
        "mech_requests": [],
        "mech_responses": [],
        "tx_loop_count": 0,
        "mech_for_twitter": False,
        "failed_mech": False,
        "check_funds_count": 0,
        "heart_cooldown_hours": 24,
        "summon_cooldown_seconds": 86400,
        "agent_details": {},
    }
    defaults.update(overrides)
    for key, value in defaults.items():
        type(sync_data).__dict__.get(key)  # ensure property mocking
        setattr(type(sync_data), key, PropertyMock(return_value=value))
    return sync_data


def make_mock_context(
    params: Optional[MagicMock] = None,
    synchronized_data: Optional[MagicMock] = None,
) -> MagicMock:
    """Create a mock context object."""
    context = MagicMock()
    context.agent_address = SENDER
    context.logger = logging.getLogger("test_memeooorr")
    context.params = params or make_mock_params()
    context.benchmark_tool.measure.return_value.__enter__ = MagicMock()
    context.benchmark_tool.measure.return_value.__exit__ = MagicMock()

    # State mock
    state = MagicMock()
    state.twitter_username = "test_user"
    state.twitter_id = "12345"
    state.twitter_display_name = "Test User"

    # round_sequence mock for get_sync_timestamp
    mock_timestamp = MagicMock()
    mock_timestamp.timestamp.return_value = 1700000000.0
    state.round_sequence.last_round_transition_timestamp = mock_timestamp
    state.env_var_status = {"needs_update": False, "env_vars": {}}

    context.state = state

    # agents_fun_db mock
    agents_fun_db = MagicMock()
    my_agent = MagicMock()
    my_agent.twitter_username = "test_user"
    my_agent.twitter_user_id = "12345"
    my_agent.posts = []
    agents_fun_db.my_agent = my_agent
    context.agents_fun_db = agents_fun_db

    return context
