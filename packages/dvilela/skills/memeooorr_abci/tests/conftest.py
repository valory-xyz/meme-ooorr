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

# pylint: disable=too-many-instance-attributes,assigning-non-slot,unused-argument,import-outside-toplevel

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
from unittest.mock import MagicMock, PropertyMock

from packages.dvilela.connections.kv_store.connection import (
    PUBLIC_ID as KV_STORE_CONNECTION_PUBLIC_ID,
)
from packages.dvilela.connections.tweepy.connection import (
    PUBLIC_ID as TWEEPY_CONNECTION_PUBLIC_ID,
)
from packages.dvilela.protocols.kv_store.message import KvStoreMessage
from packages.dvilela.skills.memeooorr_abci.rounds import Event, SynchronizedData
from packages.valory.protocols.srr.message import SrrMessage
from packages.valory.skills.abstract_round_abci.base import AbciAppDB
from packages.valory.skills.abstract_round_abci.test_tools.base import (
    FSMBehaviourBaseCase,
)
from packages.valory.skills.abstract_round_abci.test_tools.rounds import (
    BaseCollectSameUntilThresholdRoundTest,
)
from packages.valory.skills.agent_db_abci.twitter_models import TwitterPost

PACKAGE_DIR = Path(__file__).parent.parent
# The chained skill is non-abstract and includes all dependent models/handlers.
# BaseSkillTestCase requires a non-abstract skill to load standalone.
CHAINED_SKILL_DIR = PACKAGE_DIR.parent / "memeooorr_chained_abci"

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


# --- Framework base classes for round tests ---


class MemeooorrRoundTestBase(
    BaseCollectSameUntilThresholdRoundTest
):  # pylint: disable=too-few-public-methods
    """Base test class for memeooorr round tests."""

    _synchronized_data_class = SynchronizedData
    _event_class = Event
    context_mock: MagicMock

    def setup_method(self) -> None:
        """Setup test method."""
        super().setup_method()
        self.context_mock = MagicMock()
        self.context_mock.params.stop_posting_if_staking_kpi_met = False
        self.context_mock.params.tx_loop_breaker_count = 3


# --- Framework base class for behaviour tests ---


class MemeooorrFSMBehaviourBaseCase(FSMBehaviourBaseCase):
    """Base test class for memeooorr behaviour tests using FSMBehaviourBaseCase."""

    path_to_skill = CHAINED_SKILL_DIR
    _store_path_dir: str

    @classmethod
    def setup_class(cls, **kwargs: Any) -> None:
        """Setup the test class, patching store_path validation for test env."""
        import os
        import tempfile

        cls._store_path_dir = tempfile.mkdtemp()
        # Patch the skill config to use our temp dir instead of /data
        # We need this before the skill loads since Params.__init__ validates
        from unittest.mock import patch as _patch

        original_isdir = os.path.isdir
        original_access = os.access

        def _isdir(p: str) -> bool:
            if p == "/data":
                return True
            return original_isdir(p)

        def _access(p: str, mode: int) -> bool:
            if p == "/data":
                return True
            return original_access(p, mode)

        with (
            _patch("os.path.isdir", side_effect=_isdir),
            _patch("os.access", side_effect=_access),
        ):
            super().setup_class(**kwargs)

        # Override store_path and setup params for test env
        test_addr = "0x0000000000000000000000000000000000000000"
        cls.behaviour.context.params.__dict__["store_path"] = cls._store_path_dir
        cls.behaviour.context.params.__dict__["setup_params"] = {
            "all_participants": [test_addr],
            "safe_contract_address": SAFE_ADDRESS,
            "consensus_threshold": 1,
        }

    @classmethod
    def teardown_class(cls) -> None:
        """Teardown the test class."""
        import shutil

        super().teardown_class()
        if hasattr(cls, "_store_path_dir"):
            shutil.rmtree(cls._store_path_dir, ignore_errors=True)

    def mock_kv_store_request(
        self, request_kwargs: Dict, response_kwargs: Dict
    ) -> None:
        """Mock a KV store protocol request/response cycle.

        :param request_kwargs: keyword arguments for request check.
        :param response_kwargs: keyword arguments for mock response.
        """
        self.assert_quantity_in_outbox(1)
        actual_message = self.get_message_from_outbox()
        assert actual_message is not None, "No message in outbox."
        has_attributes, error_str = self.message_has_attributes(
            actual_message=actual_message,
            message_type=KvStoreMessage,
            to=str(KV_STORE_CONNECTION_PUBLIC_ID),
            sender=str(self.skill.skill_context.skill_id),
            **request_kwargs,
        )
        assert has_attributes, error_str
        self.behaviour.act_wrapper()
        incoming_message = self.build_incoming_message(
            message_type=KvStoreMessage,
            dialogue_reference=(actual_message.dialogue_reference[0], "stub"),
            target=actual_message.message_id,
            message_id=-1,
            to=str(self.skill.skill_context.skill_id),
            sender=str(KV_STORE_CONNECTION_PUBLIC_ID),
            **response_kwargs,
        )
        self.skill.skill_context.handlers.kv_store.handle(incoming_message)
        self.behaviour.act_wrapper()

    def mock_srr_request(self, request_kwargs: Dict, response_kwargs: Dict) -> None:
        """Mock an SRR (Tweepy/Twikit) protocol request/response cycle.

        :param request_kwargs: keyword arguments for request check.
        :param response_kwargs: keyword arguments for mock response.
        """
        self.assert_quantity_in_outbox(1)
        actual_message = self.get_message_from_outbox()
        assert actual_message is not None, "No message in outbox."
        has_attributes, error_str = self.message_has_attributes(
            actual_message=actual_message,
            message_type=SrrMessage,
            to=str(TWEEPY_CONNECTION_PUBLIC_ID),
            sender=str(self.skill.skill_context.skill_id),
            **request_kwargs,
        )
        assert has_attributes, error_str
        self.behaviour.act_wrapper()
        incoming_message = self.build_incoming_message(
            message_type=SrrMessage,
            dialogue_reference=(actual_message.dialogue_reference[0], "stub"),
            target=actual_message.message_id,
            message_id=-1,
            to=str(self.skill.skill_context.skill_id),
            sender=str(TWEEPY_CONNECTION_PUBLIC_ID),
            **response_kwargs,
        )
        self.skill.skill_context.handlers.srr.handle(incoming_message)
        self.behaviour.act_wrapper()

    def fast_forward_to_behaviour_with_data(
        self,
        behaviour_cls: Any,
        data: Dict[str, Any],
    ) -> None:
        """Convenience: fast_forward with SynchronizedData built from dict."""
        self.fast_forward_to_behaviour(
            self.behaviour,
            behaviour_cls.auto_behaviour_id(),
            SynchronizedData(AbciAppDB(setup_data=AbciAppDB.data_to_lists(data))),
        )


# --- Mock helpers for behaviour tests (custom protocols: KV store, SRR) ---


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
    my_agent: Optional[MagicMock] = None,
    active_agents: Optional[list] = None,
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
    if my_agent is not None:
        agents_fun_db.my_agent = my_agent
    else:
        agent = MagicMock()
        agent.twitter_username = "test_user"
        agent.twitter_user_id = "12345"
        agent.posts = []
        agents_fun_db.my_agent = agent
    agents_fun_db.get_active_agents = MagicMock(return_value=active_agents or [])
    context.agents_fun_db = agents_fun_db

    return context


def make_mock_my_agent(
    posts: Optional[List] = None, twitter_username: str = "test_agent"
) -> MagicMock:
    """Create mock AgentsFunAgent my_agent."""
    agent = MagicMock()
    agent.posts = posts if posts is not None else []
    agent.twitter_username = twitter_username
    agent.add_interaction = MagicMock()
    return agent


def make_twitter_post(
    tweet_id: str = "123",
    text: str = "test tweet",
    timestamp: Optional[datetime] = None,
    reply_to_tweet_id: Optional[str] = None,
) -> TwitterPost:
    """Create a TwitterPost object."""
    return TwitterPost(
        tweet_id=tweet_id,
        text=text,
        timestamp=timestamp or datetime.now(timezone.utc),
        reply_to_tweet_id=reply_to_tweet_id,
    )


def run_generator(gen: Generator, yields: Optional[list] = None) -> Any:
    """Run a generator to completion, returning its return value."""
    yields = yields or []
    idx = 0
    try:
        next(gen)
        while True:
            send_val = yields[idx] if idx < len(yields) else None
            idx += 1
            gen.send(send_val)
    except StopIteration as e:
        return e.value
