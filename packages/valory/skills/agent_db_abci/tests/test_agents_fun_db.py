# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023-2025 Valory AG
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

"""Tests for agents_fun_db.py."""

# pylint: disable=W0212,W0613,R0903,E1136

from datetime import datetime, timedelta, timezone
from typing import Any, Generator
from unittest.mock import MagicMock

import pytest

from packages.valory.skills.agent_db_abci.agent_db_client import AgentDBClient
from packages.valory.skills.agent_db_abci.agent_db_models import (
    AgentInstance,
    AgentType,
    AttributeDefinition,
    AttributeInstance,
)
from packages.valory.skills.agent_db_abci.agents_fun_db import (
    AGENT_TYPE_DESCRIPTION,
    AgentsFunAgent,
    AgentsFunDatabase,
    MEMEOOORR,
    REQUIRED_AGENT_TYPE_ATTRIBUTE_DEFINITIONS,
)
from packages.valory.skills.agent_db_abci.twitter_models import (
    TwitterFollow,
    TwitterLike,
    TwitterPost,
    TwitterRewtweet,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NOW = datetime.now(timezone.utc)
TIMESTAMP = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

AGENT_TYPE_OBJ = AgentType(type_id=1, type_name="memeooorr", description="desc")

ATTR_DEF_INTERACTIONS = AttributeDefinition(
    attr_def_id=3,
    type_id=1,
    attr_name="twitter_interactions",
    data_type="json",
    is_required=False,
    default_value="{}",
)


def _exhaust(gen: Generator) -> Any:
    """Drive a generator to completion and return its return value."""
    try:
        while True:
            next(gen)
    except StopIteration as exc:
        return exc.value


def make_agent_instance(
    agent_id: int = 1,
    type_id: int = 1,
    agent_name: str = "agent-1",
    eth_address: str = "0xabc",
) -> AgentInstance:
    """Create a test AgentInstance."""
    return AgentInstance(
        agent_id=agent_id,
        type_id=type_id,
        agent_name=agent_name,
        eth_address=eth_address,
        created_at=TIMESTAMP,
    )


def make_mock_client() -> MagicMock:
    """Create a mock AgentDBClient."""
    client = MagicMock(spec=AgentDBClient)
    client.address = "0xabc"
    client.logger = MagicMock()
    return client


def _gen_return(value):
    """Create a generator that returns value immediately."""

    def gen(*args, **kwargs):
        return value
        yield  # noqa: E501

    return gen


# --- Section: constants ---


class TestConstants:
    """Tests for module-level constants."""

    def test_memeooorr_constant(self) -> None:
        """Test MEMEOOORR constant."""
        assert MEMEOOORR == "memeooorr"

    def test_agent_type_description(self) -> None:
        """Test AGENT_TYPE_DESCRIPTION constant."""
        assert AGENT_TYPE_DESCRIPTION == "Agent type for Memeooorr skill"

    def test_required_attribute_definitions_count(self) -> None:
        """Test the count of required attribute definitions."""
        assert len(REQUIRED_AGENT_TYPE_ATTRIBUTE_DEFINITIONS) == 3

    def test_required_attribute_definitions_names(self) -> None:
        """Test that required attribute definitions have expected names."""
        names = {ad.attr_name for ad in REQUIRED_AGENT_TYPE_ATTRIBUTE_DEFINITIONS}
        assert names == {"twitter_username", "twitter_user_id", "twitter_interactions"}

    def test_required_attribute_definitions_types(self) -> None:
        """Test data types of required attribute definitions."""
        type_map = {
            ad.attr_name: ad.data_type
            for ad in REQUIRED_AGENT_TYPE_ATTRIBUTE_DEFINITIONS
        }
        assert type_map["twitter_username"] == "string"
        assert type_map["twitter_user_id"] == "string"
        assert type_map["twitter_interactions"] == "json"


# ---------------------------------------------------------------------------
# Tests: AgentsFunAgent construction / non-generator
# ---------------------------------------------------------------------------


class TestAgentsFunAgentBasic:
    """Tests for AgentsFunAgent construction and non-generator methods."""

    def test_construction(self) -> None:
        """Test basic construction."""
        client = make_mock_client()
        agent_instance = make_agent_instance()
        agent = AgentsFunAgent(client=client, agent_instance=agent_instance)
        assert agent.client is client
        assert agent.agent_instance is agent_instance
        assert agent.twitter_username is None
        assert agent.twitter_user_id is None
        assert agent.posts == []
        assert agent.likes == []
        assert agent.retweets == []
        assert agent.follows == []
        assert agent.loaded is False

    def test_action_to_class_mapping(self) -> None:
        """Test the action_to_class mapping."""
        assert AgentsFunAgent.action_to_class["post"] is TwitterPost
        assert AgentsFunAgent.action_to_class["retweet"] is TwitterRewtweet
        assert AgentsFunAgent.action_to_class["follow"] is TwitterFollow
        assert AgentsFunAgent.action_to_class["like"] is TwitterLike
        assert len(AgentsFunAgent.action_to_class) == 4

    def test_str_representation_unloaded(self) -> None:
        """Test string representation when not loaded."""
        agent = AgentsFunAgent(
            client=make_mock_client(),
            agent_instance=make_agent_instance(agent_id=42),
        )
        result = str(agent)
        assert "42" in result
        assert "loaded=False" in result

    def test_str_representation_loaded(self) -> None:
        """Test string representation when loaded with username."""
        agent = AgentsFunAgent(
            client=make_mock_client(),
            agent_instance=make_agent_instance(agent_id=7),
        )
        agent.loaded = True
        agent.twitter_username = "testuser"
        result = str(agent)
        assert "7" in result
        assert "loaded=True" in result
        assert "@testuser" in result


# --- Section: AgentsFunAgent.delete (generator) ---


class TestAgentsFunAgentDelete:
    """Tests for AgentsFunAgent.delete generator."""

    def test_delete(self) -> None:
        """Test delete delegates to client."""
        client = make_mock_client()
        client.delete_agent_instance = _gen_return(None)
        agent_instance = make_agent_instance()
        agent = AgentsFunAgent(client=client, agent_instance=agent_instance)
        _exhaust(agent.delete())


# --- Section: AgentsFunAgent.load (generator) ---


class TestAgentsFunAgentLoad:
    """Tests for AgentsFunAgent.load generator."""

    def test_load_username_and_user_id(self) -> None:
        """Test load sets twitter_username and twitter_user_id."""
        client = make_mock_client()
        client.get_all_agent_instance_attributes_parsed = _gen_return(
            [
                {"attr_name": "twitter_username", "attr_value": "alice"},
                {"attr_name": "twitter_user_id", "attr_value": "12345"},
            ]
        )
        agent = AgentsFunAgent(client=client, agent_instance=make_agent_instance())
        _exhaust(agent.load())
        assert agent.twitter_username == "alice"
        assert agent.twitter_user_id == "12345"
        assert agent.loaded is True

    def test_load_post_interaction(self) -> None:
        """Test load parses post interactions."""
        client = make_mock_client()
        client.get_all_agent_instance_attributes_parsed = _gen_return(
            [
                {
                    "attr_name": "twitter_interactions",
                    "attr_value": {
                        "action": "post",
                        "timestamp": "2025-01-15T12:00:00Z",
                        "details": {"tweet_id": "t1", "text": "hello"},
                    },
                },
            ]
        )
        agent = AgentsFunAgent(client=client, agent_instance=make_agent_instance())
        _exhaust(agent.load())
        assert len(agent.posts) == 1
        assert agent.posts[0].tweet_id == "t1"
        assert agent.loaded is True

    def test_load_retweet_interaction(self) -> None:
        """Test load parses retweet interactions."""
        client = make_mock_client()
        client.get_all_agent_instance_attributes_parsed = _gen_return(
            [
                {
                    "attr_name": "twitter_interactions",
                    "attr_value": {
                        "action": "retweet",
                        "timestamp": "2025-01-15T12:00:00Z",
                        "details": {"tweet_id": "rt1"},
                    },
                },
            ]
        )
        agent = AgentsFunAgent(client=client, agent_instance=make_agent_instance())
        _exhaust(agent.load())
        assert len(agent.retweets) == 1
        assert agent.retweets[0].tweet_id == "rt1"

    def test_load_like_interaction(self) -> None:
        """Test load parses like interactions."""
        client = make_mock_client()
        client.get_all_agent_instance_attributes_parsed = _gen_return(
            [
                {
                    "attr_name": "twitter_interactions",
                    "attr_value": {
                        "action": "like",
                        "timestamp": "2025-01-15T12:00:00Z",
                        "details": {"tweet_id": "lk1"},
                    },
                },
            ]
        )
        agent = AgentsFunAgent(client=client, agent_instance=make_agent_instance())
        _exhaust(agent.load())
        assert len(agent.likes) == 1
        assert agent.likes[0].tweet_id == "lk1"

    def test_load_follow_interaction(self) -> None:
        """Test load parses follow interactions."""
        client = make_mock_client()
        client.get_all_agent_instance_attributes_parsed = _gen_return(
            [
                {
                    "attr_name": "twitter_interactions",
                    "attr_value": {
                        "action": "follow",
                        "timestamp": "2025-01-15T12:00:00Z",
                        "details": {"username": "bob"},
                    },
                },
            ]
        )
        agent = AgentsFunAgent(client=client, agent_instance=make_agent_instance())
        _exhaust(agent.load())
        assert len(agent.follows) == 1
        assert agent.follows[0].username == "bob"

    def test_load_unknown_action_raises(self) -> None:
        """Test load raises for unknown action type."""
        client = make_mock_client()
        client.get_all_agent_instance_attributes_parsed = _gen_return(
            [
                {
                    "attr_name": "twitter_interactions",
                    "attr_value": {
                        "action": "unknown_action",
                        "timestamp": "2025-01-15T12:00:00Z",
                        "details": {},
                    },
                },
            ]
        )
        agent = AgentsFunAgent(client=client, agent_instance=make_agent_instance())
        with pytest.raises(ValueError, match="Unknown Twitter action"):
            _exhaust(agent.load())

    def test_load_sorts_by_timestamp(self) -> None:
        """Test load sorts interactions by timestamp."""
        client = make_mock_client()
        client.get_all_agent_instance_attributes_parsed = _gen_return(
            [
                {
                    "attr_name": "twitter_interactions",
                    "attr_value": {
                        "action": "post",
                        "timestamp": "2025-01-16T12:00:00Z",
                        "details": {"tweet_id": "t2", "text": "second"},
                    },
                },
                {
                    "attr_name": "twitter_interactions",
                    "attr_value": {
                        "action": "post",
                        "timestamp": "2025-01-15T12:00:00Z",
                        "details": {"tweet_id": "t1", "text": "first"},
                    },
                },
            ]
        )
        agent = AgentsFunAgent(client=client, agent_instance=make_agent_instance())
        _exhaust(agent.load())
        assert agent.posts[0].tweet_id == "t1"
        assert agent.posts[1].tweet_id == "t2"

    def test_load_mixed_interactions(self) -> None:
        """Test load separates different interaction types."""
        client = make_mock_client()
        client.get_all_agent_instance_attributes_parsed = _gen_return(
            [
                {
                    "attr_name": "twitter_interactions",
                    "attr_value": {
                        "action": "post",
                        "timestamp": "2025-01-15T12:00:00Z",
                        "details": {"tweet_id": "t1", "text": "hi"},
                    },
                },
                {
                    "attr_name": "twitter_interactions",
                    "attr_value": {
                        "action": "like",
                        "timestamp": "2025-01-15T13:00:00Z",
                        "details": {"tweet_id": "lk1"},
                    },
                },
                {
                    "attr_name": "twitter_interactions",
                    "attr_value": {
                        "action": "retweet",
                        "timestamp": "2025-01-15T14:00:00Z",
                        "details": {"tweet_id": "rt1"},
                    },
                },
                {
                    "attr_name": "twitter_interactions",
                    "attr_value": {
                        "action": "follow",
                        "timestamp": "2025-01-15T15:00:00Z",
                        "details": {"username": "bob"},
                    },
                },
            ]
        )
        agent = AgentsFunAgent(client=client, agent_instance=make_agent_instance())
        _exhaust(agent.load())
        assert len(agent.posts) == 1
        assert len(agent.likes) == 1
        assert len(agent.retweets) == 1
        assert len(agent.follows) == 1

    def test_load_empty_attributes(self) -> None:
        """Test load with empty attributes list."""
        client = make_mock_client()
        client.get_all_agent_instance_attributes_parsed = _gen_return([])
        agent = AgentsFunAgent(client=client, agent_instance=make_agent_instance())
        _exhaust(agent.load())
        assert agent.loaded is True
        assert agent.posts == []


# --- Section: AgentsFunAgent.add_interaction (generator) ---


class TestAgentsFunAgentAddInteraction:
    """Tests for AgentsFunAgent.add_interaction generator."""

    def test_add_interaction_raises_for_unknown_action(self) -> None:
        """Test raises for unknown action."""
        client = make_mock_client()
        agent = AgentsFunAgent(client=client, agent_instance=make_agent_instance())
        mock_interaction = MagicMock()
        mock_interaction.action = "bad_action"
        gen = agent.add_interaction(mock_interaction)
        with pytest.raises(ValueError, match="Unknown Twitter action"):
            gen.send(None)

    def test_add_interaction_raises_when_attr_def_not_found(self) -> None:
        """Test raises when attribute definition is not found."""
        client = make_mock_client()
        client.get_attribute_definition_by_name = _gen_return(None)
        agent = AgentsFunAgent(client=client, agent_instance=make_agent_instance())
        post = TwitterPost(tweet_id="t1", text="hello", timestamp=NOW)
        gen = agent.add_interaction(post)
        with pytest.raises(ValueError, match="Attribute definition not found"):
            _exhaust(gen)

    def test_add_interaction_success(self) -> None:
        """Test successful add_interaction."""
        attr_instance = AttributeInstance(
            attribute_id=1,
            attr_def_id=3,
            agent_id=1,
            last_updated=NOW,
            string_value=None,
            integer_value=None,
            float_value=None,
            boolean_value=None,
            date_value=None,
            json_value={"action": "post"},
        )
        client = make_mock_client()
        client.get_attribute_definition_by_name = _gen_return(ATTR_DEF_INTERACTIONS)
        client.create_attribute_instance = _gen_return(attr_instance)
        agent = AgentsFunAgent(client=client, agent_instance=make_agent_instance())
        post = TwitterPost(tweet_id="t1", text="hello", timestamp=NOW)
        result = _exhaust(agent.add_interaction(post))
        assert result is attr_instance


# --- Section: AgentsFunAgent.update_twitter_details (generator) ---


class TestAgentsFunAgentUpdateTwitterDetails:
    """Tests for AgentsFunAgent.update_twitter_details generator."""

    def test_update_twitter_details(self) -> None:
        """Test update_twitter_details calls update_or_create for both fields."""
        call_args = []

        def mock_update(attr_name, attr_value):
            call_args.append((attr_name, attr_value))
            return True
            yield  # noqa: E501

        client = make_mock_client()
        client.update_or_create_agent_attribute = mock_update
        agent = AgentsFunAgent(client=client, agent_instance=make_agent_instance())
        agent.twitter_username = "alice"
        agent.twitter_user_id = "123"
        _exhaust(agent.update_twitter_details())
        assert ("twitter_username", "alice") in call_args
        assert ("twitter_user_id", "123") in call_args


# ---------------------------------------------------------------------------
# Tests: AgentsFunDatabase construction / non-generator
# ---------------------------------------------------------------------------


class TestAgentsFunDatabase:
    """Tests for AgentsFunDatabase."""

    def _make_db(self) -> AgentsFunDatabase:
        """Create an AgentsFunDatabase with mocked Model.__init__."""
        db = AgentsFunDatabase.__new__(AgentsFunDatabase)
        db.client = None
        db.agent_type = None
        db.agents = []
        db.my_agent = None
        db.logger = None
        return db

    def test_construction_state(self) -> None:
        """Test initial state after construction."""
        db = self._make_db()
        assert db.client is None
        assert db.agent_type is None
        assert not db.agents
        assert db.my_agent is None

    def test_initialize(self) -> None:
        """Test initialize method."""
        db = self._make_db()
        client = make_mock_client()
        db.initialize(client)
        assert db.client is client
        assert db.logger is client.logger
        assert client.agent_type_name == MEMEOOORR
        assert client.agent_name_template == "memeooorr-agent-{address}"

    def test_str_empty(self) -> None:
        """Test string representation with no agents."""
        db = self._make_db()
        assert str(db) == "AgentsFunDatabase with 0 agents"

    def test_str_with_agents(self) -> None:
        """Test string representation with agents."""
        db = self._make_db()
        db.agents = [MagicMock(), MagicMock(), MagicMock()]
        assert str(db) == "AgentsFunDatabase with 3 agents"


# --- Section: AgentsFunDatabase.load (generator) ---


class TestAgentsFunDatabaseLoad:
    """Tests for AgentsFunDatabase.load generator."""

    def _make_db_with_client(self) -> AgentsFunDatabase:
        """Create a database with a mock client."""
        db = AgentsFunDatabase.__new__(AgentsFunDatabase)
        db.client = make_mock_client()
        db.agent_type = None
        db.agents = []
        db.my_agent = None
        db.logger = db.client.logger
        return db

    def test_load_sets_agent_type(self) -> None:
        """Test load fetches agent type."""
        db = self._make_db_with_client()
        db.client._ensure_agent_type_definition = _gen_return(None)
        db.client._ensure_agent_type_attribute_definition = _gen_return(None)
        db.client._ensure_agent_instance = _gen_return(None)
        db.client.get_agent_type_by_type_name = _gen_return(AGENT_TYPE_OBJ)
        db.client.get_agent_instances_by_type_id = _gen_return([])
        db.client.agent = None
        _exhaust(db.load())
        assert db.agent_type is AGENT_TYPE_OBJ

    def test_load_no_agent_type_logs_error(self) -> None:
        """Test load logs error when agent type cannot be fetched."""
        db = self._make_db_with_client()
        db.client._ensure_agent_type_definition = _gen_return(None)
        db.client._ensure_agent_type_attribute_definition = _gen_return(None)
        db.client._ensure_agent_instance = _gen_return(None)
        db.client.get_agent_type_by_type_name = _gen_return(None)
        _exhaust(db.load())
        db.logger.error.assert_called_once()

    def test_load_with_agents(self) -> None:
        """Test load populates agents list."""
        ai = make_agent_instance(eth_address="0xabc")
        db = self._make_db_with_client()
        db.client._ensure_agent_type_definition = _gen_return(None)
        db.client._ensure_agent_type_attribute_definition = _gen_return(None)
        db.client._ensure_agent_instance = _gen_return(None)
        db.client.get_agent_type_by_type_name = _gen_return(AGENT_TYPE_OBJ)
        db.client.get_agent_instances_by_type_id = _gen_return([ai])
        db.client.get_all_agent_instance_attributes_parsed = _gen_return(
            [
                {"attr_name": "twitter_username", "attr_value": "alice"},
            ]
        )
        db.client.agent = None
        _exhaust(db.load())
        assert len(db.agents) == 1
        assert db.agents[0].twitter_username == "alice"
        # eth_address matches client.address so my_agent is set
        assert db.my_agent is db.agents[0]

    def test_load_my_agent_fallback(self) -> None:
        """Test my_agent fallback when address not in agent_instances."""
        ai = make_agent_instance(eth_address="0xOTHER")
        db = self._make_db_with_client()
        db.client._ensure_agent_type_definition = _gen_return(None)
        db.client._ensure_agent_type_attribute_definition = _gen_return(None)
        db.client._ensure_agent_instance = _gen_return(None)
        db.client.get_agent_type_by_type_name = _gen_return(AGENT_TYPE_OBJ)
        db.client.get_agent_instances_by_type_id = _gen_return([ai])
        db.client.get_all_agent_instance_attributes_parsed = _gen_return([])
        # client.agent is set (was registered), so fallback should work
        db.client.agent = make_agent_instance(eth_address="0xabc")
        _exhaust(db.load())
        assert db.my_agent is not None
        assert db.my_agent.agent_instance.eth_address == "0xabc"
        # Should be appended to agents list
        assert len(db.agents) == 2

    def test_load_no_my_agent_when_client_agent_none(self) -> None:
        """Test my_agent stays None when client.agent is None and address not found."""
        ai = make_agent_instance(eth_address="0xOTHER")
        db = self._make_db_with_client()
        db.client._ensure_agent_type_definition = _gen_return(None)
        db.client._ensure_agent_type_attribute_definition = _gen_return(None)
        db.client._ensure_agent_instance = _gen_return(None)
        db.client.get_agent_type_by_type_name = _gen_return(AGENT_TYPE_OBJ)
        db.client.get_agent_instances_by_type_id = _gen_return([ai])
        db.client.get_all_agent_instance_attributes_parsed = _gen_return([])
        db.client.agent = None
        _exhaust(db.load())
        assert db.my_agent is None

    def test_load_uses_cached_agent_type(self) -> None:
        """Test load does not re-fetch agent type if already set."""
        db = self._make_db_with_client()
        db.agent_type = AGENT_TYPE_OBJ
        db.client._ensure_agent_type_definition = _gen_return(None)
        db.client._ensure_agent_type_attribute_definition = _gen_return(None)
        db.client._ensure_agent_instance = _gen_return(None)
        # get_agent_type_by_type_name should NOT be called since agent_type is set
        db.client.get_agent_type_by_type_name = _gen_return(None)
        db.client.get_agent_instances_by_type_id = _gen_return([])
        db.client.agent = None
        _exhaust(db.load())
        # agent_type should still be the original object
        assert db.agent_type is AGENT_TYPE_OBJ


# --- Section: AgentsFunDatabase.get_tweet_likes_number (generator) ---


class TestGetTweetLikesNumber:
    """Tests for get_tweet_likes_number."""

    def _make_db_with_agents(self, agents_data):
        """Create database with pre-configured agents."""
        db = AgentsFunDatabase.__new__(AgentsFunDatabase)
        db.client = None
        db.agent_type = None
        db.agents = []
        db.my_agent = None
        db.logger = MagicMock()

        for data in agents_data:
            agent = AgentsFunAgent.__new__(AgentsFunAgent)
            agent.client = MagicMock()
            agent.agent_instance = make_agent_instance(agent_id=data.get("id", 1))
            agent.loaded = data.get("loaded", True)
            agent.twitter_username = data.get("username")
            agent.twitter_user_id = None
            agent.posts = []
            agent.likes = data.get("likes", [])
            agent.retweets = data.get("retweets", [])
            agent.follows = []
            db.agents.append(agent)
        return db

    def test_no_likes(self) -> None:
        """Test returns 0 when no agent liked the tweet."""
        db = self._make_db_with_agents(
            [
                {"loaded": True, "likes": []},
            ]
        )
        result = _exhaust(db.get_tweet_likes_number("t1"))
        assert result == 0

    def test_one_like(self) -> None:
        """Test returns 1 when one agent liked the tweet."""
        like = TwitterLike(tweet_id="t1", timestamp=NOW)
        db = self._make_db_with_agents(
            [
                {"loaded": True, "likes": [like]},
            ]
        )
        result = _exhaust(db.get_tweet_likes_number("t1"))
        assert result == 1

    def test_multiple_agents_with_likes(self) -> None:
        """Test counts across multiple agents."""
        like1 = TwitterLike(tweet_id="t1", timestamp=NOW)
        like2 = TwitterLike(tweet_id="t1", timestamp=NOW)
        like_other = TwitterLike(tweet_id="t2", timestamp=NOW)
        db = self._make_db_with_agents(
            [
                {"loaded": True, "likes": [like1]},
                {"loaded": True, "likes": [like_other]},
                {"loaded": True, "likes": [like2]},
            ]
        )
        result = _exhaust(db.get_tweet_likes_number("t1"))
        assert result == 2

    def test_loads_unloaded_agent(self) -> None:
        """Test loads unloaded agent before checking likes."""
        like = TwitterLike(tweet_id="t1", timestamp=NOW)
        db = self._make_db_with_agents(
            [
                {"loaded": False, "likes": [like]},
            ]
        )
        # Mock load generator
        db.agents[0].load = _gen_return(None)
        # After load, set loaded=True
        original_load = db.agents[0].load

        def load_and_set():
            yield from original_load()
            db.agents[0].loaded = True

        db.agents[0].load = load_and_set
        # likes are already set, so counting should work
        result = _exhaust(db.get_tweet_likes_number("t1"))
        assert result == 1


# --- Section: AgentsFunDatabase.get_tweet_retweets_number (generator) ---


class TestGetTweetRetweetsNumber:
    """Tests for get_tweet_retweets_number."""

    def _make_db_with_agents(self, agents_data):
        db = AgentsFunDatabase.__new__(AgentsFunDatabase)
        db.client = None
        db.agent_type = None
        db.agents = []
        db.my_agent = None
        db.logger = MagicMock()
        for data in agents_data:
            agent = AgentsFunAgent.__new__(AgentsFunAgent)
            agent.client = MagicMock()
            agent.agent_instance = make_agent_instance()
            agent.loaded = data.get("loaded", True)
            agent.twitter_username = None
            agent.twitter_user_id = None
            agent.posts = []
            agent.likes = []
            agent.retweets = data.get("retweets", [])
            agent.follows = []
            db.agents.append(agent)
        return db

    def test_no_retweets(self) -> None:
        """Test returns 0 when no agent retweeted."""
        db = self._make_db_with_agents([{"loaded": True, "retweets": []}])
        result = _exhaust(db.get_tweet_retweets_number("t1"))
        assert result == 0

    def test_one_retweet(self) -> None:
        """Test returns 1."""
        rt = TwitterRewtweet(tweet_id="t1", timestamp=NOW)
        db = self._make_db_with_agents([{"loaded": True, "retweets": [rt]}])
        result = _exhaust(db.get_tweet_retweets_number("t1"))
        assert result == 1

    def test_loads_unloaded_agent(self) -> None:
        """Test loads unloaded agent before counting."""
        rt = TwitterRewtweet(tweet_id="t1", timestamp=NOW)
        db = self._make_db_with_agents([{"loaded": False, "retweets": [rt]}])
        db.agents[0].load = _gen_return(None)
        result = _exhaust(db.get_tweet_retweets_number("t1"))
        assert result == 1


# --- Section: AgentsFunDatabase.get_tweet_replies (generator) ---


class TestGetTweetReplies:
    """Tests for get_tweet_replies."""

    def _make_db_with_agents(self, agents_data):
        db = AgentsFunDatabase.__new__(AgentsFunDatabase)
        db.client = None
        db.agent_type = None
        db.agents = []
        db.my_agent = None
        db.logger = MagicMock()
        for data in agents_data:
            agent = AgentsFunAgent.__new__(AgentsFunAgent)
            agent.client = MagicMock()
            agent.agent_instance = make_agent_instance()
            agent.loaded = data.get("loaded", True)
            agent.twitter_username = None
            agent.twitter_user_id = None
            agent.posts = data.get("posts", [])
            agent.likes = []
            agent.retweets = []
            agent.follows = []
            db.agents.append(agent)
        return db

    def test_no_replies(self) -> None:
        """Test returns empty list when no replies."""
        db = self._make_db_with_agents([{"loaded": True, "posts": []}])
        result = _exhaust(db.get_tweet_replies("t1"))
        assert not result

    def test_one_reply(self) -> None:
        """Test returns reply post."""
        post = TwitterPost(
            tweet_id="r1",
            text="reply",
            timestamp=NOW,
            reply_to_tweet_id="t1",
        )
        db = self._make_db_with_agents([{"loaded": True, "posts": [post]}])
        result = _exhaust(db.get_tweet_replies("t1"))
        assert len(result) == 1
        assert result[0].tweet_id == "r1"

    def test_non_reply_post_excluded(self) -> None:
        """Test non-reply posts are not returned."""
        post = TwitterPost(tweet_id="p1", text="not a reply", timestamp=NOW)
        db = self._make_db_with_agents([{"loaded": True, "posts": [post]}])
        result = _exhaust(db.get_tweet_replies("t1"))
        assert not result

    def test_loads_unloaded_agent(self) -> None:
        """Test loads unloaded agent before checking."""
        post = TwitterPost(
            tweet_id="r1",
            text="reply",
            timestamp=NOW,
            reply_to_tweet_id="t1",
        )
        db = self._make_db_with_agents([{"loaded": False, "posts": [post]}])
        db.agents[0].load = _gen_return(None)
        result = _exhaust(db.get_tweet_replies("t1"))
        assert len(result) == 1


# --- Section: AgentsFunDatabase.get_tweet_feedback (generator) ---


class TestGetTweetFeedback:
    """Tests for get_tweet_feedback."""

    def test_feedback_aggregation(self) -> None:
        """Test get_tweet_feedback aggregates likes, retweets, and replies."""
        db = AgentsFunDatabase.__new__(AgentsFunDatabase)
        db.client = None
        db.agent_type = None
        db.agents = []
        db.my_agent = None
        db.logger = MagicMock()

        like = TwitterLike(tweet_id="t1", timestamp=NOW)
        rt = TwitterRewtweet(tweet_id="t1", timestamp=NOW)
        reply = TwitterPost(
            tweet_id="r1",
            text="reply",
            timestamp=NOW,
            reply_to_tweet_id="t1",
        )

        agent = AgentsFunAgent.__new__(AgentsFunAgent)
        agent.client = MagicMock()
        agent.agent_instance = make_agent_instance()
        agent.loaded = True
        agent.twitter_username = None
        agent.twitter_user_id = None
        agent.posts = [reply]
        agent.likes = [like]
        agent.retweets = [rt]
        agent.follows = []
        db.agents.append(agent)

        result = _exhaust(db.get_tweet_feedback("t1"))
        assert result["likes"] == 1
        assert result["retweets"] == 1
        assert len(result["replies"]) == 1


# --- Section: get_active_agents (non-generator) ---


class TestGetActiveAgents:
    """Tests for get_active_agents (non-generator method)."""

    def _make_db_with_agents(self, agent_configs: list) -> AgentsFunDatabase:
        """Create a database with configured agents."""
        db = AgentsFunDatabase.__new__(AgentsFunDatabase)
        db.client = None
        db.agent_type = None
        db.agents = []
        db.my_agent = None
        db.logger = MagicMock()

        for cfg in agent_configs:
            agent = AgentsFunAgent.__new__(AgentsFunAgent)
            agent.client = MagicMock()
            agent.agent_instance = make_agent_instance(agent_id=cfg.get("agent_id", 1))
            agent.loaded = cfg["loaded"]
            agent.twitter_username = cfg.get("twitter_username")
            agent.twitter_user_id = cfg.get("twitter_user_id")
            agent.posts = []
            agent.likes = []
            agent.retweets = []
            agent.follows = []

            if cfg.get("last_post_timestamp"):
                post = TwitterPost(
                    tweet_id="1",
                    text="test",
                    timestamp=cfg["last_post_timestamp"],
                )
                agent.posts = [post]

            db.agents.append(agent)
        return db

    def test_empty_agents(self) -> None:
        """Test with no agents."""
        db = self._make_db_with_agents([])
        assert not db.get_active_agents()

    def test_unloaded_agent_skipped(self) -> None:
        """Test that unloaded agents are skipped."""
        db = self._make_db_with_agents(
            [
                {"loaded": False, "twitter_username": "u", "last_post_timestamp": NOW},
            ]
        )
        assert not db.get_active_agents()

    def test_no_posts_skipped(self) -> None:
        """Test that agents with no posts are skipped."""
        db = self._make_db_with_agents(
            [
                {"loaded": True, "twitter_username": "u", "last_post_timestamp": None},
            ]
        )
        assert not db.get_active_agents()

    def test_old_post_skipped(self) -> None:
        """Test that agents with posts older than 7 days are skipped."""
        old = datetime.now(timezone.utc) - timedelta(days=8)
        db = self._make_db_with_agents(
            [
                {"loaded": True, "twitter_username": "u", "last_post_timestamp": old},
            ]
        )
        assert not db.get_active_agents()

    def test_active_with_username_included(self) -> None:
        """Test active agent with username is included."""
        recent = datetime.now(timezone.utc) - timedelta(days=1)
        db = self._make_db_with_agents(
            [
                {
                    "loaded": True,
                    "twitter_username": "active",
                    "last_post_timestamp": recent,
                },
            ]
        )
        result = db.get_active_agents()
        assert len(result) == 1
        assert result[0].twitter_username == "active"

    def test_active_without_username_excluded(self) -> None:
        """Test active agent without username is excluded."""
        recent = datetime.now(timezone.utc) - timedelta(days=1)
        db = self._make_db_with_agents(
            [
                {
                    "loaded": True,
                    "twitter_username": None,
                    "last_post_timestamp": recent,
                },
            ]
        )
        assert not db.get_active_agents()

    def test_mixed_agents(self) -> None:
        """Test with a mix of active, inactive, and unloaded agents."""
        recent = datetime.now(timezone.utc) - timedelta(hours=12)
        old = datetime.now(timezone.utc) - timedelta(days=10)
        db = self._make_db_with_agents(
            [
                {
                    "agent_id": 1,
                    "loaded": True,
                    "twitter_username": "a1",
                    "last_post_timestamp": recent,
                },
                {
                    "agent_id": 2,
                    "loaded": True,
                    "twitter_username": "i1",
                    "last_post_timestamp": old,
                },
                {
                    "agent_id": 3,
                    "loaded": False,
                    "twitter_username": "u1",
                    "last_post_timestamp": recent,
                },
                {
                    "agent_id": 4,
                    "loaded": True,
                    "twitter_username": "a2",
                    "last_post_timestamp": recent,
                },
                {
                    "agent_id": 5,
                    "loaded": True,
                    "twitter_username": None,
                    "last_post_timestamp": recent,
                },
            ]
        )
        result = db.get_active_agents()
        assert len(result) == 2
        assert {a.twitter_username for a in result} == {"a1", "a2"}

    def test_boundary_exactly_7_days(self) -> None:
        """Test agent with post exactly at the 7-day boundary is excluded."""
        boundary = datetime.now(timezone.utc) - timedelta(days=7)
        db = self._make_db_with_agents(
            [
                {
                    "loaded": True,
                    "twitter_username": "b",
                    "last_post_timestamp": boundary,
                },
            ]
        )
        assert not db.get_active_agents()

    def test_just_within_7_days(self) -> None:
        """Test agent with post just within the 7-day window is included."""
        within = datetime.now(timezone.utc) - timedelta(days=6, hours=23)
        db = self._make_db_with_agents(
            [
                {
                    "loaded": True,
                    "twitter_username": "w",
                    "last_post_timestamp": within,
                },
            ]
        )
        assert len(db.get_active_agents()) == 1

    def test_logger_warns_for_unloaded_agent(self) -> None:
        """Test that a warning is logged for unloaded agents."""
        db = self._make_db_with_agents(
            [
                {
                    "agent_id": 99,
                    "loaded": False,
                    "twitter_username": None,
                    "last_post_timestamp": NOW,
                },
            ]
        )
        db.get_active_agents()
        db.logger.warning.assert_called_once()

    def test_logger_warns_for_active_no_username(self) -> None:
        """Test that a warning is logged for active agents without username."""
        recent = datetime.now(timezone.utc) - timedelta(hours=1)
        db = self._make_db_with_agents(
            [
                {
                    "agent_id": 55,
                    "loaded": True,
                    "twitter_username": None,
                    "last_post_timestamp": recent,
                },
            ]
        )
        db.get_active_agents()
        db.logger.warning.assert_called_once()
        assert "55" in db.logger.warning.call_args[0][0]

    def test_no_logger_no_warnings(self) -> None:
        """Test no crash when logger is None and unloaded agent present."""
        db = self._make_db_with_agents(
            [
                {
                    "loaded": False,
                    "twitter_username": None,
                    "last_post_timestamp": None,
                },
            ]
        )
        db.logger = None
        # Should not raise
        db.get_active_agents()
