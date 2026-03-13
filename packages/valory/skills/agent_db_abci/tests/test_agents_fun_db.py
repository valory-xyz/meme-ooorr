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

"""Tests for agents_fun_db module."""

# pylint: disable=too-few-public-methods

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from packages.valory.skills.agent_db_abci.agent_db_models import (
    AgentInstance,
    AgentType,
    AttributeDefinition,
)
from packages.valory.skills.agent_db_abci.agents_fun_db import (
    AgentsFunAgent,
    AgentsFunDatabase,
    MEMEOOORR,
)
from packages.valory.skills.agent_db_abci.twitter_models import (
    TwitterLike,
    TwitterPost,
    TwitterRewtweet,
)


def _make_agent_instance(**overrides):
    """Create a test AgentInstance."""
    defaults = {
        "agent_id": 1,
        "type_id": 10,
        "agent_name": "test-agent",
        "eth_address": "0xABC",
        "created_at": "2025-01-01T00:00:00+00:00",
    }
    defaults.update(overrides)
    return AgentInstance(**defaults)


def _make_agent_type(**overrides):
    """Create a test AgentType."""
    defaults = {
        "type_id": 10,
        "type_name": "memeooorr",
        "description": "desc",
    }
    defaults.update(overrides)
    return AgentType(**defaults)


def _make_client_mock():
    """Create a mock AgentDBClient."""
    client = MagicMock()
    client.address = "0xABC"
    client.agent = _make_agent_instance()
    client.logger = MagicMock()
    client.agent_type_name = MEMEOOORR
    client.agent_name_template = "memeooorr-agent-{address}"
    return client


def _make_database() -> AgentsFunDatabase:
    """Create an AgentsFunDatabase with Model.__init__ mocked out."""
    with patch("packages.valory.skills.agent_db_abci.agents_fun_db.Model.__init__"):
        db = AgentsFunDatabase()
    # __init__ lines 189-194 have already executed, setting these attributes.
    # We verify they were set correctly.
    return db


def _exhaust_gen(gen):
    """Drive a generator to completion, returning its return value."""
    result = None
    try:
        gen.send(None)
        while True:
            gen.send(None)
    except StopIteration as e:
        result = e.value
    return result


class TestAgentsFunDatabaseInit:
    """Test AgentsFunDatabase.__init__ (lines 189-194)."""

    def test_init_defaults(self):
        """Test default attribute values after init."""
        db = _make_database()
        assert db.client is None
        assert db.agent_type is None
        assert not db.agents
        assert db.my_agent is None
        assert db.logger is None


class TestAgentsFunDatabaseInitialize:
    """Test AgentsFunDatabase.initialize."""

    def test_initialize_sets_client(self):
        """Test initialize sets client and logger."""
        db = _make_database()
        mock_client = _make_client_mock()
        db.initialize(mock_client)

        assert db.client is mock_client
        assert db.logger is mock_client.logger
        assert mock_client.agent_type_name == MEMEOOORR
        assert mock_client.agent_name_template == "memeooorr-agent-{address}"


class TestAgentsFunAgent:
    """Test AgentsFunAgent."""

    def test_init(self):
        """Test constructor sets defaults."""
        client = _make_client_mock()
        ai = _make_agent_instance()
        agent = AgentsFunAgent(client, ai)

        assert agent.client is client
        assert agent.agent_instance is ai
        assert agent.twitter_username is None
        assert agent.twitter_user_id is None
        assert agent.posts == []
        assert agent.likes == []
        assert agent.retweets == []
        assert agent.follows == []
        assert agent.loaded is False

    def test_str(self):
        """Test string representation."""
        client = _make_client_mock()
        ai = _make_agent_instance(agent_id=42)
        agent = AgentsFunAgent(client, ai)
        agent.twitter_username = "testuser"
        agent.loaded = True
        result = str(agent)
        assert "42" in result
        assert "testuser" in result

    def test_delete(self):
        """Test delete yields from client.delete_agent_instance."""
        client = _make_client_mock()
        ai = _make_agent_instance()
        agent = AgentsFunAgent(client, ai)

        def mock_delete(instance):
            yield

        client.delete_agent_instance = mock_delete

        gen = agent.delete()
        _exhaust_gen(gen)

    def test_load_all_interaction_types(self):
        """Test load parses all Twitter interaction types."""
        client = _make_client_mock()
        ai = _make_agent_instance()
        agent = AgentsFunAgent(client, ai)

        now = datetime.now(timezone.utc)
        parsed_attrs = [
            {"attr_name": "twitter_username", "attr_value": "testuser"},
            {"attr_name": "twitter_user_id", "attr_value": "12345"},
            {
                "attr_name": "twitter_interactions",
                "attr_value": {
                    "action": "post",
                    "timestamp": now.isoformat(),
                    "details": {"tweet_id": "t1", "text": "hello"},
                },
            },
            {
                "attr_name": "twitter_interactions",
                "attr_value": {
                    "action": "retweet",
                    "timestamp": now.isoformat(),
                    "details": {"tweet_id": "t2"},
                },
            },
            {
                "attr_name": "twitter_interactions",
                "attr_value": {
                    "action": "like",
                    "timestamp": now.isoformat(),
                    "details": {"tweet_id": "t3"},
                },
            },
            {
                "attr_name": "twitter_interactions",
                "attr_value": {
                    "action": "follow",
                    "timestamp": now.isoformat(),
                    "details": {"username": "otheruser"},
                },
            },
        ]

        def mock_get_attrs(ai):
            yield
            return parsed_attrs

        client.get_all_agent_instance_attributes_parsed = mock_get_attrs

        gen = agent.load()
        _exhaust_gen(gen)

        assert agent.twitter_username == "testuser"
        assert agent.twitter_user_id == "12345"
        assert len(agent.posts) == 1
        assert len(agent.retweets) == 1
        assert len(agent.likes) == 1
        assert len(agent.follows) == 1
        assert agent.loaded is True

    def test_load_unknown_attr_name_skipped(self):
        """Test load skips unknown attribute names (falls through all if/elif)."""
        client = _make_client_mock()
        ai = _make_agent_instance()
        agent = AgentsFunAgent(client, ai)

        parsed_attrs = [
            {"attr_name": "some_other_field", "attr_value": "irrelevant"},
        ]

        def mock_get_attrs(ai):
            yield
            return parsed_attrs

        client.get_all_agent_instance_attributes_parsed = mock_get_attrs

        gen = agent.load()
        _exhaust_gen(gen)

        assert agent.twitter_username is None
        assert agent.loaded is True

    def test_load_unknown_action_raises(self):
        """Test load raises ValueError for unknown Twitter action (branch 111->106)."""
        client = _make_client_mock()
        ai = _make_agent_instance()
        agent = AgentsFunAgent(client, ai)

        now = datetime.now(timezone.utc)
        parsed_attrs = [
            {
                "attr_name": "twitter_interactions",
                "attr_value": {
                    "action": "unknown_action",
                    "timestamp": now.isoformat(),
                    "details": {},
                },
            },
        ]

        def mock_get_attrs(ai):
            yield
            return parsed_attrs

        client.get_all_agent_instance_attributes_parsed = mock_get_attrs

        gen = agent.load()
        with pytest.raises(ValueError, match="Unknown Twitter action: unknown_action"):
            _exhaust_gen(gen)

    def test_load_empty_attributes(self):
        """Test load with no attributes."""
        client = _make_client_mock()
        ai = _make_agent_instance()
        agent = AgentsFunAgent(client, ai)

        def mock_get_attrs(ai):
            yield
            return []

        client.get_all_agent_instance_attributes_parsed = mock_get_attrs

        gen = agent.load()
        _exhaust_gen(gen)
        assert agent.loaded is True
        assert agent.posts == []

    def test_add_interaction(self):
        """Test add_interaction creates attribute instance."""
        client = _make_client_mock()
        ai = _make_agent_instance()
        agent = AgentsFunAgent(client, ai)

        now = datetime.now(timezone.utc)
        post = TwitterPost(action="post", timestamp=now, tweet_id="t1", text="hello")

        attr_def = AttributeDefinition(
            attr_def_id=5,
            type_id=10,
            attr_name="twitter_interactions",
            data_type="json",
            is_required=False,
            default_value="{}",
        )

        def mock_get_attr_def(name):
            yield
            return attr_def

        def mock_create_attr_inst(agent_instance, attribute_def, value, value_type):
            yield
            return MagicMock()

        client.get_attribute_definition_by_name = mock_get_attr_def
        client.create_attribute_instance = mock_create_attr_inst

        gen = agent.add_interaction(post)
        result = _exhaust_gen(gen)
        assert result is not None

    def test_add_interaction_unknown_action_raises(self):
        """Test add_interaction raises for unknown action."""
        client = _make_client_mock()
        ai = _make_agent_instance()
        agent = AgentsFunAgent(client, ai)

        interaction = MagicMock()
        interaction.action = "bogus"

        gen = agent.add_interaction(interaction)
        with pytest.raises(ValueError, match="Unknown Twitter action: bogus"):
            _exhaust_gen(gen)

    def test_add_interaction_attr_def_not_found_raises(self):
        """Test add_interaction raises when attr definition not found."""
        client = _make_client_mock()
        ai = _make_agent_instance()
        agent = AgentsFunAgent(client, ai)

        now = datetime.now(timezone.utc)
        post = TwitterPost(action="post", timestamp=now, tweet_id="t1", text="hello")

        def mock_get_attr_def(name):
            yield

        client.get_attribute_definition_by_name = mock_get_attr_def

        gen = agent.add_interaction(post)
        with pytest.raises(ValueError, match="Attribute definition not found"):
            _exhaust_gen(gen)

    def test_update_twitter_details(self):
        """Test update_twitter_details calls update_or_create_agent_attribute."""
        client = _make_client_mock()
        ai = _make_agent_instance()
        agent = AgentsFunAgent(client, ai)
        agent.twitter_username = "user1"
        agent.twitter_user_id = "id1"

        def mock_update(name, value):
            yield
            return True

        client.update_or_create_agent_attribute = mock_update

        gen = agent.update_twitter_details()
        _exhaust_gen(gen)


class TestAgentsFunDatabaseLoad:
    """Test AgentsFunDatabase.load."""

    def test_load_success(self):
        """Test successful load with agents."""
        db = _make_database()
        client = _make_client_mock()
        db.initialize(client)

        agent_type = _make_agent_type()
        agent_instances = [_make_agent_instance()]

        def mock_ensure_type_def(desc):
            yield

        def mock_ensure_attr_def(defs):
            yield

        def mock_ensure_agent():
            yield

        def mock_get_type(name):
            yield
            return agent_type

        def mock_get_instances(type_id):
            yield
            return agent_instances

        def mock_get_attrs(ai):
            yield
            return []

        client._ensure_agent_type_definition = mock_ensure_type_def
        client._ensure_agent_type_attribute_definition = mock_ensure_attr_def
        client._ensure_agent_instance = mock_ensure_agent
        client.get_agent_type_by_type_name = mock_get_type
        client.get_agent_instances_by_type_id = mock_get_instances
        client.get_all_agent_instance_attributes_parsed = mock_get_attrs

        gen = db.load()
        _exhaust_gen(gen)

        assert db.agent_type is not None
        assert len(db.agents) == 1
        assert db.my_agent is not None  # eth_address matches client.address

    def test_load_agent_type_already_set(self):
        """Test load skips get_agent_type_by_type_name when agent_type already set."""
        db = _make_database()
        client = _make_client_mock()
        db.initialize(client)

        agent_type = _make_agent_type()
        db.agent_type = agent_type  # Pre-set

        def mock_ensure_type_def(desc):
            yield

        def mock_ensure_attr_def(defs):
            yield

        def mock_ensure_agent():
            yield

        def mock_get_instances(type_id):
            yield
            return []

        client._ensure_agent_type_definition = mock_ensure_type_def
        client._ensure_agent_type_attribute_definition = mock_ensure_attr_def
        client._ensure_agent_instance = mock_ensure_agent
        client.get_agent_instances_by_type_id = mock_get_instances

        gen = db.load()
        _exhaust_gen(gen)

        assert db.agent_type is agent_type

    def test_load_agent_type_not_found(self):
        """Test load when agent type can't be found."""
        db = _make_database()
        client = _make_client_mock()
        db.initialize(client)

        def mock_ensure_type_def(desc):
            yield

        def mock_ensure_attr_def(defs):
            yield

        def mock_ensure_agent():
            yield

        def mock_get_type(name):
            yield

        client._ensure_agent_type_definition = mock_ensure_type_def
        client._ensure_agent_type_attribute_definition = mock_ensure_attr_def
        client._ensure_agent_instance = mock_ensure_agent
        client.get_agent_type_by_type_name = mock_get_type

        gen = db.load()
        _exhaust_gen(gen)

        assert db.agent_type is None
        client.logger.error.assert_called_once()

    def test_load_my_agent_fallback(self):
        """Test load creates my_agent from client.agent if not found in instances."""
        db = _make_database()
        client = _make_client_mock()
        db.initialize(client)

        agent_type = _make_agent_type()
        # Agent instances don't include our address
        other_instance = _make_agent_instance(agent_id=99, eth_address="0xOTHER")

        def mock_ensure_type_def(desc):
            yield

        def mock_ensure_attr_def(defs):
            yield

        def mock_ensure_agent():
            yield

        def mock_get_type(name):
            yield
            return agent_type

        def mock_get_instances(type_id):
            yield
            return [other_instance]

        def mock_get_attrs(ai):
            yield
            return []

        client._ensure_agent_type_definition = mock_ensure_type_def
        client._ensure_agent_type_attribute_definition = mock_ensure_attr_def
        client._ensure_agent_instance = mock_ensure_agent
        client.get_agent_type_by_type_name = mock_get_type
        client.get_agent_instances_by_type_id = mock_get_instances
        client.get_all_agent_instance_attributes_parsed = mock_get_attrs

        gen = db.load()
        _exhaust_gen(gen)

        # my_agent should be set from client.agent as fallback
        assert db.my_agent is not None
        assert len(db.agents) == 2  # other_instance + my_agent fallback


class TestAgentsFunDatabaseGetTweetLikesNumber:
    """Test AgentsFunDatabase.get_tweet_likes_number."""

    def test_likes_already_loaded(self):
        """Test counting likes with agents already loaded."""
        db = _make_database()
        client = _make_client_mock()
        db.client = client
        db.logger = client.logger

        now = datetime.now(timezone.utc)
        ai = _make_agent_instance()
        agent = AgentsFunAgent(client, ai)
        agent.loaded = True
        agent.likes = [
            TwitterLike(action="like", timestamp=now, tweet_id="t1"),
            TwitterLike(action="like", timestamp=now, tweet_id="t2"),
        ]
        db.agents = [agent]

        gen = db.get_tweet_likes_number("t1")
        result = _exhaust_gen(gen)
        assert result == 1

    def test_likes_not_loaded(self):
        """Test loading agent that wasn't loaded."""
        db = _make_database()
        client = _make_client_mock()
        db.client = client
        db.logger = client.logger

        ai = _make_agent_instance()
        agent = AgentsFunAgent(client, ai)
        agent.loaded = False

        def mock_load():
            agent.loaded = True
            agent.likes = [
                TwitterLike(
                    action="like", timestamp=datetime.now(timezone.utc), tweet_id="t1"
                ),
            ]
            yield

        agent.load = mock_load
        db.agents = [agent]

        gen = db.get_tweet_likes_number("t1")
        result = _exhaust_gen(gen)
        assert result == 1

    def test_likes_no_match(self):
        """Test no matching likes — inner loop exhausts without break."""
        db = _make_database()
        client = _make_client_mock()
        db.client = client
        db.logger = client.logger

        agent = MagicMock()
        agent.loaded = True
        like = MagicMock()
        like.tweet_id = "other_tweet"
        agent.likes = [like]
        db.agents = [agent]

        gen = db.get_tweet_likes_number("t1")
        result = _exhaust_gen(gen)
        assert result == 0


class TestAgentsFunDatabaseGetTweetRetweetsNumber:
    """Test AgentsFunDatabase.get_tweet_retweets_number."""

    def test_retweets_already_loaded(self):
        """Test with agents already loaded (branch 254->253)."""
        db = _make_database()
        client = _make_client_mock()
        db.client = client
        db.logger = client.logger

        now = datetime.now(timezone.utc)
        ai = _make_agent_instance()
        agent = AgentsFunAgent(client, ai)
        agent.loaded = True
        agent.retweets = [
            TwitterRewtweet(action="retweet", timestamp=now, tweet_id="t1"),
        ]
        db.agents = [agent]

        gen = db.get_tweet_retweets_number("t1")
        result = _exhaust_gen(gen)
        assert result == 1

    def test_retweets_not_loaded(self):
        """Test with agent not loaded."""
        db = _make_database()
        client = _make_client_mock()
        db.client = client
        db.logger = client.logger

        ai = _make_agent_instance()
        agent = AgentsFunAgent(client, ai)
        agent.loaded = False

        def mock_load():
            agent.loaded = True
            agent.retweets = [
                TwitterRewtweet(
                    action="retweet",
                    timestamp=datetime.now(timezone.utc),
                    tweet_id="t1",
                ),
            ]
            yield

        agent.load = mock_load
        db.agents = [agent]

        gen = db.get_tweet_retweets_number("t1")
        result = _exhaust_gen(gen)
        assert result == 1

    def test_retweets_no_match(self):
        """Test no matching retweets."""
        db = _make_database()
        client = _make_client_mock()
        db.client = client
        db.logger = client.logger

        ai = _make_agent_instance()
        agent = AgentsFunAgent(client, ai)
        agent.loaded = True
        agent.retweets = [
            TwitterRewtweet(
                action="retweet", timestamp=datetime.now(timezone.utc), tweet_id="t2"
            ),
        ]
        db.agents = [agent]

        gen = db.get_tweet_retweets_number("t999")
        result = _exhaust_gen(gen)
        assert result == 0


class TestAgentsFunDatabaseGetTweetReplies:
    """Test AgentsFunDatabase.get_tweet_replies."""

    def test_replies_found(self):
        """Test finding replies to a tweet."""
        db = _make_database()
        client = _make_client_mock()
        db.client = client
        db.logger = client.logger

        now = datetime.now(timezone.utc)
        ai = _make_agent_instance()
        agent = AgentsFunAgent(client, ai)
        agent.loaded = True
        agent.posts = [
            TwitterPost(
                action="post",
                timestamp=now,
                tweet_id="r1",
                text="reply",
                reply_to_tweet_id="t1",
            ),
            TwitterPost(action="post", timestamp=now, tweet_id="t2", text="unrelated"),
        ]
        db.agents = [agent]

        gen = db.get_tweet_replies("t1")
        result = _exhaust_gen(gen)
        assert len(result) == 1
        assert result[0].tweet_id == "r1"

    def test_replies_agent_not_loaded(self):
        """Test loading agent for replies."""
        db = _make_database()
        client = _make_client_mock()
        db.client = client
        db.logger = client.logger

        ai = _make_agent_instance()
        agent = AgentsFunAgent(client, ai)
        agent.loaded = False

        def mock_load():
            agent.loaded = True
            agent.posts = [
                TwitterPost(
                    action="post",
                    timestamp=datetime.now(timezone.utc),
                    tweet_id="r1",
                    text="reply",
                    reply_to_tweet_id="t1",
                ),
            ]
            yield

        agent.load = mock_load
        db.agents = [agent]

        gen = db.get_tweet_replies("t1")
        result = _exhaust_gen(gen)
        assert len(result) == 1

    def test_replies_no_match(self):
        """Test no matching replies — inner loop exhausts without break."""
        db = _make_database()
        client = _make_client_mock()
        db.client = client
        db.logger = client.logger

        now = datetime.now(timezone.utc)
        ai = _make_agent_instance()
        agent = AgentsFunAgent(client, ai)
        agent.loaded = True
        agent.posts = [
            TwitterPost(action="post", timestamp=now, tweet_id="p1", text="unrelated"),
        ]
        db.agents = [agent]

        gen = db.get_tweet_replies("nonexistent")
        result = _exhaust_gen(gen)
        assert len(result) == 0


class TestAgentsFunDatabaseGetTweetFeedback:
    """Test AgentsFunDatabase.get_tweet_feedback."""

    def test_feedback(self):
        """Test get_tweet_feedback aggregates likes, retweets, replies."""
        db = _make_database()
        client = _make_client_mock()
        db.client = client
        db.logger = client.logger

        now = datetime.now(timezone.utc)
        ai = _make_agent_instance()
        agent = AgentsFunAgent(client, ai)
        agent.loaded = True
        agent.likes = [TwitterLike(action="like", timestamp=now, tweet_id="t1")]
        agent.retweets = [
            TwitterRewtweet(action="retweet", timestamp=now, tweet_id="t1")
        ]
        agent.posts = [
            TwitterPost(
                action="post",
                timestamp=now,
                tweet_id="r1",
                text="reply",
                reply_to_tweet_id="t1",
            ),
        ]
        db.agents = [agent]

        gen = db.get_tweet_feedback("t1")
        result = _exhaust_gen(gen)
        assert result["likes"] == 1
        assert result["retweets"] == 1
        assert len(result["replies"]) == 1


class TestAgentsFunDatabaseGetActiveAgents:
    """Test AgentsFunDatabase.get_active_agents."""

    def test_active_agent_with_username(self):
        """Test active agent with username is included."""
        db = _make_database()
        db.logger = MagicMock()
        client = _make_client_mock()

        ai = _make_agent_instance()
        agent = AgentsFunAgent(client, ai)
        agent.loaded = True
        agent.twitter_username = "active_user"
        now = datetime.now(timezone.utc)
        agent.posts = [
            TwitterPost(
                action="post",
                timestamp=now - timedelta(days=1),
                tweet_id="t1",
                text="hello",
            )
        ]
        db.agents = [agent]

        result = db.get_active_agents()
        assert len(result) == 1

    def test_agent_not_loaded_skipped_with_logger(self):
        """Test unloaded agent is skipped and logged (branch 313->289)."""
        db = _make_database()
        db.logger = MagicMock()
        client = _make_client_mock()

        ai = _make_agent_instance()
        agent = AgentsFunAgent(client, ai)
        agent.loaded = False
        db.agents = [agent]

        result = db.get_active_agents()
        assert len(result) == 0
        db.logger.warning.assert_called_once()

    def test_agent_not_loaded_skipped_without_logger(self):
        """Test unloaded agent is skipped without logger."""
        db = _make_database()
        db.logger = None
        client = _make_client_mock()

        ai = _make_agent_instance()
        agent = AgentsFunAgent(client, ai)
        agent.loaded = False
        db.agents = [agent]

        result = db.get_active_agents()
        assert len(result) == 0

    def test_agent_no_posts_skipped(self):
        """Test loaded agent with no posts is skipped."""
        db = _make_database()
        db.logger = MagicMock()
        client = _make_client_mock()

        ai = _make_agent_instance()
        agent = AgentsFunAgent(client, ai)
        agent.loaded = True
        agent.posts = []
        db.agents = [agent]

        result = db.get_active_agents()
        assert len(result) == 0

    def test_agent_old_posts_skipped(self):
        """Test loaded agent with posts older than 7 days is skipped."""
        db = _make_database()
        db.logger = MagicMock()
        client = _make_client_mock()

        ai = _make_agent_instance()
        agent = AgentsFunAgent(client, ai)
        agent.loaded = True
        agent.twitter_username = "old_user"
        old_time = datetime.now(timezone.utc) - timedelta(days=10)
        agent.posts = [
            TwitterPost(
                action="post", timestamp=old_time, tweet_id="t1", text="old post"
            )
        ]
        db.agents = [agent]

        result = db.get_active_agents()
        assert len(result) == 0

    def test_active_agent_no_username_logged(self):
        """Test active agent without username is logged and not included."""
        db = _make_database()
        db.logger = MagicMock()
        client = _make_client_mock()

        ai = _make_agent_instance()
        agent = AgentsFunAgent(client, ai)
        agent.loaded = True
        agent.twitter_username = None
        now = datetime.now(timezone.utc)
        agent.posts = [
            TwitterPost(
                action="post",
                timestamp=now - timedelta(days=1),
                tweet_id="t1",
                text="hello",
            )
        ]
        db.agents = [agent]

        result = db.get_active_agents()
        assert len(result) == 0
        db.logger.warning.assert_called_once()
        assert "no twitter_username" in db.logger.warning.call_args[0][0]

    def test_active_agent_no_username_no_logger(self):
        """Test active agent without username when logger is None."""
        db = _make_database()
        db.logger = None
        client = _make_client_mock()

        ai = _make_agent_instance()
        agent = AgentsFunAgent(client, ai)
        agent.loaded = True
        agent.twitter_username = None
        now = datetime.now(timezone.utc)
        agent.posts = [
            TwitterPost(
                action="post",
                timestamp=now - timedelta(days=1),
                tweet_id="t1",
                text="hello",
            )
        ]
        db.agents = [agent]

        result = db.get_active_agents()
        assert len(result) == 0


class TestAgentsFunDatabaseStr:
    """Test AgentsFunDatabase.__str__."""

    def test_str(self):
        """Test string representation."""
        db = _make_database()
        db.agents = [MagicMock(), MagicMock()]
        assert "2 agents" in str(db)
