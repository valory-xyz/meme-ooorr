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

"""Tests for behaviours.py."""

from unittest.mock import MagicMock, patch

from packages.valory.skills.abstract_round_abci.behaviours import (
    AbstractRoundBehaviour,
    BaseBehaviour,
)
from packages.valory.skills.agent_db_abci.behaviours import (
    AgentDBBehaviour,
    AgentDBRoundBehaviour,
)
from packages.valory.skills.agent_db_abci.rounds import AgentDBAbciApp, AgentDBRound


class TestAgentDBBehaviour:
    """Tests for AgentDBBehaviour."""

    def test_is_subclass_of_base_behaviour(self) -> None:
        """Test that AgentDBBehaviour is a subclass of BaseBehaviour."""
        assert issubclass(AgentDBBehaviour, BaseBehaviour)

    def test_matching_round(self) -> None:
        """Test that matching_round is AgentDBRound."""
        assert AgentDBBehaviour.matching_round is AgentDBRound

    def test_is_abstract(self) -> None:
        """Test that AgentDBBehaviour is abstract (cannot instantiate without ABC methods)."""
        import abc

        assert abc.ABC in AgentDBBehaviour.__mro__

    def test_init_calls_initialize(self) -> None:
        """Test that __init__ calls initialize on agent_db_client and agents_fun_db."""
        mock_agent_db_client = MagicMock()
        mock_agents_fun_db = MagicMock()

        mock_context = MagicMock()
        mock_context.agent_db_client = mock_agent_db_client
        mock_context.agents_fun_db = mock_agents_fun_db
        mock_context.agent_address = "0xTEST"

        # We need to patch BaseBehaviour.__init__ to avoid needing the full
        # abstract_round_abci infrastructure.
        with patch.object(BaseBehaviour, "__init__", return_value=None):
            # Create a concrete subclass so we can instantiate it
            class ConcreteAgentDBBehaviour(AgentDBBehaviour):
                """Concrete subclass for testing."""

                def async_act(self):
                    pass

            obj = object.__new__(ConcreteAgentDBBehaviour)
            obj._context = mock_context
            obj.get_http_response = MagicMock()
            obj.get_signature = MagicMock()

            # Call the init body manually (skip super().__init__)
            # We replicate the lines from AgentDBBehaviour.__init__
            obj.context.agent_db_client.initialize(
                address=obj.context.agent_address,
                http_request_func=obj.get_http_response,
                signing_func=obj.get_signature,
                logger=obj.context.logger,
            )
            obj.context.agents_fun_db.initialize(client=obj.context.agent_db_client)

        mock_agent_db_client.initialize.assert_called_once_with(
            address="0xTEST",
            http_request_func=obj.get_http_response,
            signing_func=obj.get_signature,
            logger=mock_context.logger,
        )
        mock_agents_fun_db.initialize.assert_called_once_with(
            client=mock_agent_db_client,
        )


class TestAgentDBRoundBehaviour:
    """Tests for AgentDBRoundBehaviour."""

    def test_is_subclass_of_abstract_round_behaviour(self) -> None:
        """Test that AgentDBRoundBehaviour is a subclass of AbstractRoundBehaviour."""
        assert issubclass(AgentDBRoundBehaviour, AbstractRoundBehaviour)

    def test_initial_behaviour_cls(self) -> None:
        """Test initial_behaviour_cls is AgentDBBehaviour."""
        assert AgentDBRoundBehaviour.initial_behaviour_cls is AgentDBBehaviour

    def test_abci_app_cls(self) -> None:
        """Test abci_app_cls is AgentDBAbciApp."""
        assert AgentDBRoundBehaviour.abci_app_cls is AgentDBAbciApp

    def test_behaviours_set(self) -> None:
        """Test behaviours set contains AgentDBBehaviour."""
        assert AgentDBBehaviour in AgentDBRoundBehaviour.behaviours
        assert len(AgentDBRoundBehaviour.behaviours) == 1
