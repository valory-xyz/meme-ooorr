# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023-2026 Valory AG
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

"""Tests for behaviours module."""

from typing import Generator
from unittest.mock import MagicMock, patch

from packages.valory.skills.agent_db_abci.behaviours import (
    AgentDBBehaviour,
    AgentDBRoundBehaviour,
)
from packages.valory.skills.agent_db_abci.rounds import AgentDBAbciApp, AgentDBRound


class TestAgentDBBehaviour:
    """Test AgentDBBehaviour."""

    def test_init_calls_initialize(self) -> None:
        """Test that __init__ calls initialize on agent_db_client and agents_fun_db."""
        mock_context = MagicMock()
        mock_context.agent_address = "0xABC"
        mock_context.agent_db_client = MagicMock()
        mock_context.agents_fun_db = MagicMock()
        mock_context.logger = MagicMock()

        # Create a concrete subclass to avoid ABC instantiation error
        class ConcreteAgentDBBehaviour(AgentDBBehaviour):
            """Concrete subclass for testing."""

            def async_act(self) -> Generator[None, None, None]:  # type: ignore[override]
                """No-op."""
                yield

        with patch(
            "packages.valory.skills.agent_db_abci.behaviours.BaseBehaviour.__init__",
            return_value=None,
        ):
            behaviour = ConcreteAgentDBBehaviour.__new__(ConcreteAgentDBBehaviour)
            behaviour._context = mock_context
            behaviour.get_http_response = MagicMock()  # type: ignore[method-assign]
            behaviour.get_signature = MagicMock()  # type: ignore[method-assign]

            ConcreteAgentDBBehaviour.__init__(behaviour)

            mock_context.agent_db_client.initialize.assert_called_once_with(
                address="0xABC",
                http_request_func=behaviour.get_http_response,
                signing_func=behaviour.get_signature,
                logger=mock_context.logger,
            )
            mock_context.agents_fun_db.initialize.assert_called_once_with(
                client=mock_context.agent_db_client,
            )

    def test_matching_round(self) -> None:
        """Test matching_round is AgentDBRound."""
        assert AgentDBBehaviour.matching_round is AgentDBRound


class TestAgentDBRoundBehaviour:  # pylint: disable=too-few-public-methods
    """Test AgentDBRoundBehaviour."""

    def test_class_attributes(self) -> None:
        """Test class attributes are set correctly."""
        assert AgentDBRoundBehaviour.initial_behaviour_cls is AgentDBBehaviour
        assert AgentDBRoundBehaviour.abci_app_cls is AgentDBAbciApp
        assert AgentDBBehaviour in AgentDBRoundBehaviour.behaviours
