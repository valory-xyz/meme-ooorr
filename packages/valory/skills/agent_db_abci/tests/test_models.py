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

"""Tests for models.py."""

# pylint: disable=R0903

from packages.valory.skills.abstract_round_abci.models import BaseParams
from packages.valory.skills.abstract_round_abci.models import (
    BenchmarkTool as BaseBenchmarkTool,
)
from packages.valory.skills.abstract_round_abci.models import Requests as BaseRequests
from packages.valory.skills.abstract_round_abci.models import (
    SharedState as BaseSharedState,
)
from packages.valory.skills.agent_db_abci.agent_db_client import (
    AgentDBClient as BaseAgentDBClient,
)
from packages.valory.skills.agent_db_abci.agents_fun_db import (
    AgentsFunDatabase as BaseAgentsFunDatabase,
)
from packages.valory.skills.agent_db_abci.models import (
    AgentDBClient,
    AgentsFunDatabase,
    BenchmarkTool,
    Params,
    Requests,
    SharedState,
)
from packages.valory.skills.agent_db_abci.rounds import AgentDBAbciApp


class TestSharedState:
    """Tests for SharedState."""

    def test_is_subclass_of_base(self) -> None:
        """Test SharedState is a subclass of BaseSharedState."""
        assert issubclass(SharedState, BaseSharedState)

    def test_abci_app_cls(self) -> None:
        """Test that abci_app_cls is set to AgentDBAbciApp."""
        assert SharedState.abci_app_cls is AgentDBAbciApp


class TestParams:
    """Tests for Params."""

    def test_is_subclass_of_base(self) -> None:
        """Test Params is a subclass of BaseParams."""
        assert issubclass(Params, BaseParams)


class TestModelAliases:
    """Tests for model aliases."""

    def test_agent_db_client_alias(self) -> None:
        """Test AgentDBClient alias."""
        assert AgentDBClient is BaseAgentDBClient

    def test_agents_fun_database_alias(self) -> None:
        """Test AgentsFunDatabase alias."""
        assert AgentsFunDatabase is BaseAgentsFunDatabase

    def test_requests_alias(self) -> None:
        """Test Requests alias."""
        assert Requests is BaseRequests

    def test_benchmark_tool_alias(self) -> None:
        """Test BenchmarkTool alias."""
        assert BenchmarkTool is BaseBenchmarkTool
