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

from unittest.mock import MagicMock, patch

from packages.dvilela.skills.memeooorr_abci.models import (
    AgentDBClient as BaseAgentDBClient,
)
from packages.dvilela.skills.memeooorr_abci.models import (
    AgentsFunDatabase as BaseAgentsFunDatabase,
)
from packages.dvilela.skills.memeooorr_abci.models import Params as MemeooorrParams
from packages.dvilela.skills.memeooorr_abci.models import (
    RandomnessApi as MemeooorrRandomnessApi,
)
from packages.dvilela.skills.memeooorr_abci.rounds import Event as MemeooorrEvent
from packages.dvilela.skills.memeooorr_chained_abci.composition import (
    MemeooorrChainedSkillAbciApp,
)
from packages.dvilela.skills.memeooorr_chained_abci.models import (
    MARGIN,
    MULTIPLIER,
    MULTIPLIER_MECH,
    AgentDBClient,
    AgentsFunDatabase,
    BenchmarkTool,
    MechResponseSpecs,
    MechToolsSpecs,
    MechsSubgraph,
    Params,
    RandomnessApi,
    Requests,
    SharedState,
)
from packages.valory.skills.abstract_round_abci.models import (
    BenchmarkTool as BaseBenchmarkTool,
)
from packages.valory.skills.abstract_round_abci.models import Requests as BaseRequests
from packages.valory.skills.agent_performance_summary_abci.models import (
    AgentPerformanceSummaryParams,
)
from packages.valory.skills.agent_performance_summary_abci.models import (
    SharedState as BaseSharedState,
)
from packages.valory.skills.mech_interact_abci.models import (
    MechResponseSpecs as BaseMechResponseSpecs,
)
from packages.valory.skills.mech_interact_abci.models import (
    MechToolsSpecs as BaseMechToolsSpecs,
)
from packages.valory.skills.mech_interact_abci.models import (
    MechsSubgraph as BaseMechsSubgraph,
)
from packages.valory.skills.mech_interact_abci.rounds import Event as MechInteractEvent
from packages.valory.skills.reset_pause_abci.rounds import Event as ResetPauseEvent
from packages.valory.skills.termination_abci.models import TerminationParams


class TestConstants:
    """Tests for module-level constants."""

    def test_margin(self) -> None:
        """Test MARGIN constant."""
        assert MARGIN == 5

    def test_multiplier(self) -> None:
        """Test MULTIPLIER constant."""
        assert MULTIPLIER == 10

    def test_multiplier_mech(self) -> None:
        """Test MULTIPLIER_MECH constant."""
        assert MULTIPLIER_MECH == 20


class TestModelAliases:
    """Tests for model aliases."""

    def test_requests_alias(self) -> None:
        """Test Requests alias."""
        assert Requests is BaseRequests

    def test_benchmark_tool_alias(self) -> None:
        """Test BenchmarkTool alias."""
        assert BenchmarkTool is BaseBenchmarkTool

    def test_randomness_api_alias(self) -> None:
        """Test RandomnessApi alias."""
        assert RandomnessApi is MemeooorrRandomnessApi

    def test_mech_response_specs_alias(self) -> None:
        """Test MechResponseSpecs alias."""
        assert MechResponseSpecs is BaseMechResponseSpecs

    def test_mech_tools_specs_alias(self) -> None:
        """Test MechToolsSpecs alias."""
        assert MechToolsSpecs is BaseMechToolsSpecs

    def test_mechs_subgraph_alias(self) -> None:
        """Test MechsSubgraph alias."""
        assert MechsSubgraph is BaseMechsSubgraph

    def test_agent_db_client_alias(self) -> None:
        """Test AgentDBClient alias."""
        assert AgentDBClient is BaseAgentDBClient

    def test_agents_fun_database_alias(self) -> None:
        """Test AgentsFunDatabase alias."""
        assert AgentsFunDatabase is BaseAgentsFunDatabase


class TestSharedState:
    """Tests for SharedState."""

    def test_is_subclass_of_base(self) -> None:
        """Test SharedState is a subclass of BaseSharedState."""
        assert issubclass(SharedState, BaseSharedState)

    def test_abci_app_cls(self) -> None:
        """Test that abci_app_cls is set to MemeooorrChainedSkillAbciApp."""
        assert SharedState.abci_app_cls is MemeooorrChainedSkillAbciApp

    def test_init_sets_env_var_status(self) -> None:
        """Test that __init__ sets env_var_status attribute."""
        mock_context = MagicMock()
        with patch.object(BaseSharedState, "__init__", return_value=None):
            shared_state = SharedState(skill_context=mock_context)
        assert shared_state.env_var_status == {
            "needs_update": False,
            "env_vars": {},
        }

    def test_setup_sets_reset_pause_round_timeout(self) -> None:
        """Test that setup sets ResetPauseEvent.ROUND_TIMEOUT."""
        mock_context = MagicMock()
        mock_context.params.round_timeout_seconds = 30.0
        mock_context.params.reset_pause_duration = 10

        with patch.object(BaseSharedState, "__init__", return_value=None):
            shared_state = SharedState(skill_context=mock_context)

        shared_state._context = mock_context

        with patch.object(BaseSharedState, "setup"):
            shared_state.setup()

        assert (
            MemeooorrChainedSkillAbciApp.event_to_timeout[
                ResetPauseEvent.ROUND_TIMEOUT
            ]
            == 30.0
        )

    def test_setup_sets_reset_and_pause_timeout(self) -> None:
        """Test that setup sets ResetPauseEvent.RESET_AND_PAUSE_TIMEOUT."""
        mock_context = MagicMock()
        mock_context.params.round_timeout_seconds = 30.0
        mock_context.params.reset_pause_duration = 10

        with patch.object(BaseSharedState, "__init__", return_value=None):
            shared_state = SharedState(skill_context=mock_context)

        shared_state._context = mock_context

        with patch.object(BaseSharedState, "setup"):
            shared_state.setup()

        assert (
            MemeooorrChainedSkillAbciApp.event_to_timeout[
                ResetPauseEvent.RESET_AND_PAUSE_TIMEOUT
            ]
            == 10 + MARGIN
        )

    def test_setup_sets_memeooorr_round_timeout(self) -> None:
        """Test that setup sets MemeooorrEvent.ROUND_TIMEOUT with MULTIPLIER."""
        mock_context = MagicMock()
        mock_context.params.round_timeout_seconds = 30.0
        mock_context.params.reset_pause_duration = 10

        with patch.object(BaseSharedState, "__init__", return_value=None):
            shared_state = SharedState(skill_context=mock_context)

        shared_state._context = mock_context

        with patch.object(BaseSharedState, "setup"):
            shared_state.setup()

        assert (
            MemeooorrChainedSkillAbciApp.event_to_timeout[
                MemeooorrEvent.ROUND_TIMEOUT
            ]
            == 30.0 * MULTIPLIER
        )

    def test_setup_sets_mech_interact_round_timeout(self) -> None:
        """Test that setup sets MechInteractEvent.ROUND_TIMEOUT with MULTIPLIER_MECH."""
        mock_context = MagicMock()
        mock_context.params.round_timeout_seconds = 30.0
        mock_context.params.reset_pause_duration = 10

        with patch.object(BaseSharedState, "__init__", return_value=None):
            shared_state = SharedState(skill_context=mock_context)

        shared_state._context = mock_context

        with patch.object(BaseSharedState, "setup"):
            shared_state.setup()

        assert (
            MemeooorrChainedSkillAbciApp.event_to_timeout[
                MechInteractEvent.ROUND_TIMEOUT
            ]
            == 30.0 * MULTIPLIER_MECH
        )

    def test_setup_calls_super_setup(self) -> None:
        """Test that setup calls super().setup()."""
        mock_context = MagicMock()
        mock_context.params.round_timeout_seconds = 30.0
        mock_context.params.reset_pause_duration = 10

        with patch.object(BaseSharedState, "__init__", return_value=None):
            shared_state = SharedState(skill_context=mock_context)

        shared_state._context = mock_context

        with patch.object(BaseSharedState, "setup") as mock_super_setup:
            shared_state.setup()
            mock_super_setup.assert_called_once()


class TestParams:
    """Tests for Params."""

    def test_is_subclass_of_memeooorr_params(self) -> None:
        """Test Params is a subclass of MemeooorrParams."""
        assert issubclass(Params, MemeooorrParams)

    def test_is_subclass_of_termination_params(self) -> None:
        """Test Params is a subclass of TerminationParams."""
        assert issubclass(Params, TerminationParams)

    def test_is_subclass_of_agent_performance_summary_params(self) -> None:
        """Test Params is a subclass of AgentPerformanceSummaryParams."""
        assert issubclass(Params, AgentPerformanceSummaryParams)

    def test_mro_includes_all_parents(self) -> None:
        """Test the MRO includes all expected parent classes."""
        mro = Params.__mro__
        assert MemeooorrParams in mro
        assert TerminationParams in mro
        assert AgentPerformanceSummaryParams in mro
