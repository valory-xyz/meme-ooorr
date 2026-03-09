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

"""Tests for behaviours.py."""

from packages.dvilela.skills.memeooorr_abci.behaviours import MemeooorrRoundBehaviour
from packages.dvilela.skills.memeooorr_chained_abci.behaviours import (
    MemeooorrChainedConsensusBehaviour,
)
from packages.dvilela.skills.memeooorr_chained_abci.composition import (
    MemeooorrChainedSkillAbciApp,
)
from packages.valory.skills.abstract_round_abci.behaviours import AbstractRoundBehaviour
from packages.valory.skills.agent_performance_summary_abci.behaviours import (
    AgentPerformanceSummaryRoundBehaviour,
)
from packages.valory.skills.mech_interact_abci.behaviours.round_behaviour import (
    MechInteractRoundBehaviour,
)
from packages.valory.skills.registration_abci.behaviours import (
    AgentRegistrationRoundBehaviour,
    RegistrationStartupBehaviour,
)
from packages.valory.skills.reset_pause_abci.behaviours import (
    ResetPauseABCIConsensusBehaviour,
)
from packages.valory.skills.termination_abci.behaviours import (
    BackgroundBehaviour,
    TerminationAbciBehaviours,
)
from packages.valory.skills.transaction_settlement_abci.behaviours import (
    TransactionSettlementRoundBehaviour,
)


class TestMemeooorrChainedConsensusBehaviour:
    """Tests for MemeooorrChainedConsensusBehaviour."""

    def test_is_subclass_of_abstract_round_behaviour(self) -> None:
        """Test it is a subclass of AbstractRoundBehaviour."""
        assert issubclass(MemeooorrChainedConsensusBehaviour, AbstractRoundBehaviour)

    def test_initial_behaviour_cls(self) -> None:
        """Test initial_behaviour_cls is RegistrationStartupBehaviour."""
        assert (
            MemeooorrChainedConsensusBehaviour.initial_behaviour_cls
            is RegistrationStartupBehaviour
        )

    def test_abci_app_cls(self) -> None:
        """Test abci_app_cls is MemeooorrChainedSkillAbciApp."""
        assert (
            MemeooorrChainedConsensusBehaviour.abci_app_cls
            is MemeooorrChainedSkillAbciApp
        )

    def test_behaviours_is_set(self) -> None:
        """Test behaviours is a set."""
        assert isinstance(MemeooorrChainedConsensusBehaviour.behaviours, set)

    def test_behaviours_contains_registration(self) -> None:
        """Test behaviours contains AgentRegistrationRoundBehaviour behaviours."""
        assert AgentRegistrationRoundBehaviour.behaviours.issubset(
            MemeooorrChainedConsensusBehaviour.behaviours
        )

    def test_behaviours_contains_performance_summary(self) -> None:
        """Test behaviours contains AgentPerformanceSummaryRoundBehaviour behaviours."""
        assert AgentPerformanceSummaryRoundBehaviour.behaviours.issubset(
            MemeooorrChainedConsensusBehaviour.behaviours
        )

    def test_behaviours_contains_reset_pause(self) -> None:
        """Test behaviours contains ResetPauseABCIConsensusBehaviour behaviours."""
        assert ResetPauseABCIConsensusBehaviour.behaviours.issubset(
            MemeooorrChainedConsensusBehaviour.behaviours
        )

    def test_behaviours_contains_memeooorr(self) -> None:
        """Test behaviours contains MemeooorrRoundBehaviour behaviours."""
        assert set(MemeooorrRoundBehaviour.behaviours).issubset(
            MemeooorrChainedConsensusBehaviour.behaviours
        )

    def test_behaviours_contains_termination(self) -> None:
        """Test behaviours contains TerminationAbciBehaviours behaviours."""
        assert TerminationAbciBehaviours.behaviours.issubset(
            MemeooorrChainedConsensusBehaviour.behaviours
        )

    def test_behaviours_contains_transaction_settlement(self) -> None:
        """Test behaviours contains TransactionSettlementRoundBehaviour behaviours."""
        assert TransactionSettlementRoundBehaviour.behaviours.issubset(
            MemeooorrChainedConsensusBehaviour.behaviours
        )

    def test_behaviours_contains_mech_interact(self) -> None:
        """Test behaviours contains MechInteractRoundBehaviour behaviours."""
        assert MechInteractRoundBehaviour.behaviours.issubset(
            MemeooorrChainedConsensusBehaviour.behaviours
        )

    def test_background_behaviours_cls(self) -> None:
        """Test background_behaviours_cls contains BackgroundBehaviour."""
        assert MemeooorrChainedConsensusBehaviour.background_behaviours_cls == {
            BackgroundBehaviour
        }

    def test_behaviours_not_empty(self) -> None:
        """Test behaviours set is not empty."""
        assert len(MemeooorrChainedConsensusBehaviour.behaviours) > 0

    def test_all_sub_behaviours_combined(self) -> None:
        """Test that all sub-behaviour sets are combined into the behaviours set."""
        expected = (
            set(AgentRegistrationRoundBehaviour.behaviours)
            | set(AgentPerformanceSummaryRoundBehaviour.behaviours)
            | set(ResetPauseABCIConsensusBehaviour.behaviours)
            | set(MemeooorrRoundBehaviour.behaviours)
            | set(TerminationAbciBehaviours.behaviours)
            | set(TransactionSettlementRoundBehaviour.behaviours)
            | set(MechInteractRoundBehaviour.behaviours)
        )
        assert MemeooorrChainedConsensusBehaviour.behaviours == expected
