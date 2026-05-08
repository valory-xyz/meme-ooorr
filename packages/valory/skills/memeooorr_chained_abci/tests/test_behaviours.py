# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024-2026 Valory AG
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

from packages.valory.skills.agent_performance_summary_abci.behaviours import (
    AgentPerformanceSummaryRoundBehaviour,
)
from packages.valory.skills.mech_interact_abci.behaviours.round_behaviour import (
    MechInteractRoundBehaviour,
)
from packages.valory.skills.memeooorr_abci.behaviours import MemeooorrRoundBehaviour
from packages.valory.skills.memeooorr_chained_abci.behaviours import (
    MemeooorrChainedConsensusBehaviour,
)
from packages.valory.skills.memeooorr_chained_abci.composition import (
    MemeooorrChainedSkillAbciApp,
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


class TestMemeooorrChainedConsensusBehaviour:  # pylint: disable=R0903
    """Tests for MemeooorrChainedConsensusBehaviour."""

    def test_all_sub_behaviours_combined(self) -> None:
        """Test that all sub-behaviour sets are combined into the behaviours set."""
        assert (
            MemeooorrChainedConsensusBehaviour.initial_behaviour_cls
            is RegistrationStartupBehaviour
        )
        assert (
            MemeooorrChainedConsensusBehaviour.abci_app_cls
            is MemeooorrChainedSkillAbciApp
        )
        assert MemeooorrChainedConsensusBehaviour.background_behaviours_cls == {
            BackgroundBehaviour
        }
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


class TestMemeooorrRoundBehaviourStandalone:  # pylint: disable=R0903
    """Standalone-mode invariants for MemeooorrRoundBehaviour."""

    def test_initial_behaviour_matches_initial_round(self) -> None:
        """The initial behaviour's matching round must equal the FSM initial round.

        The framework only checks that ``initial_behaviour_cls`` is in
        the ``behaviours`` list. It does not check that its
        ``matching_round`` is in ``initial_states``. This invariant
        catches the latent mis-start that would surface if anyone runs
        ``memeooorr_abci`` standalone.
        """
        from packages.valory.skills.memeooorr_abci.rounds import MemeooorrAbciApp

        assert (
            MemeooorrRoundBehaviour.initial_behaviour_cls.matching_round
            is MemeooorrAbciApp.initial_round_cls
        )
