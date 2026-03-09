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

from packages.dvilela.skills.memeooorr_abci.behaviour_classes.chain import (
    ActionPreparationBehaviour,
    CallCheckpointBehaviour,
    CheckFundsBehaviour,
    CheckStakingBehaviour,
    PostTxDecisionMakingBehaviour,
    PullMemesBehaviour,
    TransactionLoopCheckBehaviour,
)
from packages.dvilela.skills.memeooorr_abci.behaviour_classes.db import (
    LoadDatabaseBehaviour,
)
from packages.dvilela.skills.memeooorr_abci.behaviour_classes.llm import (
    ActionDecisionBehaviour,
)
from packages.dvilela.skills.memeooorr_abci.behaviour_classes.mech import (
    FailedMechRequestBehaviour,
    FailedMechResponseBehaviour,
    PostMechResponseBehaviour,
)
from packages.dvilela.skills.memeooorr_abci.behaviour_classes.twitter import (
    ActionTweetBehaviour,
    CollectFeedbackBehaviour,
    EngageTwitterBehaviour,
)
from packages.dvilela.skills.memeooorr_abci.behaviours import MemeooorrRoundBehaviour
from packages.dvilela.skills.memeooorr_abci.rounds import MemeooorrAbciApp
from packages.valory.skills.abstract_round_abci.behaviours import (
    AbstractRoundBehaviour,
    BaseBehaviour,
)


class TestMemeooorrRoundBehaviour:
    """Tests for MemeooorrRoundBehaviour."""

    def test_is_subclass_of_abstract_round_behaviour(self) -> None:
        """Test that MemeooorrRoundBehaviour is a subclass of AbstractRoundBehaviour."""
        assert issubclass(MemeooorrRoundBehaviour, AbstractRoundBehaviour)

    def test_initial_behaviour_cls(self) -> None:
        """Test initial_behaviour_cls is CollectFeedbackBehaviour."""
        assert MemeooorrRoundBehaviour.initial_behaviour_cls is CollectFeedbackBehaviour

    def test_abci_app_cls(self) -> None:
        """Test abci_app_cls is MemeooorrAbciApp."""
        assert MemeooorrRoundBehaviour.abci_app_cls is MemeooorrAbciApp

    def test_behaviours_set_contains_all_behaviours(self) -> None:
        """Test behaviours set contains all expected behaviour classes."""
        expected_behaviours = {
            LoadDatabaseBehaviour,
            CheckStakingBehaviour,
            PullMemesBehaviour,
            CollectFeedbackBehaviour,
            EngageTwitterBehaviour,
            ActionDecisionBehaviour,
            ActionPreparationBehaviour,
            CheckFundsBehaviour,
            ActionTweetBehaviour,
            PostTxDecisionMakingBehaviour,
            CallCheckpointBehaviour,
            TransactionLoopCheckBehaviour,
            PostMechResponseBehaviour,
            FailedMechRequestBehaviour,
            FailedMechResponseBehaviour,
        }
        assert set(MemeooorrRoundBehaviour.behaviours) == expected_behaviours

    def test_behaviours_count(self) -> None:
        """Test the total number of behaviours."""
        assert len(MemeooorrRoundBehaviour.behaviours) == 15

    def test_all_behaviours_are_base_behaviour_subclasses(self) -> None:
        """Test that all behaviours in the set are subclasses of BaseBehaviour."""
        for behaviour_cls in MemeooorrRoundBehaviour.behaviours:
            assert issubclass(behaviour_cls, BaseBehaviour), (
                f"{behaviour_cls.__name__} is not a subclass of BaseBehaviour"
            )
