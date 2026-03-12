# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2025 Valory AG
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

"""Tests for the rounds module of the agent_performance_summary_abci skill."""

# pylint: disable=R0903

import pytest

from packages.valory.skills.abstract_round_abci.base import (
    BaseSynchronizedData,
    DegenerateRound,
    VotingRound,
    get_name,
)
from packages.valory.skills.agent_performance_summary_abci.payloads import (
    FetchPerformanceDataPayload,
)
from packages.valory.skills.agent_performance_summary_abci.rounds import (
    AgentPerformanceSummaryAbciApp,
    Event,
    FetchPerformanceDataRound,
    FinishedFetchPerformanceDataRound,
)


class TestFetchPerformanceDataRound:
    """Tests for the FetchPerformanceDataRound."""

    def test_is_voting_round(self) -> None:
        """Test that FetchPerformanceDataRound is a VotingRound."""
        assert issubclass(FetchPerformanceDataRound, VotingRound)

    def test_payload_class(self) -> None:
        """Test the payload_class attribute."""
        assert FetchPerformanceDataRound.payload_class is FetchPerformanceDataPayload

    def test_synchronized_data_class(self) -> None:
        """Test the synchronized_data_class attribute."""
        assert FetchPerformanceDataRound.synchronized_data_class is BaseSynchronizedData

    def test_done_event(self) -> None:
        """Test the done_event attribute."""
        assert FetchPerformanceDataRound.done_event == Event.DONE

    def test_negative_event(self) -> None:
        """Test the negative_event attribute."""
        assert FetchPerformanceDataRound.negative_event == Event.FAIL

    def test_none_event(self) -> None:
        """Test the none_event attribute."""
        assert FetchPerformanceDataRound.none_event == Event.NONE

    def test_no_majority_event(self) -> None:
        """Test the no_majority_event attribute."""
        assert FetchPerformanceDataRound.no_majority_event == Event.NO_MAJORITY

    def test_collection_key(self) -> None:
        """Test the collection_key attribute."""
        assert FetchPerformanceDataRound.collection_key == get_name(
            BaseSynchronizedData.participant_to_votes
        )


class TestFinishedFetchPerformanceDataRound:
    """Tests for the FinishedFetchPerformanceDataRound."""

    def test_is_degenerate_round(self) -> None:
        """Test that FinishedFetchPerformanceDataRound is a DegenerateRound."""
        assert issubclass(FinishedFetchPerformanceDataRound, DegenerateRound)


class TestAgentPerformanceSummaryAbciApp:
    """Tests for the AgentPerformanceSummaryAbciApp."""

    def test_initial_round_cls(self) -> None:
        """Test the initial_round_cls attribute."""
        assert (
            AgentPerformanceSummaryAbciApp.initial_round_cls
            is FetchPerformanceDataRound
        )

    def test_transition_function_keys(self) -> None:
        """Test that transition function has the expected round keys."""
        tf = AgentPerformanceSummaryAbciApp.transition_function
        assert set(tf.keys()) == {
            FetchPerformanceDataRound,
            FinishedFetchPerformanceDataRound,
        }

    @pytest.mark.parametrize(
        "event, expected_round",
        [
            (Event.DONE, FinishedFetchPerformanceDataRound),
            (Event.NONE, FetchPerformanceDataRound),
            (Event.FAIL, FinishedFetchPerformanceDataRound),
            (Event.ROUND_TIMEOUT, FinishedFetchPerformanceDataRound),
            (Event.NO_MAJORITY, FetchPerformanceDataRound),
        ],
    )
    def test_fetch_round_transitions(self, event: Event, expected_round: type) -> None:
        """Test all transitions from FetchPerformanceDataRound."""
        tf = AgentPerformanceSummaryAbciApp.transition_function
        assert tf[FetchPerformanceDataRound][event] is expected_round

    def test_finished_round_has_no_transitions(self) -> None:
        """Test that FinishedFetchPerformanceDataRound has no transitions."""
        tf = AgentPerformanceSummaryAbciApp.transition_function
        assert tf[FinishedFetchPerformanceDataRound] == {}

    def test_final_states(self) -> None:
        """Test the final_states attribute."""
        assert AgentPerformanceSummaryAbciApp.final_states == {
            FinishedFetchPerformanceDataRound,
        }

    def test_event_to_timeout(self) -> None:
        """Test the event_to_timeout attribute."""
        assert AgentPerformanceSummaryAbciApp.event_to_timeout == {
            Event.ROUND_TIMEOUT: 30.0,
        }

    def test_db_pre_conditions(self) -> None:
        """Test the db_pre_conditions attribute."""
        assert AgentPerformanceSummaryAbciApp.db_pre_conditions == {
            FetchPerformanceDataRound: set(),
        }

    def test_db_post_conditions(self) -> None:
        """Test the db_post_conditions attribute."""
        assert AgentPerformanceSummaryAbciApp.db_post_conditions == {
            FinishedFetchPerformanceDataRound: set(),
        }
