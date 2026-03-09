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

"""Tests for rounds.py."""

import pytest

from packages.valory.skills.agent_db_abci.payloads import AgentDBPayload
from packages.valory.skills.agent_db_abci.rounds import (
    AgentDBAbciApp,
    AgentDBRound,
    Event,
    FinishedReadingRound,
    SynchronizedData,
)


class TestEvent:
    """Tests for the Event enum."""

    def test_round_timeout(self) -> None:
        """Test ROUND_TIMEOUT event."""
        assert Event.ROUND_TIMEOUT.value == "round_timeout"

    def test_no_majority(self) -> None:
        """Test NO_MAJORITY event."""
        assert Event.NO_MAJORITY.value == "no_majority"

    def test_done(self) -> None:
        """Test DONE event."""
        assert Event.DONE.value == "done"

    def test_event_count(self) -> None:
        """Test the total number of events."""
        assert len(Event) == 3


class TestSynchronizedData:
    """Tests for SynchronizedData."""

    def test_is_subclass(self) -> None:
        """Test that SynchronizedData is a subclass of BaseSynchronizedData."""
        from packages.valory.skills.abstract_round_abci.base import (
            BaseSynchronizedData,
        )

        assert issubclass(SynchronizedData, BaseSynchronizedData)


class TestAgentDBRound:
    """Tests for AgentDBRound."""

    def test_payload_class(self) -> None:
        """Test the payload class attribute."""
        assert AgentDBRound.payload_class is AgentDBPayload

    def test_synchronized_data_class(self) -> None:
        """Test the synchronized data class attribute."""
        assert AgentDBRound.synchronized_data_class is SynchronizedData


class TestAgentDBAbciApp:
    """Tests for the AgentDBAbciApp."""

    def test_initial_round_cls(self) -> None:
        """Test that the initial round is AgentDBRound."""
        assert AgentDBAbciApp.initial_round_cls is AgentDBRound

    def test_initial_states(self) -> None:
        """Test that the initial states contain AgentDBRound."""
        assert AgentDBAbciApp.initial_states == {AgentDBRound}

    def test_final_states(self) -> None:
        """Test that the final states contain FinishedReadingRound."""
        assert AgentDBAbciApp.final_states == {FinishedReadingRound}

    def test_transition_function_keys(self) -> None:
        """Test that the transition function has the correct keys."""
        assert set(AgentDBAbciApp.transition_function.keys()) == {
            AgentDBRound,
            FinishedReadingRound,
        }

    def test_agent_db_round_transitions(self) -> None:
        """Test AgentDBRound transitions."""
        transitions = AgentDBAbciApp.transition_function[AgentDBRound]
        assert transitions[Event.DONE] is FinishedReadingRound
        assert transitions[Event.NO_MAJORITY] is AgentDBRound
        assert transitions[Event.ROUND_TIMEOUT] is AgentDBRound

    def test_finished_reading_round_transitions(self) -> None:
        """Test FinishedReadingRound has no transitions (terminal)."""
        assert AgentDBAbciApp.transition_function[FinishedReadingRound] == {}

    def test_event_to_timeout(self) -> None:
        """Test event timeout mapping."""
        assert Event.ROUND_TIMEOUT in AgentDBAbciApp.event_to_timeout
        assert AgentDBAbciApp.event_to_timeout[Event.ROUND_TIMEOUT] == 30.0

    def test_cross_period_persisted_keys(self) -> None:
        """Test cross-period persisted keys is empty."""
        assert AgentDBAbciApp.cross_period_persisted_keys == frozenset()

    def test_db_pre_conditions(self) -> None:
        """Test DB pre-conditions."""
        assert AgentDBAbciApp.db_pre_conditions == {AgentDBRound: set()}

    def test_db_post_conditions(self) -> None:
        """Test DB post-conditions."""
        assert AgentDBAbciApp.db_post_conditions == {FinishedReadingRound: set()}
