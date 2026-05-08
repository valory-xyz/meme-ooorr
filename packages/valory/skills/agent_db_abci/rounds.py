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

"""This package contains the rounds of AgentDBAbciApp."""

from enum import Enum
from typing import Dict, FrozenSet, Set

from packages.valory.skills.abstract_round_abci.base import (
    AbciApp,
    AbciAppTransitionFunction,
    AppState,
    BaseSynchronizedData,
    CollectSameUntilThresholdRound,
    CollectionRound,
    DegenerateRound,
    DeserializedCollection,
    EventToTimeout,
    get_name,
)
from packages.valory.skills.agent_db_abci.payloads import AgentDBPayload


class Event(Enum):
    """AgentDBAbciApp Events"""

    ROUND_TIMEOUT = "round_timeout"
    NO_MAJORITY = "no_majority"
    DONE = "done"
    NONE = "none"


class SynchronizedData(BaseSynchronizedData):
    """
    Class to represent the synchronized data.

    This data is replicated by the tendermint application.
    """

    def _get_deserialized(self, key: str) -> DeserializedCollection:
        """Strictly get a collection and return it deserialized."""
        serialized = self.db.get_strict(key)
        return CollectionRound.deserialize_collection(serialized)

    @property
    def participants_to_agent_db(self) -> DeserializedCollection:
        """Get the participants to the AgentDB round."""
        return self._get_deserialized("participants_to_agent_db")

    @property
    def agent_db_content(self) -> str:
        """Get the most-voted AgentDB payload content."""
        value = self.db.get_strict("agent_db_content")
        if not isinstance(value, str):
            raise TypeError(
                f"agent_db_content must be str, got {type(value).__name__}"
            )
        return value


class AgentDBRound(CollectSameUntilThresholdRound):
    """AgentDBRound"""

    payload_class = AgentDBPayload
    synchronized_data_class = SynchronizedData
    done_event = Event.DONE
    # AgentDBPayload.content is non-Optional str, so most_voted_payload
    # cannot be None and the (AgentDBRound, NONE) self-loop is unreachable
    # at runtime. Declared to satisfy the metaclass contract.
    none_event = Event.NONE
    no_majority_event = Event.NO_MAJORITY
    collection_key = get_name(SynchronizedData.participants_to_agent_db)
    selection_key = get_name(SynchronizedData.agent_db_content)


class FinishedReadingRound(DegenerateRound):
    """FinishedReadingRound"""


class AgentDBAbciApp(AbciApp[Event]):
    """AgentDBAbciApp

    Initial round: AgentDBRound

    Initial states: {AgentDBRound}

    Transition states:
        0. AgentDBRound
            - done: 1.
            - none: 0.
            - no majority: 0.
            - round timeout: 0.
        1. FinishedReadingRound

    Final states: {FinishedReadingRound}

    Timeouts:
        round timeout: 30.0
    """

    initial_round_cls: AppState = AgentDBRound
    initial_states: Set[AppState] = {AgentDBRound}
    transition_function: AbciAppTransitionFunction = {
        AgentDBRound: {
            Event.DONE: FinishedReadingRound,
            Event.NONE: AgentDBRound,
            Event.NO_MAJORITY: AgentDBRound,
            Event.ROUND_TIMEOUT: AgentDBRound,
        },
        FinishedReadingRound: {},
    }
    final_states: Set[AppState] = {FinishedReadingRound}
    event_to_timeout: EventToTimeout = {
        Event.ROUND_TIMEOUT: 30.0,
    }
    cross_period_persisted_keys: FrozenSet[str] = frozenset()
    db_pre_conditions: Dict[AppState, Set[str]] = {
        AgentDBRound: set(),
    }
    db_post_conditions: Dict[AppState, Set[str]] = {
        FinishedReadingRound: set(),
    }
