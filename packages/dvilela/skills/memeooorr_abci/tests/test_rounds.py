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

"""Tests for rounds.py."""

# pylint: disable=too-few-public-methods,too-many-public-methods

import json
from dataclasses import dataclass
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from packages.dvilela.skills.memeooorr_abci.payloads import (
    ActionDecisionPayload,
    ActionPreparationPayload,
    ActionTweetPayload,
    CallCheckpointPayload,
    CheckFundsPayload,
    CheckStakingPayload,
    CollectFeedbackPayload,
    EngageTwitterPayload,
    LoadDatabasePayload,
    MechPayload,
    PostTxDecisionMakingPayload,
    PullMemesPayload,
    TransactionLoopCheckPayload,
)
from packages.dvilela.skills.memeooorr_abci.rounds import (
    ActionDecisionRound,
    ActionPreparationRound,
    ActionTweetRound,
    CallCheckpointRound,
    CheckFundsRound,
    CheckStakingRound,
    CollectFeedbackRound,
    DataclassEncoder,
    EngageTwitterRound,
    Event,
    EventRoundBase,
    FailedMechRequestRound,
    FailedMechResponseRound,
    FinishedForMechRequestRound,
    FinishedForMechResponseRound,
    FinishedToResetRound,
    FinishedToSettlementRound,
    LoadDatabaseRound,
    MAX_CHECK_FUNDS_COUNT,
    MechRoundBase,
    MemeooorrAbciApp,
    PostMechResponseRound,
    PostTxDecisionMakingRound,
    PullMemesRound,
    StakingState,
    SynchronizedData,
    TransactionLoopCheckRound,
)
from packages.valory.skills.abstract_round_abci.base import (
    BaseSynchronizedData,
    BaseTxPayload,
    CollectionRound,
    DegenerateRound,
)


class TestStakingState:
    """Tests for StakingState enum."""

    def test_unstaked(self) -> None:
        """Test UNSTAKED value."""
        assert StakingState.UNSTAKED.value == 0

    def test_staked(self) -> None:
        """Test STAKED value."""
        assert StakingState.STAKED.value == 1

    def test_evicted(self) -> None:
        """Test EVICTED value."""
        assert StakingState.EVICTED.value == 2

    def test_member_count(self) -> None:
        """Test number of members."""
        assert len(StakingState) == 3


class TestEvent:
    """Tests for Event enum."""

    def test_done(self) -> None:
        """Test DONE event."""
        assert Event.DONE.value == "done"

    def test_no_funds(self) -> None:
        """Test NO_FUNDS event."""
        assert Event.NO_FUNDS.value == "no_funds"

    def test_settle(self) -> None:
        """Test SETTLE event."""
        assert Event.SETTLE.value == "settle"

    def test_error(self) -> None:
        """Test ERROR event."""
        assert Event.ERROR.value == "ERROR"

    def test_no_majority(self) -> None:
        """Test NO_MAJORITY event."""
        assert Event.NO_MAJORITY.value == "no_majority"

    def test_round_timeout(self) -> None:
        """Test ROUND_TIMEOUT event."""
        assert Event.ROUND_TIMEOUT.value == "round_timeout"

    def test_wait(self) -> None:
        """Test WAIT event."""
        assert Event.WAIT.value == "wait"

    def test_action(self) -> None:
        """Test ACTION event."""
        assert Event.ACTION.value == "action"

    def test_missing_tweet(self) -> None:
        """Test MISSING_TWEET event."""
        assert Event.MISSING_TWEET.value == "missing_tweet"

    def test_mech(self) -> None:
        """Test MECH event."""
        assert Event.MECH.value == "mech"

    def test_retry(self) -> None:
        """Test RETRY event."""
        assert Event.RETRY.value == "retry"

    def test_skip(self) -> None:
        """Test SKIP event."""
        assert Event.SKIP.value == "skip"

    def test_invalid_auth(self) -> None:
        """Test INVALID_AUTH event."""
        assert Event.INVALID_AUTH.value == "invalid_auth"

    def test_none(self) -> None:
        """Test NONE event."""
        assert Event.NONE.value == "none"

    def test_event_count(self) -> None:
        """Test total number of events."""
        assert len(Event) == 14

    def test_event_from_value(self) -> None:
        """Test creating event from value."""
        assert Event("done") == Event.DONE
        assert Event("no_funds") == Event.NO_FUNDS


class TestMaxCheckFundsCount:
    """Tests for the MAX_CHECK_FUNDS_COUNT constant."""

    def test_value(self) -> None:
        """Test the constant value."""
        assert MAX_CHECK_FUNDS_COUNT == 15


class TestSynchronizedData:
    """Tests for SynchronizedData."""

    def test_is_subclass(self) -> None:
        """Test that SynchronizedData is a subclass of BaseSynchronizedData."""
        assert issubclass(SynchronizedData, BaseSynchronizedData)

    def test_persona_default(self) -> None:
        """Test persona property with default."""
        db = MagicMock()
        db.get.return_value = None
        sd = SynchronizedData(db=db)
        assert sd.persona is None

    def test_persona_set(self) -> None:
        """Test persona property with value."""
        db = MagicMock()
        db.get.return_value = "test_persona"
        sd = SynchronizedData(db=db)
        assert sd.persona == "test_persona"

    def test_heart_cooldown_hours(self) -> None:
        """Test heart_cooldown_hours property."""
        db = MagicMock()
        db.get.return_value = 24
        sd = SynchronizedData(db=db)
        assert sd.heart_cooldown_hours == 24

    def test_heart_cooldown_hours_default(self) -> None:
        """Test heart_cooldown_hours property default."""
        db = MagicMock()
        db.get.return_value = None
        sd = SynchronizedData(db=db)
        assert sd.heart_cooldown_hours is None

    def test_summon_cooldown_seconds(self) -> None:
        """Test summon_cooldown_seconds property."""
        db = MagicMock()
        db.get.return_value = 3600
        sd = SynchronizedData(db=db)
        assert sd.summon_cooldown_seconds == 3600

    def test_summon_cooldown_seconds_default(self) -> None:
        """Test summon_cooldown_seconds default."""
        db = MagicMock()
        db.get.return_value = None
        sd = SynchronizedData(db=db)
        assert sd.summon_cooldown_seconds is None

    def test_meme_coins_default(self) -> None:
        """Test meme_coins default value."""
        db = MagicMock()
        db.get.return_value = "[]"
        sd = SynchronizedData(db=db)
        assert sd.meme_coins == []

    def test_meme_coins_with_data(self) -> None:
        """Test meme_coins with data."""
        db = MagicMock()
        db.get.return_value = '[{"name": "test"}]'
        sd = SynchronizedData(db=db)
        assert sd.meme_coins == [{"name": "test"}]

    def test_pending_tweet_none(self) -> None:
        """Test pending_tweet when None."""
        db = MagicMock()
        db.get.return_value = None
        sd = SynchronizedData(db=db)
        assert sd.pending_tweet == []

    def test_pending_tweet_with_data(self) -> None:
        """Test pending_tweet with data."""
        db = MagicMock()
        db.get.return_value = '["tweet1", "tweet2"]'
        sd = SynchronizedData(db=db)
        assert sd.pending_tweet == ["tweet1", "tweet2"]

    def test_feedback_none(self) -> None:
        """Test feedback when None."""
        db = MagicMock()
        db.get.return_value = None
        sd = SynchronizedData(db=db)
        assert sd.feedback == []

    def test_feedback_with_data(self) -> None:
        """Test feedback with data."""
        db = MagicMock()
        db.get.return_value = '["good", "bad"]'
        sd = SynchronizedData(db=db)
        assert sd.feedback == ["good", "bad"]

    def test_token_action_default(self) -> None:
        """Test token_action default."""
        db = MagicMock()
        db.get.return_value = "{}"
        sd = SynchronizedData(db=db)
        assert sd.token_action == {}

    def test_token_action_with_data(self) -> None:
        """Test token_action with data."""
        db = MagicMock()
        db.get.return_value = '{"action": "summon"}'
        sd = SynchronizedData(db=db)
        assert sd.token_action == {"action": "summon"}

    def test_most_voted_tx_hash(self) -> None:
        """Test most_voted_tx_hash."""
        db = MagicMock()
        db.get_strict.return_value = "0xhash"
        sd = SynchronizedData(db=db)
        assert sd.most_voted_tx_hash == "0xhash"

    def test_final_tx_hash_none(self) -> None:
        """Test final_tx_hash when not set."""
        db = MagicMock()
        db.get.return_value = None
        sd = SynchronizedData(db=db)
        assert sd.final_tx_hash is None

    def test_final_tx_hash_set(self) -> None:
        """Test final_tx_hash when set."""
        db = MagicMock()
        db.get.return_value = "0xfinal"
        sd = SynchronizedData(db=db)
        assert sd.final_tx_hash == "0xfinal"

    def test_tx_submitter(self) -> None:
        """Test tx_submitter."""
        db = MagicMock()
        db.get_strict.return_value = "submitter"
        sd = SynchronizedData(db=db)
        assert sd.tx_submitter == "submitter"

    def test_is_staking_kpi_met_none(self) -> None:
        """Test is_staking_kpi_met with None."""
        db = MagicMock()
        db.get.return_value = None
        sd = SynchronizedData(db=db)
        assert sd.is_staking_kpi_met is False

    def test_is_staking_kpi_met_true(self) -> None:
        """Test is_staking_kpi_met with True."""
        db = MagicMock()
        db.get.return_value = True
        sd = SynchronizedData(db=db)
        assert sd.is_staking_kpi_met is True

    def test_participant_to_staking(self) -> None:
        """Test participant_to_staking property."""
        db = MagicMock()
        serialized = {"addr1": '{"payload": "data"}'}
        db.get_strict.return_value = serialized
        sd = SynchronizedData(db=db)
        with patch.object(
            CollectionRound,
            "deserialize_collection",
            return_value={"addr1": "data"},
        ):
            # _get_deserialized calls db.get_strict then deserialize_collection
            result = sd.participant_to_staking
            assert result == {"addr1": "data"}

    def test_mech_requests_empty(self) -> None:
        """Test mech_requests when empty."""
        db = MagicMock()
        db.get.return_value = "[]"
        sd = SynchronizedData(db=db)
        assert sd.mech_requests == []

    def test_mech_requests_none(self) -> None:
        """Test mech_requests when None."""
        db = MagicMock()
        db.get.return_value = None
        sd = SynchronizedData(db=db)
        assert sd.mech_requests == []

    def test_mech_responses_empty(self) -> None:
        """Test mech_responses when empty."""
        db = MagicMock()
        db.get.return_value = "[]"
        sd = SynchronizedData(db=db)
        assert sd.mech_responses == []

    def test_mech_responses_none(self) -> None:
        """Test mech_responses when None (via string)."""
        db = MagicMock()
        db.get.return_value = None
        sd = SynchronizedData(db=db)
        assert sd.mech_responses == []

    def test_mech_responses_already_list(self) -> None:
        """Test mech_responses when already deserialized as list."""
        db = MagicMock()
        db.get.return_value = []
        sd = SynchronizedData(db=db)
        assert sd.mech_responses == []

    def test_tx_loop_count_default(self) -> None:
        """Test tx_loop_count default."""
        db = MagicMock()
        db.get.return_value = 0
        sd = SynchronizedData(db=db)
        assert sd.tx_loop_count == 0

    def test_tx_loop_count_set(self) -> None:
        """Test tx_loop_count set."""
        db = MagicMock()
        db.get.return_value = 3
        sd = SynchronizedData(db=db)
        assert sd.tx_loop_count == 3

    def test_mech_for_twitter_default(self) -> None:
        """Test mech_for_twitter default."""
        db = MagicMock()
        db.get.return_value = False
        sd = SynchronizedData(db=db)
        assert sd.mech_for_twitter is False

    def test_mech_for_twitter_true(self) -> None:
        """Test mech_for_twitter true."""
        db = MagicMock()
        db.get.return_value = True
        sd = SynchronizedData(db=db)
        assert sd.mech_for_twitter is True

    def test_failed_mech_default(self) -> None:
        """Test failed_mech default."""
        db = MagicMock()
        db.get.return_value = False
        sd = SynchronizedData(db=db)
        assert sd.failed_mech is False

    def test_failed_mech_true(self) -> None:
        """Test failed_mech true."""
        db = MagicMock()
        db.get.return_value = True
        sd = SynchronizedData(db=db)
        assert sd.failed_mech is True

    def test_check_funds_count_default(self) -> None:
        """Test check_funds_count default."""
        db = MagicMock()
        db.get.return_value = 0
        sd = SynchronizedData(db=db)
        assert sd.check_funds_count == 0

    def test_check_funds_count_set(self) -> None:
        """Test check_funds_count set."""
        db = MagicMock()
        db.get.return_value = 5
        sd = SynchronizedData(db=db)
        assert sd.check_funds_count == 5

    def test_agent_details_default(self) -> None:
        """Test agent_details default."""
        db = MagicMock()
        db.get.return_value = "{}"
        sd = SynchronizedData(db=db)
        assert sd.agent_details == {}

    def test_agent_details_with_data(self) -> None:
        """Test agent_details with data."""
        db = MagicMock()
        db.get.return_value = '{"twitter_username": "bot"}'
        sd = SynchronizedData(db=db)
        assert sd.agent_details == {"twitter_username": "bot"}

    def test_participants_to_db(self) -> None:
        """Test participants_to_db property."""
        db = MagicMock()
        serialized = {"addr1": '{"payload": "data"}'}
        db.get_strict.return_value = serialized
        sd = SynchronizedData(db=db)
        with patch.object(
            CollectionRound,
            "deserialize_collection",
            return_value={"addr1": "data"},
        ):
            result = sd.participants_to_db
            assert result == {"addr1": "data"}


class TestDataclassEncoder:
    """Tests for DataclassEncoder."""

    def test_encode_dataclass(self) -> None:
        """Test encoding a dataclass."""

        @dataclass
        class Sample:
            """Sample dataclass for testing."""

            name: str
            value: int

        obj = Sample(name="test", value=42)
        result = json.dumps(obj, cls=DataclassEncoder)
        assert json.loads(result) == {"name": "test", "value": 42}

    def test_encode_non_dataclass(self) -> None:
        """Test encoding a non-dataclass falls back to default."""
        encoder = DataclassEncoder()
        with pytest.raises(TypeError):
            encoder.default(set())

    def test_encode_regular_types(self) -> None:
        """Test encoding regular types works normally."""
        result = json.dumps({"key": "value", "num": 42}, cls=DataclassEncoder)
        assert json.loads(result) == {"key": "value", "num": 42}

    def test_encode_dataclass_type_not_instance(self) -> None:
        """Test that passing a dataclass type (not instance) falls back."""

        @dataclass
        class Sample:
            """Sample dataclass for testing."""

            name: str

        encoder = DataclassEncoder()
        with pytest.raises(TypeError):
            encoder.default(Sample)


class TestEventRoundBase:
    """Tests for EventRoundBase."""

    def test_payload_class(self) -> None:
        """Test default payload class."""
        assert EventRoundBase.payload_class is BaseTxPayload

    def test_synchronized_data_class(self) -> None:
        """Test synchronized data class."""
        assert EventRoundBase.synchronized_data_class is SynchronizedData

    def test_end_block_threshold_reached(self) -> None:
        """Test end_block when threshold is reached."""
        round_instance = EventRoundBase.__new__(EventRoundBase)
        mock_data = MagicMock()

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=True,
        ), patch.object(
            type(round_instance),
            "most_voted_payload",
            new_callable=PropertyMock,
            return_value="done",
        ):
            result = round_instance.end_block()
            assert result is not None
            assert result[1] == Event.DONE

    def test_end_block_no_majority_possible(self) -> None:
        """Test end_block when no majority is possible."""
        round_instance = EventRoundBase.__new__(EventRoundBase)
        mock_data = MagicMock()
        mock_data.nb_participants = 4
        round_instance.collection = {}

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=False,
        ), patch.object(
            EventRoundBase,
            "is_majority_possible",
            return_value=False,
        ):
            result = round_instance.end_block()
            assert result is not None
            assert result[1] == Event.NO_MAJORITY

    def test_end_block_none(self) -> None:
        """Test end_block returns None when waiting."""
        round_instance = EventRoundBase.__new__(EventRoundBase)
        mock_data = MagicMock()
        mock_data.nb_participants = 4
        round_instance.collection = {}

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=False,
        ), patch.object(
            EventRoundBase,
            "is_majority_possible",
            return_value=True,
        ):
            result = round_instance.end_block()
            assert result is None


class TestLoadDatabaseRound:
    """Tests for LoadDatabaseRound."""

    def test_payload_class(self) -> None:
        """Test payload class."""
        assert LoadDatabaseRound.payload_class is LoadDatabasePayload

    def test_synchronized_data_class(self) -> None:
        """Test synchronized data class."""
        assert LoadDatabaseRound.synchronized_data_class is SynchronizedData

    def test_end_block_threshold_done(self) -> None:
        """Test end_block returns DONE when twitter_username present."""
        round_instance = LoadDatabaseRound.__new__(LoadDatabaseRound)
        mock_data = MagicMock()
        mock_data.update.return_value = MagicMock()

        agent_details = json.dumps({"twitter_username": "bot"})
        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=True,
        ), patch.object(
            type(round_instance),
            "most_voted_payload_values",
            new_callable=PropertyMock,
            return_value=("persona", 24, 3600, agent_details),
        ):
            result = round_instance.end_block()
            assert result is not None
            assert result[1] == Event.DONE

    def test_end_block_threshold_invalid_auth(self) -> None:
        """Test end_block returns INVALID_AUTH when twitter_username is None."""
        round_instance = LoadDatabaseRound.__new__(LoadDatabaseRound)
        mock_data = MagicMock()
        mock_data.update.return_value = MagicMock()

        agent_details = json.dumps({"twitter_username": None})
        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=True,
        ), patch.object(
            type(round_instance),
            "most_voted_payload_values",
            new_callable=PropertyMock,
            return_value=("persona", 24, 3600, agent_details),
        ):
            result = round_instance.end_block()
            assert result is not None
            assert result[1] == Event.INVALID_AUTH

    def test_end_block_no_majority(self) -> None:
        """Test end_block returns NO_MAJORITY."""
        round_instance = LoadDatabaseRound.__new__(LoadDatabaseRound)
        mock_data = MagicMock()
        mock_data.nb_participants = 4
        round_instance.collection = {}

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=False,
        ), patch.object(
            LoadDatabaseRound,
            "is_majority_possible",
            return_value=False,
        ):
            result = round_instance.end_block()
            assert result is not None
            assert result[1] == Event.NO_MAJORITY

    def test_end_block_none(self) -> None:
        """Test end_block returns None when still collecting."""
        round_instance = LoadDatabaseRound.__new__(LoadDatabaseRound)
        mock_data = MagicMock()
        mock_data.nb_participants = 4
        round_instance.collection = {}

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=False,
        ), patch.object(
            LoadDatabaseRound,
            "is_majority_possible",
            return_value=True,
        ):
            result = round_instance.end_block()
            assert result is None


class TestCheckStakingRound:
    """Tests for CheckStakingRound."""

    def test_payload_class(self) -> None:
        """Test payload class."""
        assert CheckStakingRound.payload_class is CheckStakingPayload

    def test_end_block_done(self) -> None:
        """Test end_block returns DONE when not skipping."""
        round_instance = CheckStakingRound.__new__(CheckStakingRound)
        mock_data = MagicMock()
        mock_data.update.return_value = MagicMock()
        round_instance.context = MagicMock()
        round_instance.context.params.stop_posting_if_staking_kpi_met = False

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=True,
        ), patch.object(
            type(round_instance),
            "most_voted_payload_values",
            new_callable=PropertyMock,
            return_value=(True,),
        ):
            result = round_instance.end_block()
            assert result is not None
            assert result[1] == Event.DONE

    def test_end_block_skip(self) -> None:
        """Test end_block returns SKIP when staking KPI met and stop_posting enabled."""
        round_instance = CheckStakingRound.__new__(CheckStakingRound)
        mock_data = MagicMock()
        mock_data.update.return_value = MagicMock()
        round_instance.context = MagicMock()
        round_instance.context.params.stop_posting_if_staking_kpi_met = True

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=True,
        ), patch.object(
            type(round_instance),
            "most_voted_payload_values",
            new_callable=PropertyMock,
            return_value=(True,),
        ):
            result = round_instance.end_block()
            assert result is not None
            assert result[1] == Event.SKIP

    def test_end_block_no_majority(self) -> None:
        """Test end_block returns NO_MAJORITY."""
        round_instance = CheckStakingRound.__new__(CheckStakingRound)
        mock_data = MagicMock()
        mock_data.nb_participants = 4
        round_instance.collection = {}

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=False,
        ), patch.object(
            CheckStakingRound,
            "is_majority_possible",
            return_value=False,
        ):
            result = round_instance.end_block()
            assert result is not None
            assert result[1] == Event.NO_MAJORITY

    def test_end_block_none(self) -> None:
        """Test end_block returns None when waiting."""
        round_instance = CheckStakingRound.__new__(CheckStakingRound)
        mock_data = MagicMock()
        mock_data.nb_participants = 4
        round_instance.collection = {}

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=False,
        ), patch.object(
            CheckStakingRound,
            "is_majority_possible",
            return_value=True,
        ):
            result = round_instance.end_block()
            assert result is None


class TestPullMemesRound:
    """Tests for PullMemesRound."""

    def test_payload_class(self) -> None:
        """Test payload class."""
        assert PullMemesRound.payload_class is PullMemesPayload

    def test_end_block_done(self) -> None:
        """Test end_block returns DONE with meme coins."""
        round_instance = PullMemesRound.__new__(PullMemesRound)
        mock_data = MagicMock()
        mock_data.update.return_value = MagicMock()

        coins_json = json.dumps([{"name": "test"}])
        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=True,
        ), patch.object(
            type(round_instance),
            "most_voted_payload_values",
            new_callable=PropertyMock,
            return_value=(coins_json, "done"),
        ), patch.object(
            type(round_instance),
            "most_voted_payload",
            new_callable=PropertyMock,
            return_value=coins_json,
        ):
            result = round_instance.end_block()
            assert result is not None
            assert result[1] == Event.DONE

    def test_end_block_skip(self) -> None:
        """Test end_block returns SKIP."""
        round_instance = PullMemesRound.__new__(PullMemesRound)
        mock_data = MagicMock()

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=True,
        ), patch.object(
            type(round_instance),
            "most_voted_payload_values",
            new_callable=PropertyMock,
            return_value=(None, "skip"),
        ):
            result = round_instance.end_block()
            assert result is not None
            assert result[1] == Event.SKIP

    def test_end_block_done_none_payload(self) -> None:
        """Test end_block with None most_voted_payload."""
        round_instance = PullMemesRound.__new__(PullMemesRound)
        mock_data = MagicMock()
        mock_data.update.return_value = MagicMock()

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=True,
        ), patch.object(
            type(round_instance),
            "most_voted_payload_values",
            new_callable=PropertyMock,
            return_value=(None, "done"),
        ), patch.object(
            type(round_instance),
            "most_voted_payload",
            new_callable=PropertyMock,
            return_value=None,
        ):
            result = round_instance.end_block()
            assert result is not None
            assert result[1] == Event.DONE

    def test_end_block_no_majority(self) -> None:
        """Test end_block returns NO_MAJORITY."""
        round_instance = PullMemesRound.__new__(PullMemesRound)
        mock_data = MagicMock()
        mock_data.nb_participants = 4
        round_instance.collection = {}

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=False,
        ), patch.object(
            PullMemesRound,
            "is_majority_possible",
            return_value=False,
        ):
            result = round_instance.end_block()
            assert result is not None
            assert result[1] == Event.NO_MAJORITY

    def test_end_block_none(self) -> None:
        """Test end_block returns None."""
        round_instance = PullMemesRound.__new__(PullMemesRound)
        mock_data = MagicMock()
        mock_data.nb_participants = 4
        round_instance.collection = {}

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=False,
        ), patch.object(
            PullMemesRound,
            "is_majority_possible",
            return_value=True,
        ):
            result = round_instance.end_block()
            assert result is None


class TestCollectFeedbackRound:
    """Tests for CollectFeedbackRound."""

    def test_payload_class(self) -> None:
        """Test payload class."""
        assert CollectFeedbackRound.payload_class is CollectFeedbackPayload

    def test_end_block_done(self) -> None:
        """Test end_block returns DONE."""
        round_instance = CollectFeedbackRound.__new__(CollectFeedbackRound)
        mock_data = MagicMock()
        mock_data.update.return_value = MagicMock()

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=True,
        ), patch.object(
            type(round_instance),
            "most_voted_payload",
            new_callable=PropertyMock,
            return_value=json.dumps(["feedback1"]),
        ):
            result = round_instance.end_block()
            assert result is not None
            assert result[1] == Event.DONE

    def test_end_block_error(self) -> None:
        """Test end_block returns ERROR when feedback is null."""
        round_instance = CollectFeedbackRound.__new__(CollectFeedbackRound)
        mock_data = MagicMock()

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=True,
        ), patch.object(
            type(round_instance),
            "most_voted_payload",
            new_callable=PropertyMock,
            return_value="null",
        ):
            result = round_instance.end_block()
            assert result is not None
            assert result[1] == Event.ERROR

    def test_end_block_no_majority(self) -> None:
        """Test end_block returns NO_MAJORITY."""
        round_instance = CollectFeedbackRound.__new__(CollectFeedbackRound)
        mock_data = MagicMock()
        mock_data.nb_participants = 4
        round_instance.collection = {}

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=False,
        ), patch.object(
            CollectFeedbackRound,
            "is_majority_possible",
            return_value=False,
        ):
            result = round_instance.end_block()
            assert result is not None
            assert result[1] == Event.NO_MAJORITY

    def test_end_block_none(self) -> None:
        """Test end_block returns None."""
        round_instance = CollectFeedbackRound.__new__(CollectFeedbackRound)
        mock_data = MagicMock()
        mock_data.nb_participants = 4
        round_instance.collection = {}

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=False,
        ), patch.object(
            CollectFeedbackRound,
            "is_majority_possible",
            return_value=True,
        ):
            result = round_instance.end_block()
            assert result is None


class TestEngageTwitterRound:
    """Tests for EngageTwitterRound."""

    def test_payload_class(self) -> None:
        """Test payload class."""
        assert EngageTwitterRound.payload_class is EngageTwitterPayload

    def test_end_block_done(self) -> None:
        """Test end_block returns DONE."""
        round_instance = EngageTwitterRound.__new__(EngageTwitterRound)
        mock_data = MagicMock()
        mock_data.update.return_value = MagicMock()
        round_instance.context = MagicMock()

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=True,
        ), patch.object(
            type(round_instance),
            "most_voted_payload_values",
            new_callable=PropertyMock,
            return_value=("done", None, "submitter", False),
        ):
            result = round_instance.end_block()
            assert result is not None
            assert result[1] == Event.DONE

    def test_end_block_mech_with_request(self) -> None:
        """Test end_block returns MECH with valid request."""
        round_instance = EngageTwitterRound.__new__(EngageTwitterRound)
        mock_data = MagicMock()
        mock_data.update.return_value = MagicMock()
        round_instance.context = MagicMock()

        mech_request = json.dumps([{"key": "value"}])
        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=True,
        ), patch.object(
            type(round_instance),
            "most_voted_payload_values",
            new_callable=PropertyMock,
            return_value=("mech", mech_request, "submitter", False),
        ):
            result = round_instance.end_block()
            assert result is not None
            assert result[1] == Event.MECH

    def test_end_block_mech_invalid_json(self) -> None:
        """Test end_block returns ERROR on invalid mech JSON."""
        round_instance = EngageTwitterRound.__new__(EngageTwitterRound)
        mock_data = MagicMock()
        round_instance.context = MagicMock()

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=True,
        ), patch.object(
            type(round_instance),
            "most_voted_payload_values",
            new_callable=PropertyMock,
            return_value=("mech", "invalid_json{", "submitter", False),
        ):
            result = round_instance.end_block()
            assert result is not None
            assert result[1] == Event.ERROR

    def test_end_block_invalid_auth(self) -> None:
        """Test end_block returns INVALID_AUTH event."""
        round_instance = EngageTwitterRound.__new__(EngageTwitterRound)
        mock_data = MagicMock()
        mock_data.update.return_value = MagicMock()
        round_instance.context = MagicMock()

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=True,
        ), patch.object(
            type(round_instance),
            "most_voted_payload_values",
            new_callable=PropertyMock,
            return_value=("invalid_auth", None, "submitter", False),
        ):
            result = round_instance.end_block()
            assert result is not None
            assert result[1] == Event.INVALID_AUTH

    def test_end_block_no_majority(self) -> None:
        """Test end_block returns NO_MAJORITY."""
        round_instance = EngageTwitterRound.__new__(EngageTwitterRound)
        mock_data = MagicMock()
        mock_data.nb_participants = 4
        round_instance.collection = {}

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=False,
        ), patch.object(
            EngageTwitterRound,
            "is_majority_possible",
            return_value=False,
        ):
            result = round_instance.end_block()
            assert result is not None
            assert result[1] == Event.NO_MAJORITY

    def test_end_block_none(self) -> None:
        """Test end_block returns None."""
        round_instance = EngageTwitterRound.__new__(EngageTwitterRound)
        mock_data = MagicMock()
        mock_data.nb_participants = 4
        round_instance.collection = {}

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=False,
        ), patch.object(
            EngageTwitterRound,
            "is_majority_possible",
            return_value=True,
        ):
            result = round_instance.end_block()
            assert result is None

    def test_end_block_mech_no_request(self) -> None:
        """Test end_block MECH event but mech_request is None."""
        round_instance = EngageTwitterRound.__new__(EngageTwitterRound)
        mock_data = MagicMock()
        mock_data.update.return_value = MagicMock()
        round_instance.context = MagicMock()

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=True,
        ), patch.object(
            type(round_instance),
            "most_voted_payload_values",
            new_callable=PropertyMock,
            return_value=("mech", None, "submitter", False),
        ):
            result = round_instance.end_block()
            assert result is not None
            # When mech_request is None, falls through to the non-mech path
            assert result[1] == Event.MECH


class TestMechRoundBase:
    """Tests for MechRoundBase."""

    def test_payload_class(self) -> None:
        """Test payload class."""
        assert MechRoundBase.payload_class is MechPayload

    def test_end_block_done(self) -> None:
        """Test end_block returns DONE."""
        round_instance = MechRoundBase.__new__(MechRoundBase)
        mock_data = MagicMock()
        mock_data.update.return_value = MagicMock()
        round_instance.context = MagicMock()

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=True,
        ), patch.object(
            type(round_instance),
            "most_voted_payload_values",
            new_callable=PropertyMock,
            return_value=(True, False),
        ):
            result = round_instance.end_block()
            assert result is not None
            assert result[1] == Event.DONE

    def test_end_block_no_majority(self) -> None:
        """Test end_block returns NO_MAJORITY."""
        round_instance = MechRoundBase.__new__(MechRoundBase)
        mock_data = MagicMock()
        mock_data.nb_participants = 4
        round_instance.collection = {}

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=False,
        ), patch.object(
            MechRoundBase,
            "is_majority_possible",
            return_value=False,
        ):
            result = round_instance.end_block()
            assert result is not None
            assert result[1] == Event.NO_MAJORITY

    def test_end_block_none(self) -> None:
        """Test end_block returns None when waiting."""
        round_instance = MechRoundBase.__new__(MechRoundBase)
        mock_data = MagicMock()
        mock_data.nb_participants = 4
        round_instance.collection = {}

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=False,
        ), patch.object(
            MechRoundBase,
            "is_majority_possible",
            return_value=True,
        ):
            result = round_instance.end_block()
            assert result is None


class TestPostMechResponseRound:
    """Tests for PostMechResponseRound."""

    def test_is_subclass(self) -> None:
        """Test it is a subclass of MechRoundBase."""
        assert issubclass(PostMechResponseRound, MechRoundBase)

    def test_payload_class(self) -> None:
        """Test payload class inherited from MechRoundBase."""
        assert PostMechResponseRound.payload_class is MechPayload


class TestFailedMechRequestRound:
    """Tests for FailedMechRequestRound."""

    def test_is_subclass(self) -> None:
        """Test it is a subclass of MechRoundBase."""
        assert issubclass(FailedMechRequestRound, MechRoundBase)


class TestFailedMechResponseRound:
    """Tests for FailedMechResponseRound."""

    def test_is_subclass(self) -> None:
        """Test it is a subclass of MechRoundBase."""
        assert issubclass(FailedMechResponseRound, MechRoundBase)


class TestActionDecisionRound:
    """Tests for ActionDecisionRound."""

    def test_payload_class(self) -> None:
        """Test payload class."""
        assert ActionDecisionRound.payload_class is ActionDecisionPayload

    def test_end_block_done_with_new_persona(self) -> None:
        """Test end_block returns DONE with new persona update."""
        round_instance = ActionDecisionRound.__new__(ActionDecisionRound)
        mock_data = MagicMock()
        mock_sd = MagicMock()
        mock_sd.update.return_value = mock_sd
        mock_data.update.return_value = mock_sd

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=True,
        ), patch.object(
            type(round_instance),
            "most_voted_payload_values",
            new_callable=PropertyMock,
            return_value=(
                "done",
                "summon",
                "0xaddr",
                1,
                "TestToken",
                "TT",
                1000000,
                0.5,
                "tweet",
                "new_persona",
                1234567890.0,
            ),
        ):
            result = round_instance.end_block()
            assert result is not None
            assert result[1] == Event.DONE

    def test_end_block_done_without_new_persona(self) -> None:
        """Test end_block returns DONE without persona update."""
        round_instance = ActionDecisionRound.__new__(ActionDecisionRound)
        mock_data = MagicMock()
        mock_data.update.return_value = MagicMock()

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=True,
        ), patch.object(
            type(round_instance),
            "most_voted_payload_values",
            new_callable=PropertyMock,
            return_value=(
                "done",
                "summon",
                "0xaddr",
                1,
                "TestToken",
                "TT",
                1000000,
                0.5,
                "tweet",
                None,
                1234567890.0,
            ),
        ):
            result = round_instance.end_block()
            assert result is not None
            assert result[1] == Event.DONE

    def test_end_block_wait(self) -> None:
        """Test end_block returns WAIT."""
        round_instance = ActionDecisionRound.__new__(ActionDecisionRound)
        mock_data = MagicMock()

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=True,
        ), patch.object(
            type(round_instance),
            "most_voted_payload_values",
            new_callable=PropertyMock,
            return_value=(
                "wait",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                0,
            ),
        ):
            result = round_instance.end_block()
            assert result is not None
            assert result[1] == Event.WAIT

    def test_end_block_no_majority(self) -> None:
        """Test end_block returns NO_MAJORITY."""
        round_instance = ActionDecisionRound.__new__(ActionDecisionRound)
        mock_data = MagicMock()
        mock_data.nb_participants = 4
        round_instance.collection = {}

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=False,
        ), patch.object(
            ActionDecisionRound,
            "is_majority_possible",
            return_value=False,
        ):
            result = round_instance.end_block()
            assert result is not None
            assert result[1] == Event.NO_MAJORITY

    def test_end_block_none(self) -> None:
        """Test end_block returns None."""
        round_instance = ActionDecisionRound.__new__(ActionDecisionRound)
        mock_data = MagicMock()
        mock_data.nb_participants = 4
        round_instance.collection = {}

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=False,
        ), patch.object(
            ActionDecisionRound,
            "is_majority_possible",
            return_value=True,
        ):
            result = round_instance.end_block()
            assert result is None


class TestActionPreparationRound:
    """Tests for ActionPreparationRound."""

    def test_payload_class(self) -> None:
        """Test payload class."""
        assert ActionPreparationRound.payload_class is ActionPreparationPayload

    def test_end_block_error(self) -> None:
        """Test end_block returns ERROR when tx_hash is None."""
        round_instance = ActionPreparationRound.__new__(ActionPreparationRound)
        mock_data = MagicMock()

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=True,
        ), patch.object(
            type(round_instance),
            "most_voted_payload_values",
            new_callable=PropertyMock,
            return_value=(None, "submitter"),
        ):
            result = round_instance.end_block()
            assert result is not None
            assert result[1] == Event.ERROR

    def test_end_block_done(self) -> None:
        """Test end_block returns DONE when tx_hash is empty string."""
        round_instance = ActionPreparationRound.__new__(ActionPreparationRound)
        mock_data = MagicMock()
        mock_data.update.return_value = MagicMock()

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=True,
        ), patch.object(
            type(round_instance),
            "most_voted_payload_values",
            new_callable=PropertyMock,
            return_value=("", "submitter"),
        ):
            result = round_instance.end_block()
            assert result is not None
            assert result[1] == Event.DONE

    def test_end_block_settle(self) -> None:
        """Test end_block returns SETTLE when tx_hash is non-empty."""
        round_instance = ActionPreparationRound.__new__(ActionPreparationRound)
        mock_data = MagicMock()
        mock_data.update.return_value = MagicMock()

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=True,
        ), patch.object(
            type(round_instance),
            "most_voted_payload_values",
            new_callable=PropertyMock,
            return_value=("0xhash123", "submitter"),
        ):
            result = round_instance.end_block()
            assert result is not None
            assert result[1] == Event.SETTLE

    def test_end_block_no_majority(self) -> None:
        """Test end_block returns NO_MAJORITY."""
        round_instance = ActionPreparationRound.__new__(ActionPreparationRound)
        mock_data = MagicMock()
        mock_data.nb_participants = 4
        round_instance.collection = {}

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=False,
        ), patch.object(
            ActionPreparationRound,
            "is_majority_possible",
            return_value=False,
        ):
            result = round_instance.end_block()
            assert result is not None
            assert result[1] == Event.NO_MAJORITY

    def test_end_block_none(self) -> None:
        """Test end_block returns None."""
        round_instance = ActionPreparationRound.__new__(ActionPreparationRound)
        mock_data = MagicMock()
        mock_data.nb_participants = 4
        round_instance.collection = {}

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=False,
        ), patch.object(
            ActionPreparationRound,
            "is_majority_possible",
            return_value=True,
        ):
            result = round_instance.end_block()
            assert result is None


class TestActionTweetRound:
    """Tests for ActionTweetRound."""

    def test_is_subclass(self) -> None:
        """Test it is a subclass of EventRoundBase."""
        assert issubclass(ActionTweetRound, EventRoundBase)

    def test_payload_class(self) -> None:
        """Test payload class."""
        assert ActionTweetRound.payload_class is ActionTweetPayload


class TestCheckFundsRound:
    """Tests for CheckFundsRound."""

    def test_payload_class(self) -> None:
        """Test payload class."""
        assert CheckFundsRound.payload_class is CheckFundsPayload

    def test_end_block_done(self) -> None:
        """Test end_block returns DONE."""
        round_instance = CheckFundsRound.__new__(CheckFundsRound)
        mock_data = MagicMock()
        round_instance.context = MagicMock()

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=True,
        ), patch.object(
            type(round_instance),
            "most_voted_payload_values",
            new_callable=PropertyMock,
            return_value=("done", 0),
        ):
            result = round_instance.end_block()
            assert result is not None
            assert result[1] == Event.DONE

    def test_end_block_no_funds_below_max(self) -> None:
        """Test end_block returns NO_FUNDS when count below max."""
        round_instance = CheckFundsRound.__new__(CheckFundsRound)
        mock_data = MagicMock()
        mock_data.update.return_value = MagicMock()
        round_instance.context = MagicMock()

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=True,
        ), patch.object(
            type(round_instance),
            "most_voted_payload_values",
            new_callable=PropertyMock,
            return_value=("no_funds", 5),
        ):
            result = round_instance.end_block()
            assert result is not None
            assert result[1] == Event.NO_FUNDS

    def test_end_block_skip_at_max(self) -> None:
        """Test end_block returns SKIP when count >= max."""
        round_instance = CheckFundsRound.__new__(CheckFundsRound)
        mock_data = MagicMock()
        round_instance.context = MagicMock()

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=True,
        ), patch.object(
            type(round_instance),
            "most_voted_payload_values",
            new_callable=PropertyMock,
            return_value=("no_funds", MAX_CHECK_FUNDS_COUNT),
        ):
            result = round_instance.end_block()
            assert result is not None
            assert result[1] == Event.SKIP

    def test_end_block_skip_above_max(self) -> None:
        """Test end_block returns SKIP when count > max."""
        round_instance = CheckFundsRound.__new__(CheckFundsRound)
        mock_data = MagicMock()
        round_instance.context = MagicMock()

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=True,
        ), patch.object(
            type(round_instance),
            "most_voted_payload_values",
            new_callable=PropertyMock,
            return_value=("no_funds", MAX_CHECK_FUNDS_COUNT + 5),
        ):
            result = round_instance.end_block()
            assert result is not None
            assert result[1] == Event.SKIP

    def test_end_block_none(self) -> None:
        """Test end_block returns None when threshold not reached."""
        round_instance = CheckFundsRound.__new__(CheckFundsRound)
        mock_data = MagicMock()

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=False,
        ):
            result = round_instance.end_block()
            assert result is None


class TestPostTxDecisionMakingRound:
    """Tests for PostTxDecisionMakingRound."""

    def test_is_subclass(self) -> None:
        """Test it is a subclass of EventRoundBase."""
        assert issubclass(PostTxDecisionMakingRound, EventRoundBase)

    def test_payload_class(self) -> None:
        """Test payload class."""
        assert PostTxDecisionMakingRound.payload_class is PostTxDecisionMakingPayload


class TestCallCheckpointRound:
    """Tests for CallCheckpointRound."""

    def test_payload_class(self) -> None:
        """Test payload class."""
        assert CallCheckpointRound.payload_class is CallCheckpointPayload

    def test_end_block_done(self) -> None:
        """Test end_block returns DONE when tx_hash is None."""
        round_instance = CallCheckpointRound.__new__(CallCheckpointRound)
        mock_data = MagicMock()

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=True,
        ), patch.object(
            type(round_instance),
            "most_voted_payload_values",
            new_callable=PropertyMock,
            return_value=("submitter", None),
        ):
            result = round_instance.end_block()
            assert result is not None
            assert result[1] == Event.DONE

    def test_end_block_settle(self) -> None:
        """Test end_block returns SETTLE when tx_hash is present."""
        round_instance = CallCheckpointRound.__new__(CallCheckpointRound)
        mock_data = MagicMock()
        mock_data.update.return_value = MagicMock()

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=True,
        ), patch.object(
            type(round_instance),
            "most_voted_payload_values",
            new_callable=PropertyMock,
            return_value=("submitter", "0xhash"),
        ):
            result = round_instance.end_block()
            assert result is not None
            assert result[1] == Event.SETTLE

    def test_end_block_no_majority(self) -> None:
        """Test end_block returns NO_MAJORITY."""
        round_instance = CallCheckpointRound.__new__(CallCheckpointRound)
        mock_data = MagicMock()
        mock_data.nb_participants = 4
        round_instance.collection = {}

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=False,
        ), patch.object(
            CallCheckpointRound,
            "is_majority_possible",
            return_value=False,
        ):
            result = round_instance.end_block()
            assert result is not None
            assert result[1] == Event.NO_MAJORITY

    def test_end_block_none(self) -> None:
        """Test end_block returns None."""
        round_instance = CallCheckpointRound.__new__(CallCheckpointRound)
        mock_data = MagicMock()
        mock_data.nb_participants = 4
        round_instance.collection = {}

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=False,
        ), patch.object(
            CallCheckpointRound,
            "is_majority_possible",
            return_value=True,
        ):
            result = round_instance.end_block()
            assert result is None


class TestTransactionLoopCheckRound:
    """Tests for TransactionLoopCheckRound."""

    def test_payload_class(self) -> None:
        """Test payload class."""
        assert TransactionLoopCheckRound.payload_class is TransactionLoopCheckPayload

    def test_end_block_done(self) -> None:
        """Test end_block returns DONE when counter >= max."""
        round_instance = TransactionLoopCheckRound.__new__(TransactionLoopCheckRound)
        mock_data = MagicMock()
        round_instance.context = MagicMock()
        round_instance.context.params.tx_loop_breaker_count = 5

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=True,
        ), patch.object(
            type(round_instance),
            "most_voted_payload_values",
            new_callable=PropertyMock,
            return_value=(5,),
        ):
            result = round_instance.end_block()
            assert result is not None
            assert result[1] == Event.DONE

    def test_end_block_retry(self) -> None:
        """Test end_block returns RETRY when counter < max."""
        round_instance = TransactionLoopCheckRound.__new__(TransactionLoopCheckRound)
        mock_data = MagicMock()
        mock_data.update.return_value = MagicMock()
        round_instance.context = MagicMock()
        round_instance.context.params.tx_loop_breaker_count = 5

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=True,
        ), patch.object(
            type(round_instance),
            "most_voted_payload_values",
            new_callable=PropertyMock,
            return_value=(3,),
        ):
            result = round_instance.end_block()
            assert result is not None
            assert result[1] == Event.RETRY

    def test_end_block_no_majority(self) -> None:
        """Test end_block returns NO_MAJORITY."""
        round_instance = TransactionLoopCheckRound.__new__(TransactionLoopCheckRound)
        mock_data = MagicMock()
        mock_data.nb_participants = 4
        round_instance.collection = {}

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=False,
        ), patch.object(
            TransactionLoopCheckRound,
            "is_majority_possible",
            return_value=False,
        ):
            result = round_instance.end_block()
            assert result is not None
            assert result[1] == Event.NO_MAJORITY

    def test_end_block_none(self) -> None:
        """Test end_block returns None."""
        round_instance = TransactionLoopCheckRound.__new__(TransactionLoopCheckRound)
        mock_data = MagicMock()
        mock_data.nb_participants = 4
        round_instance.collection = {}

        with patch.object(
            type(round_instance),
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_data,
        ), patch.object(
            type(round_instance),
            "threshold_reached",
            new_callable=PropertyMock,
            return_value=False,
        ), patch.object(
            TransactionLoopCheckRound,
            "is_majority_possible",
            return_value=True,
        ):
            result = round_instance.end_block()
            assert result is None


class TestDegenerateRounds:
    """Tests for degenerate (final) round classes."""

    def test_finished_to_reset_round(self) -> None:
        """Test FinishedToResetRound is DegenerateRound."""
        assert issubclass(FinishedToResetRound, DegenerateRound)

    def test_finished_to_settlement_round(self) -> None:
        """Test FinishedToSettlementRound is DegenerateRound."""
        assert issubclass(FinishedToSettlementRound, DegenerateRound)

    def test_finished_for_mech_request_round(self) -> None:
        """Test FinishedForMechRequestRound is DegenerateRound."""
        assert issubclass(FinishedForMechRequestRound, DegenerateRound)

    def test_finished_for_mech_response_round(self) -> None:
        """Test FinishedForMechResponseRound is DegenerateRound."""
        assert issubclass(FinishedForMechResponseRound, DegenerateRound)


class TestMemeooorrAbciApp:
    """Tests for MemeooorrAbciApp."""

    def test_initial_round_cls(self) -> None:
        """Test initial round class."""
        assert MemeooorrAbciApp.initial_round_cls is LoadDatabaseRound

    def test_initial_states(self) -> None:
        """Test initial states."""
        expected = {
            LoadDatabaseRound,
            CheckStakingRound,
            PullMemesRound,
            ActionPreparationRound,
            PostTxDecisionMakingRound,
            PostMechResponseRound,
            TransactionLoopCheckRound,
            FailedMechRequestRound,
            FailedMechResponseRound,
        }
        assert MemeooorrAbciApp.initial_states == expected

    def test_final_states(self) -> None:
        """Test final states."""
        expected = {
            FinishedToResetRound,
            FinishedToSettlementRound,
            FinishedForMechRequestRound,
            FinishedForMechResponseRound,
        }
        assert MemeooorrAbciApp.final_states == expected

    def test_transition_function_keys(self) -> None:
        """Test all round classes are in transition function."""
        expected_keys = {
            LoadDatabaseRound,
            CheckStakingRound,
            PullMemesRound,
            CollectFeedbackRound,
            EngageTwitterRound,
            ActionDecisionRound,
            ActionPreparationRound,
            ActionTweetRound,
            CheckFundsRound,
            PostTxDecisionMakingRound,
            CallCheckpointRound,
            PostMechResponseRound,
            TransactionLoopCheckRound,
            FailedMechRequestRound,
            FailedMechResponseRound,
            FinishedToResetRound,
            FinishedToSettlementRound,
            FinishedForMechRequestRound,
            FinishedForMechResponseRound,
        }
        assert set(MemeooorrAbciApp.transition_function.keys()) == expected_keys

    def test_load_database_transitions(self) -> None:
        """Test LoadDatabaseRound transitions."""
        t = MemeooorrAbciApp.transition_function[LoadDatabaseRound]
        assert t[Event.DONE] is CheckStakingRound
        assert t[Event.NO_MAJORITY] is LoadDatabaseRound
        assert t[Event.ROUND_TIMEOUT] is LoadDatabaseRound
        assert t[Event.INVALID_AUTH] is CallCheckpointRound

    def test_check_staking_transitions(self) -> None:
        """Test CheckStakingRound transitions."""
        t = MemeooorrAbciApp.transition_function[CheckStakingRound]
        assert t[Event.DONE] is PullMemesRound
        assert t[Event.SKIP] is CallCheckpointRound
        assert t[Event.NO_MAJORITY] is CheckStakingRound
        assert t[Event.ROUND_TIMEOUT] is CheckStakingRound

    def test_pull_memes_transitions(self) -> None:
        """Test PullMemesRound transitions."""
        t = MemeooorrAbciApp.transition_function[PullMemesRound]
        assert t[Event.DONE] is CollectFeedbackRound
        assert t[Event.SKIP] is CollectFeedbackRound
        assert t[Event.NO_MAJORITY] is PullMemesRound
        assert t[Event.ROUND_TIMEOUT] is PullMemesRound

    def test_collect_feedback_transitions(self) -> None:
        """Test CollectFeedbackRound transitions."""
        t = MemeooorrAbciApp.transition_function[CollectFeedbackRound]
        assert t[Event.DONE] is EngageTwitterRound
        assert t[Event.ERROR] is EngageTwitterRound
        assert t[Event.NO_MAJORITY] is CollectFeedbackRound
        assert t[Event.ROUND_TIMEOUT] is CollectFeedbackRound

    def test_engage_twitter_transitions(self) -> None:
        """Test EngageTwitterRound transitions."""
        t = MemeooorrAbciApp.transition_function[EngageTwitterRound]
        assert t[Event.DONE] is ActionDecisionRound
        assert t[Event.INVALID_AUTH] is ActionDecisionRound
        assert t[Event.MECH] is FinishedForMechRequestRound
        assert t[Event.ERROR] is EngageTwitterRound
        assert t[Event.NO_MAJORITY] is EngageTwitterRound
        assert t[Event.ROUND_TIMEOUT] is EngageTwitterRound

    def test_action_decision_transitions(self) -> None:
        """Test ActionDecisionRound transitions."""
        t = MemeooorrAbciApp.transition_function[ActionDecisionRound]
        assert t[Event.DONE] is ActionPreparationRound
        assert t[Event.WAIT] is CallCheckpointRound
        assert t[Event.SKIP] is CallCheckpointRound
        assert t[Event.RETRY] is ActionDecisionRound
        assert t[Event.NO_MAJORITY] is ActionDecisionRound
        assert t[Event.ROUND_TIMEOUT] is ActionDecisionRound

    def test_action_preparation_transitions(self) -> None:
        """Test ActionPreparationRound transitions."""
        t = MemeooorrAbciApp.transition_function[ActionPreparationRound]
        assert t[Event.DONE] is ActionTweetRound
        assert t[Event.ERROR] is CallCheckpointRound
        assert t[Event.SETTLE] is CheckFundsRound
        assert t[Event.NO_MAJORITY] is ActionPreparationRound
        assert t[Event.ROUND_TIMEOUT] is ActionPreparationRound

    def test_action_tweet_transitions(self) -> None:
        """Test ActionTweetRound transitions."""
        t = MemeooorrAbciApp.transition_function[ActionTweetRound]
        assert t[Event.DONE] is CallCheckpointRound
        assert t[Event.ERROR] is CallCheckpointRound
        assert t[Event.MISSING_TWEET] is CallCheckpointRound
        assert t[Event.NO_MAJORITY] is ActionTweetRound
        assert t[Event.ROUND_TIMEOUT] is ActionTweetRound

    def test_check_funds_transitions(self) -> None:
        """Test CheckFundsRound transitions."""
        t = MemeooorrAbciApp.transition_function[CheckFundsRound]
        assert t[Event.DONE] is FinishedToSettlementRound
        assert t[Event.NO_FUNDS] is CheckFundsRound
        assert t[Event.NO_MAJORITY] is CheckFundsRound
        assert t[Event.ROUND_TIMEOUT] is CheckFundsRound
        assert t[Event.SKIP] is FinishedToResetRound

    def test_post_tx_decision_transitions(self) -> None:
        """Test PostTxDecisionMakingRound transitions."""
        t = MemeooorrAbciApp.transition_function[PostTxDecisionMakingRound]
        assert t[Event.DONE] is FinishedToResetRound
        assert t[Event.ACTION] is ActionPreparationRound
        assert t[Event.NONE] is PostTxDecisionMakingRound
        assert t[Event.NO_MAJORITY] is PostTxDecisionMakingRound
        assert t[Event.ROUND_TIMEOUT] is PostTxDecisionMakingRound
        assert t[Event.MECH] is FinishedForMechResponseRound

    def test_call_checkpoint_transitions(self) -> None:
        """Test CallCheckpointRound transitions."""
        t = MemeooorrAbciApp.transition_function[CallCheckpointRound]
        assert t[Event.DONE] is FinishedToResetRound
        assert t[Event.SETTLE] is FinishedToSettlementRound
        assert t[Event.ROUND_TIMEOUT] is CallCheckpointRound
        assert t[Event.NO_MAJORITY] is CallCheckpointRound

    def test_post_mech_response_transitions(self) -> None:
        """Test PostMechResponseRound transitions."""
        t = MemeooorrAbciApp.transition_function[PostMechResponseRound]
        assert t[Event.DONE] is EngageTwitterRound
        assert t[Event.NO_MAJORITY] is PostMechResponseRound
        assert t[Event.ROUND_TIMEOUT] is PostMechResponseRound
        assert t[Event.ERROR] is FailedMechResponseRound

    def test_transaction_loop_check_transitions(self) -> None:
        """Test TransactionLoopCheckRound transitions."""
        t = MemeooorrAbciApp.transition_function[TransactionLoopCheckRound]
        assert t[Event.DONE] is FinishedToResetRound
        assert t[Event.RETRY] is FinishedToSettlementRound
        assert t[Event.NO_MAJORITY] is TransactionLoopCheckRound
        assert t[Event.ROUND_TIMEOUT] is TransactionLoopCheckRound

    def test_failed_mech_request_transitions(self) -> None:
        """Test FailedMechRequestRound transitions."""
        t = MemeooorrAbciApp.transition_function[FailedMechRequestRound]
        assert t[Event.DONE] is EngageTwitterRound
        assert t[Event.NO_MAJORITY] is EngageTwitterRound
        assert t[Event.ROUND_TIMEOUT] is EngageTwitterRound
        assert t[Event.ERROR] is EngageTwitterRound

    def test_failed_mech_response_transitions(self) -> None:
        """Test FailedMechResponseRound transitions."""
        t = MemeooorrAbciApp.transition_function[FailedMechResponseRound]
        assert t[Event.DONE] is EngageTwitterRound
        assert t[Event.NO_MAJORITY] is EngageTwitterRound
        assert t[Event.ROUND_TIMEOUT] is EngageTwitterRound
        assert t[Event.ERROR] is EngageTwitterRound

    def test_final_round_transitions_empty(self) -> None:
        """Test final rounds have no transitions."""
        assert MemeooorrAbciApp.transition_function[FinishedToResetRound] == {}
        assert MemeooorrAbciApp.transition_function[FinishedToSettlementRound] == {}
        assert MemeooorrAbciApp.transition_function[FinishedForMechRequestRound] == {}
        assert MemeooorrAbciApp.transition_function[FinishedForMechResponseRound] == {}

    def test_event_to_timeout(self) -> None:
        """Test event timeout mapping."""
        assert Event.ROUND_TIMEOUT in MemeooorrAbciApp.event_to_timeout
        assert MemeooorrAbciApp.event_to_timeout[Event.ROUND_TIMEOUT] == 30

    def test_cross_period_persisted_keys(self) -> None:
        """Test cross-period persisted keys."""
        expected = frozenset(
            ["persona", "heart_cooldown_hours", "summon_cooldown_seconds"]
        )
        assert MemeooorrAbciApp.cross_period_persisted_keys == expected

    def test_db_pre_conditions(self) -> None:
        """Test DB pre-conditions."""
        for _, pre_conditions in MemeooorrAbciApp.db_pre_conditions.items():
            assert pre_conditions == set()

    def test_db_post_conditions(self) -> None:
        """Test DB post-conditions."""
        assert MemeooorrAbciApp.db_post_conditions[FinishedToResetRound] == set()
        assert MemeooorrAbciApp.db_post_conditions[FinishedForMechRequestRound] == set()
        assert (
            MemeooorrAbciApp.db_post_conditions[FinishedForMechResponseRound] == set()
        )
        assert MemeooorrAbciApp.db_post_conditions[FinishedToSettlementRound] == {
            "most_voted_tx_hash"
        }
