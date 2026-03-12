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

"""Tests for rounds.py using framework test infrastructure."""

# pylint: disable=too-few-public-methods,too-many-public-methods

import json
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Optional
from unittest.mock import MagicMock, patch

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
    SynchronizedData,
    TransactionLoopCheckRound,
)
from packages.dvilela.skills.memeooorr_abci.tests.conftest import MemeooorrRoundTestBase
from packages.valory.skills.abstract_round_abci.base import (
    AbciAppDB,
    CollectionRound,
    DegenerateRound,
    get_name,
)

# --- Helper functions to create payload dicts ---


def get_participant_to_load_database(
    participants: FrozenSet[str],
    persona: str = "test_persona",
    heart_cooldown_hours: int = 24,
    summon_cooldown_seconds: int = 3600,
    agent_details: str = '{"twitter_username": "bot"}',
) -> Dict[str, LoadDatabasePayload]:
    """Get participant to LoadDatabasePayload mapping."""
    return {
        p: LoadDatabasePayload(
            sender=p,
            persona=persona,
            heart_cooldown_hours=heart_cooldown_hours,
            summon_cooldown_seconds=summon_cooldown_seconds,
            agent_details=agent_details,
        )
        for p in sorted(participants)
    }


def get_participant_to_check_staking(
    participants: FrozenSet[str],
    is_staking_kpi_met: bool = False,
) -> Dict[str, CheckStakingPayload]:
    """Get participant to CheckStakingPayload mapping."""
    return {
        p: CheckStakingPayload(sender=p, is_staking_kpi_met=is_staking_kpi_met)
        for p in sorted(participants)
    }


def get_participant_to_pull_memes(
    participants: FrozenSet[str],
    meme_coins: Optional[str] = "[]",
    event: str = "done",
) -> Dict[str, PullMemesPayload]:
    """Get participant to PullMemesPayload mapping."""
    return {
        p: PullMemesPayload(sender=p, meme_coins=meme_coins, event=event)
        for p in sorted(participants)
    }


def get_participant_to_collect_feedback(
    participants: FrozenSet[str],
    feedback: str = "[]",
) -> Dict[str, CollectFeedbackPayload]:
    """Get participant to CollectFeedbackPayload mapping."""
    return {
        p: CollectFeedbackPayload(sender=p, feedback=feedback)
        for p in sorted(participants)
    }


def get_participant_to_engage_twitter(
    participants: FrozenSet[str],
    event: str = "done",
    mech_request: Optional[str] = None,
    tx_submitter: str = "submitter",
    failed_mech: bool = False,
) -> Dict[str, EngageTwitterPayload]:
    """Get participant to EngageTwitterPayload mapping."""
    return {
        p: EngageTwitterPayload(
            sender=p,
            event=event,
            mech_request=mech_request,
            tx_submitter=tx_submitter,
            failed_mech=failed_mech,
        )
        for p in sorted(participants)
    }


def get_participant_to_action_decision(
    participants: FrozenSet[str],
    event: str = "done",
    action: Optional[str] = "summon",
    token_address: Optional[str] = "0xaddr",
    token_nonce: Optional[int] = 1,
    token_name: Optional[str] = "TestToken",
    token_ticker: Optional[str] = "TT",
    token_supply: Optional[int] = 1000000,
    amount: Optional[float] = 0.5,
    tweet: Optional[str] = "tweet",
    new_persona: Optional[str] = None,
    timestamp: float = 1234567890.0,
) -> Dict[str, ActionDecisionPayload]:
    """Get participant to ActionDecisionPayload mapping."""
    return {
        p: ActionDecisionPayload(
            sender=p,
            event=event,
            action=action,
            token_address=token_address,
            token_nonce=token_nonce,
            token_name=token_name,
            token_ticker=token_ticker,
            token_supply=token_supply,
            amount=amount,
            tweet=tweet,
            new_persona=new_persona,
            timestamp=timestamp,
        )
        for p in sorted(participants)
    }


def get_participant_to_action_preparation(
    participants: FrozenSet[str],
    tx_hash: Optional[str] = "0xhash",
    tx_submitter: str = "submitter",
) -> Dict[str, ActionPreparationPayload]:
    """Get participant to ActionPreparationPayload mapping."""
    return {
        p: ActionPreparationPayload(
            sender=p, tx_hash=tx_hash, tx_submitter=tx_submitter
        )
        for p in sorted(participants)
    }


def get_participant_to_action_tweet(
    participants: FrozenSet[str],
    event: str = "done",
) -> Dict[str, ActionTweetPayload]:
    """Get participant to ActionTweetPayload mapping."""
    return {p: ActionTweetPayload(sender=p, event=event) for p in sorted(participants)}


def get_participant_to_check_funds(
    participants: FrozenSet[str],
    event: str = "done",
    check_funds_count: int = 0,
) -> Dict[str, CheckFundsPayload]:
    """Get participant to CheckFundsPayload mapping."""
    return {
        p: CheckFundsPayload(sender=p, event=event, check_funds_count=check_funds_count)
        for p in sorted(participants)
    }


def get_participant_to_post_tx_decision(
    participants: FrozenSet[str],
    event: str = "done",
) -> Dict[str, PostTxDecisionMakingPayload]:
    """Get participant to PostTxDecisionMakingPayload mapping."""
    return {
        p: PostTxDecisionMakingPayload(sender=p, event=event)
        for p in sorted(participants)
    }


def get_participant_to_call_checkpoint(
    participants: FrozenSet[str],
    tx_submitter: str = "submitter",
    tx_hash: Optional[str] = None,
) -> Dict[str, CallCheckpointPayload]:
    """Get participant to CallCheckpointPayload mapping."""
    return {
        p: CallCheckpointPayload(sender=p, tx_submitter=tx_submitter, tx_hash=tx_hash)
        for p in sorted(participants)
    }


def get_participant_to_mech(
    participants: FrozenSet[str],
    mech_for_twitter: bool = True,
    failed_mech: bool = False,
) -> Dict[str, MechPayload]:
    """Get participant to MechPayload mapping."""
    return {
        p: MechPayload(
            sender=p, mech_for_twitter=mech_for_twitter, failed_mech=failed_mech
        )
        for p in sorted(participants)
    }


def get_participant_to_tx_loop_check(
    participants: FrozenSet[str],
    counter: int = 0,
) -> Dict[str, TransactionLoopCheckPayload]:
    """Get participant to TransactionLoopCheckPayload mapping."""
    return {
        p: TransactionLoopCheckPayload(sender=p, counter=counter)
        for p in sorted(participants)
    }


# --- Enum / constant tests (kept as-is) ---


# --- SynchronizedData tests (using real AbciAppDB) ---


class TestSynchronizedData:
    """Tests for SynchronizedData."""

    def _make_sd(self, **kwargs: object) -> SynchronizedData:
        """Create SynchronizedData with given DB values."""
        setup_data: Dict = dict(
            participants=[("agent_0",)],
            all_participants=[("agent_0",)],
            consensus_threshold=[1],
        )
        # Convert tuple kwargs to lists for AbciAppDB format
        for key, val in kwargs.items():
            setup_data[key] = list(val) if isinstance(val, tuple) else [val]
        return SynchronizedData(db=AbciAppDB(setup_data=setup_data))

    def test_persona_default(self) -> None:
        """Test persona property with default."""
        sd = self._make_sd()
        assert sd.persona is None

    def test_persona_set(self) -> None:
        """Test persona property with value."""
        sd = self._make_sd(persona=("test_persona",))
        assert sd.persona == "test_persona"

    def test_heart_cooldown_hours(self) -> None:
        """Test heart_cooldown_hours property."""
        sd = self._make_sd(heart_cooldown_hours=(24,))
        assert sd.heart_cooldown_hours == 24

    def test_heart_cooldown_hours_default(self) -> None:
        """Test heart_cooldown_hours property default."""
        sd = self._make_sd()
        assert sd.heart_cooldown_hours is None

    def test_summon_cooldown_seconds(self) -> None:
        """Test summon_cooldown_seconds property."""
        sd = self._make_sd(summon_cooldown_seconds=(3600,))
        assert sd.summon_cooldown_seconds == 3600

    def test_summon_cooldown_seconds_default(self) -> None:
        """Test summon_cooldown_seconds default."""
        sd = self._make_sd()
        assert sd.summon_cooldown_seconds is None

    def test_meme_coins_default(self) -> None:
        """Test meme_coins default value."""
        sd = self._make_sd()
        assert sd.meme_coins == []

    def test_meme_coins_with_data(self) -> None:
        """Test meme_coins with data."""
        sd = self._make_sd(meme_coins=('[{"name": "test"}]',))
        assert sd.meme_coins == [{"name": "test"}]

    def test_pending_tweet_none(self) -> None:
        """Test pending_tweet when None."""
        sd = self._make_sd()
        assert sd.pending_tweet == []

    def test_pending_tweet_with_data(self) -> None:
        """Test pending_tweet with data."""
        sd = self._make_sd(pending_tweet=('["tweet1", "tweet2"]',))
        assert sd.pending_tweet == ["tweet1", "tweet2"]

    def test_feedback_none(self) -> None:
        """Test feedback when None."""
        sd = self._make_sd()
        assert sd.feedback == []

    def test_feedback_with_data(self) -> None:
        """Test feedback with data."""
        sd = self._make_sd(feedback=('["good", "bad"]',))
        assert sd.feedback == ["good", "bad"]

    def test_token_action_default(self) -> None:
        """Test token_action default."""
        sd = self._make_sd()
        assert sd.token_action == {}

    def test_token_action_with_data(self) -> None:
        """Test token_action with data."""
        sd = self._make_sd(token_action=('{"action": "summon"}',))
        assert sd.token_action == {"action": "summon"}

    def test_most_voted_tx_hash(self) -> None:
        """Test most_voted_tx_hash."""
        sd = self._make_sd(most_voted_tx_hash=("0xhash",))
        assert sd.most_voted_tx_hash == "0xhash"

    def test_final_tx_hash_none(self) -> None:
        """Test final_tx_hash when not set."""
        sd = self._make_sd()
        assert sd.final_tx_hash is None

    def test_final_tx_hash_set(self) -> None:
        """Test final_tx_hash when set."""
        sd = self._make_sd(final_tx_hash=("0xfinal",))
        assert sd.final_tx_hash == "0xfinal"

    def test_tx_submitter(self) -> None:
        """Test tx_submitter."""
        sd = self._make_sd(tx_submitter=("submitter",))
        assert sd.tx_submitter == "submitter"

    def test_is_staking_kpi_met_none(self) -> None:
        """Test is_staking_kpi_met with None."""
        sd = self._make_sd()
        assert sd.is_staking_kpi_met is False

    def test_is_staking_kpi_met_true(self) -> None:
        """Test is_staking_kpi_met with True."""
        sd = self._make_sd(is_staking_kpi_met=(True,))
        assert sd.is_staking_kpi_met is True

    def test_mech_requests_empty(self) -> None:
        """Test mech_requests when empty."""
        sd = self._make_sd()
        assert sd.mech_requests == []

    def test_mech_requests_none(self) -> None:
        """Test mech_requests when None."""
        sd = self._make_sd(mech_requests=(None,))
        assert sd.mech_requests == []

    def test_mech_responses_empty(self) -> None:
        """Test mech_responses when empty."""
        sd = self._make_sd()
        assert sd.mech_responses == []

    def test_mech_responses_none(self) -> None:
        """Test mech_responses when None."""
        sd = self._make_sd(mech_responses=(None,))
        assert sd.mech_responses == []

    def test_mech_responses_already_list(self) -> None:
        """Test mech_responses when value is a list."""
        sd = self._make_sd(mech_responses=([],))
        assert sd.mech_responses == []

    def test_tx_loop_count_default(self) -> None:
        """Test tx_loop_count default."""
        sd = self._make_sd()
        assert sd.tx_loop_count == 0

    def test_tx_loop_count_set(self) -> None:
        """Test tx_loop_count set."""
        sd = self._make_sd(tx_loop_count=(3,))
        assert sd.tx_loop_count == 3

    def test_mech_for_twitter_default(self) -> None:
        """Test mech_for_twitter default."""
        sd = self._make_sd()
        assert sd.mech_for_twitter is False

    def test_mech_for_twitter_true(self) -> None:
        """Test mech_for_twitter true."""
        sd = self._make_sd(mech_for_twitter=(True,))
        assert sd.mech_for_twitter is True

    def test_failed_mech_default(self) -> None:
        """Test failed_mech default."""
        sd = self._make_sd()
        assert sd.failed_mech is False

    def test_failed_mech_true(self) -> None:
        """Test failed_mech true."""
        sd = self._make_sd(failed_mech=(True,))
        assert sd.failed_mech is True

    def test_check_funds_count_default(self) -> None:
        """Test check_funds_count default."""
        sd = self._make_sd()
        assert sd.check_funds_count == 0

    def test_check_funds_count_set(self) -> None:
        """Test check_funds_count set."""
        sd = self._make_sd(check_funds_count=(5,))
        assert sd.check_funds_count == 5

    def test_agent_details_default(self) -> None:
        """Test agent_details default."""
        sd = self._make_sd()
        assert sd.agent_details == {}

    def test_agent_details_with_data(self) -> None:
        """Test agent_details with data."""
        sd = self._make_sd(agent_details=('{"twitter_username": "bot"}',))
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
            result = sd.participant_to_staking
            assert result == {"addr1": "data"}


# --- DataclassEncoder tests ---


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


# --- Round end_block tests using framework ---


class TestLoadDatabaseRound(MemeooorrRoundTestBase):
    """Tests for LoadDatabaseRound using framework pattern."""

    def test_round_done(self) -> None:
        """Test round reaches DONE when twitter_username present."""
        agent_details = json.dumps({"twitter_username": "bot"})
        test_round = LoadDatabaseRound(
            synchronized_data=self.synchronized_data,
            context=self.context_mock,
        )
        self._complete_run(
            self._test_round(
                test_round=test_round,
                round_payloads=get_participant_to_load_database(
                    self.participants, agent_details=agent_details
                ),
                synchronized_data_update_fn=lambda sd, _: sd.update(
                    synchronized_data_class=SynchronizedData,
                    **{
                        get_name(SynchronizedData.persona): "test_persona",
                        get_name(SynchronizedData.heart_cooldown_hours): 24,
                        get_name(SynchronizedData.summon_cooldown_seconds): 3600,
                        get_name(SynchronizedData.agent_details): agent_details,
                    },
                ),
                synchronized_data_attr_checks=[
                    lambda sd: sd.db.get("persona"),
                    lambda sd: sd.db.get("heart_cooldown_hours"),
                    lambda sd: sd.db.get("summon_cooldown_seconds"),
                    lambda sd: sd.db.get("agent_details"),
                ],
                most_voted_payload="test_persona",
                exit_event=Event.DONE,
            )
        )

    def test_round_invalid_auth(self) -> None:
        """Test round reaches INVALID_AUTH when twitter_username is None."""
        agent_details = json.dumps({"twitter_username": None})
        test_round = LoadDatabaseRound(
            synchronized_data=self.synchronized_data,
            context=self.context_mock,
        )
        self._complete_run(
            self._test_round(
                test_round=test_round,
                round_payloads=get_participant_to_load_database(
                    self.participants, agent_details=agent_details
                ),
                synchronized_data_update_fn=lambda sd, _: sd.update(
                    synchronized_data_class=SynchronizedData,
                    **{
                        get_name(SynchronizedData.persona): "test_persona",
                        get_name(SynchronizedData.heart_cooldown_hours): 24,
                        get_name(SynchronizedData.summon_cooldown_seconds): 3600,
                        get_name(SynchronizedData.agent_details): agent_details,
                    },
                ),
                synchronized_data_attr_checks=[
                    lambda sd: sd.db.get("persona"),
                    lambda sd: sd.db.get("agent_details"),
                ],
                most_voted_payload="test_persona",
                exit_event=Event.INVALID_AUTH,
            )
        )


class TestCheckStakingRound(MemeooorrRoundTestBase):
    """Tests for CheckStakingRound using framework pattern."""

    def test_round_done(self) -> None:
        """Test round reaches DONE when not skipping."""
        self.context_mock.params.stop_posting_if_staking_kpi_met = False
        test_round = CheckStakingRound(
            synchronized_data=self.synchronized_data,
            context=self.context_mock,
        )
        self._complete_run(
            self._test_round(
                test_round=test_round,
                round_payloads=get_participant_to_check_staking(
                    self.participants, is_staking_kpi_met=True
                ),
                synchronized_data_update_fn=lambda sd, _: sd.update(
                    synchronized_data_class=SynchronizedData,
                    **{
                        get_name(SynchronizedData.is_staking_kpi_met): True,
                    },
                ),
                synchronized_data_attr_checks=[
                    lambda sd: sd.db.get("is_staking_kpi_met"),
                ],
                most_voted_payload=True,
                exit_event=Event.DONE,
            )
        )

    def test_round_skip(self) -> None:
        """Test round reaches SKIP when staking KPI met and stop enabled."""
        self.context_mock.params.stop_posting_if_staking_kpi_met = True
        test_round = CheckStakingRound(
            synchronized_data=self.synchronized_data,
            context=self.context_mock,
        )
        self._complete_run(
            self._test_round(
                test_round=test_round,
                round_payloads=get_participant_to_check_staking(
                    self.participants, is_staking_kpi_met=True
                ),
                synchronized_data_update_fn=lambda sd, _: sd.update(
                    synchronized_data_class=SynchronizedData,
                    **{
                        get_name(SynchronizedData.is_staking_kpi_met): True,
                    },
                ),
                synchronized_data_attr_checks=[
                    lambda sd: sd.db.get("is_staking_kpi_met"),
                ],
                most_voted_payload=True,
                exit_event=Event.SKIP,
            )
        )


class TestPullMemesRound(MemeooorrRoundTestBase):
    """Tests for PullMemesRound using framework pattern."""

    def test_round_done(self) -> None:
        """Test round reaches DONE with meme coins."""
        coins_json = json.dumps([{"name": "test"}])
        test_round = PullMemesRound(
            synchronized_data=self.synchronized_data,
            context=self.context_mock,
        )
        self._complete_run(
            self._test_round(
                test_round=test_round,
                round_payloads=get_participant_to_pull_memes(
                    self.participants, meme_coins=coins_json, event="done"
                ),
                synchronized_data_update_fn=lambda sd, _: sd.update(
                    synchronized_data_class=SynchronizedData,
                    **{
                        get_name(SynchronizedData.meme_coins): json.dumps(
                            json.loads(coins_json), sort_keys=True
                        ),
                    },
                ),
                synchronized_data_attr_checks=[
                    lambda sd: sd.db.get("meme_coins"),
                ],
                most_voted_payload=coins_json,
                exit_event=Event.DONE,
            )
        )

    def test_round_skip(self) -> None:
        """Test round reaches SKIP."""
        test_round = PullMemesRound(
            synchronized_data=self.synchronized_data,
            context=self.context_mock,
        )
        self._complete_run(
            self._test_round(
                test_round=test_round,
                round_payloads=get_participant_to_pull_memes(
                    self.participants, meme_coins=None, event="skip"
                ),
                synchronized_data_update_fn=lambda sd, _: sd,
                synchronized_data_attr_checks=[],
                most_voted_payload=None,
                exit_event=Event.SKIP,
            )
        )

    def test_round_done_none_payload(self) -> None:
        """Test round reaches DONE with None most_voted_payload."""
        test_round = PullMemesRound(
            synchronized_data=self.synchronized_data,
            context=self.context_mock,
        )
        self._complete_run(
            self._test_round(
                test_round=test_round,
                round_payloads=get_participant_to_pull_memes(
                    self.participants, meme_coins=None, event="done"
                ),
                synchronized_data_update_fn=lambda sd, _: sd.update(
                    synchronized_data_class=SynchronizedData,
                    **{
                        get_name(SynchronizedData.meme_coins): json.dumps(
                            [], sort_keys=True
                        ),
                    },
                ),
                synchronized_data_attr_checks=[
                    lambda sd: sd.db.get("meme_coins"),
                ],
                most_voted_payload=None,
                exit_event=Event.DONE,
            )
        )


class TestCollectFeedbackRound(MemeooorrRoundTestBase):
    """Tests for CollectFeedbackRound using framework pattern."""

    def test_round_done(self) -> None:
        """Test round reaches DONE."""
        feedback_json = json.dumps(["feedback1"])
        test_round = CollectFeedbackRound(
            synchronized_data=self.synchronized_data,
            context=self.context_mock,
        )
        self._complete_run(
            self._test_round(
                test_round=test_round,
                round_payloads=get_participant_to_collect_feedback(
                    self.participants, feedback=feedback_json
                ),
                synchronized_data_update_fn=lambda sd, _: sd.update(
                    synchronized_data_class=SynchronizedData,
                    **{
                        get_name(SynchronizedData.feedback): json.dumps(
                            ["feedback1"], sort_keys=True
                        ),
                    },
                ),
                synchronized_data_attr_checks=[
                    lambda sd: sd.db.get("feedback"),
                ],
                most_voted_payload=feedback_json,
                exit_event=Event.DONE,
            )
        )

    def test_round_error(self) -> None:
        """Test round reaches ERROR when feedback is null."""
        test_round = CollectFeedbackRound(
            synchronized_data=self.synchronized_data,
            context=self.context_mock,
        )
        self._complete_run(
            self._test_round(
                test_round=test_round,
                round_payloads=get_participant_to_collect_feedback(
                    self.participants, feedback="null"
                ),
                synchronized_data_update_fn=lambda sd, _: sd,
                synchronized_data_attr_checks=[],
                most_voted_payload="null",
                exit_event=Event.ERROR,
            )
        )


class TestEngageTwitterRound(MemeooorrRoundTestBase):
    """Tests for EngageTwitterRound using framework pattern."""

    def test_round_done(self) -> None:
        """Test round reaches DONE."""
        test_round = EngageTwitterRound(
            synchronized_data=self.synchronized_data,
            context=self.context_mock,
        )
        self._complete_run(
            self._test_round(
                test_round=test_round,
                round_payloads=get_participant_to_engage_twitter(
                    self.participants, event="done"
                ),
                synchronized_data_update_fn=lambda sd, _: sd.update(
                    synchronized_data_class=SynchronizedData,
                    **{
                        get_name(SynchronizedData.mech_for_twitter): False,
                        get_name(SynchronizedData.final_tx_hash): None,
                        get_name(SynchronizedData.failed_mech): False,
                    },
                ),
                synchronized_data_attr_checks=[
                    lambda sd: sd.db.get("mech_for_twitter"),
                    lambda sd: sd.db.get("final_tx_hash"),
                    lambda sd: sd.db.get("failed_mech"),
                ],
                most_voted_payload="done",
                exit_event=Event.DONE,
            )
        )

    def test_round_mech_with_request(self) -> None:
        """Test round reaches MECH with valid mech request."""
        mech_request = json.dumps([{"key": "value"}])
        test_round = EngageTwitterRound(
            synchronized_data=self.synchronized_data,
            context=self.context_mock,
        )
        self._complete_run(
            self._test_round(
                test_round=test_round,
                round_payloads=get_participant_to_engage_twitter(
                    self.participants, event="mech", mech_request=mech_request
                ),
                synchronized_data_update_fn=lambda sd, _: sd.update(
                    synchronized_data_class=SynchronizedData,
                    **{
                        get_name(SynchronizedData.mech_requests): json.dumps(
                            [{"key": "value"}], cls=DataclassEncoder
                        ),
                    },
                ),
                synchronized_data_attr_checks=[
                    lambda sd: sd.db.get("mech_requests"),
                ],
                most_voted_payload="mech",
                exit_event=Event.MECH,
            )
        )

    def test_round_mech_invalid_json(self) -> None:
        """Test round reaches ERROR on invalid mech JSON."""
        test_round = EngageTwitterRound(
            synchronized_data=self.synchronized_data,
            context=self.context_mock,
        )
        self._complete_run(
            self._test_round(
                test_round=test_round,
                round_payloads=get_participant_to_engage_twitter(
                    self.participants, event="mech", mech_request="invalid_json{"
                ),
                synchronized_data_update_fn=lambda sd, _: sd,
                synchronized_data_attr_checks=[],
                most_voted_payload="mech",
                exit_event=Event.ERROR,
            )
        )

    def test_round_mech_no_request(self) -> None:
        """Test round with MECH event but no mech_request falls through."""
        test_round = EngageTwitterRound(
            synchronized_data=self.synchronized_data,
            context=self.context_mock,
        )
        self._complete_run(
            self._test_round(
                test_round=test_round,
                round_payloads=get_participant_to_engage_twitter(
                    self.participants, event="mech", mech_request=None
                ),
                synchronized_data_update_fn=lambda sd, _: sd.update(
                    synchronized_data_class=SynchronizedData,
                    **{
                        get_name(SynchronizedData.mech_for_twitter): False,
                        get_name(SynchronizedData.final_tx_hash): None,
                        get_name(SynchronizedData.failed_mech): False,
                    },
                ),
                synchronized_data_attr_checks=[],
                most_voted_payload="mech",
                exit_event=Event.MECH,
            )
        )

    def test_round_invalid_auth(self) -> None:
        """Test round reaches INVALID_AUTH."""
        test_round = EngageTwitterRound(
            synchronized_data=self.synchronized_data,
            context=self.context_mock,
        )
        self._complete_run(
            self._test_round(
                test_round=test_round,
                round_payloads=get_participant_to_engage_twitter(
                    self.participants, event="invalid_auth"
                ),
                synchronized_data_update_fn=lambda sd, _: sd.update(
                    synchronized_data_class=SynchronizedData,
                    **{
                        get_name(SynchronizedData.mech_for_twitter): False,
                        get_name(SynchronizedData.final_tx_hash): None,
                        get_name(SynchronizedData.failed_mech): False,
                    },
                ),
                synchronized_data_attr_checks=[],
                most_voted_payload="invalid_auth",
                exit_event=Event.INVALID_AUTH,
            )
        )


class TestMechRoundBase(MemeooorrRoundTestBase):
    """Tests for MechRoundBase using framework pattern."""

    def test_round_done(self) -> None:
        """Test round reaches DONE."""
        test_round = PostMechResponseRound(
            synchronized_data=self.synchronized_data,
            context=self.context_mock,
        )
        self._complete_run(
            self._test_round(
                test_round=test_round,
                round_payloads=get_participant_to_mech(
                    self.participants, mech_for_twitter=True, failed_mech=False
                ),
                synchronized_data_update_fn=lambda sd, _: sd.update(
                    synchronized_data_class=SynchronizedData,
                    **{
                        get_name(SynchronizedData.mech_for_twitter): True,
                        get_name(SynchronizedData.failed_mech): False,
                    },
                ),
                synchronized_data_attr_checks=[
                    lambda sd: sd.db.get("mech_for_twitter"),
                    lambda sd: sd.db.get("failed_mech"),
                ],
                most_voted_payload=True,
                exit_event=Event.DONE,
            )
        )


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


class TestActionDecisionRound(MemeooorrRoundTestBase):
    """Tests for ActionDecisionRound using framework pattern."""

    def test_round_done_with_new_persona(self) -> None:
        """Test round reaches DONE with new persona update."""
        test_round = ActionDecisionRound(
            synchronized_data=self.synchronized_data,
            context=self.context_mock,
        )
        self._complete_run(
            self._test_round(
                test_round=test_round,
                round_payloads=get_participant_to_action_decision(
                    self.participants,
                    event="done",
                    new_persona="new_persona",
                ),
                synchronized_data_update_fn=lambda sd, _: sd.update(
                    synchronized_data_class=SynchronizedData,
                    **{
                        get_name(SynchronizedData.token_action): json.dumps(
                            {
                                "action": "summon",
                                "token_address": "0xaddr",
                                "token_nonce": 1,
                                "token_name": "TestToken",
                                "token_ticker": "TT",
                                "token_supply": 1000000,
                                "amount": 0.5,
                                "tweet": "tweet",
                                "timestamp": 1234567890.0,
                            },
                            sort_keys=True,
                        ),
                    },
                ).update(
                    synchronized_data_class=SynchronizedData,
                    **{
                        get_name(SynchronizedData.persona): "new_persona",
                    },
                ),
                synchronized_data_attr_checks=[
                    lambda sd: sd.db.get("token_action"),
                    lambda sd: sd.db.get("persona"),
                ],
                most_voted_payload="done",
                exit_event=Event.DONE,
            )
        )

    def test_round_done_without_new_persona(self) -> None:
        """Test round reaches DONE without persona update."""
        test_round = ActionDecisionRound(
            synchronized_data=self.synchronized_data,
            context=self.context_mock,
        )
        self._complete_run(
            self._test_round(
                test_round=test_round,
                round_payloads=get_participant_to_action_decision(
                    self.participants,
                    event="done",
                    new_persona=None,
                ),
                synchronized_data_update_fn=lambda sd, _: sd.update(
                    synchronized_data_class=SynchronizedData,
                    **{
                        get_name(SynchronizedData.token_action): json.dumps(
                            {
                                "action": "summon",
                                "token_address": "0xaddr",
                                "token_nonce": 1,
                                "token_name": "TestToken",
                                "token_ticker": "TT",
                                "token_supply": 1000000,
                                "amount": 0.5,
                                "tweet": "tweet",
                                "timestamp": 1234567890.0,
                            },
                            sort_keys=True,
                        ),
                    },
                ),
                synchronized_data_attr_checks=[
                    lambda sd: sd.db.get("token_action"),
                ],
                most_voted_payload="done",
                exit_event=Event.DONE,
            )
        )

    def test_round_wait(self) -> None:
        """Test round reaches WAIT."""
        test_round = ActionDecisionRound(
            synchronized_data=self.synchronized_data,
            context=self.context_mock,
        )
        self._complete_run(
            self._test_round(
                test_round=test_round,
                round_payloads=get_participant_to_action_decision(
                    self.participants,
                    event="wait",
                    action=None,
                    token_address=None,
                    token_nonce=None,
                    token_name=None,
                    token_ticker=None,
                    token_supply=None,
                    amount=None,
                    tweet=None,
                    new_persona=None,
                    timestamp=0,
                ),
                synchronized_data_update_fn=lambda sd, _: sd,
                synchronized_data_attr_checks=[],
                most_voted_payload="wait",
                exit_event=Event.WAIT,
            )
        )


class TestActionPreparationRound(MemeooorrRoundTestBase):
    """Tests for ActionPreparationRound using framework pattern."""

    def test_round_error(self) -> None:
        """Test round reaches ERROR when tx_hash is None."""
        test_round = ActionPreparationRound(
            synchronized_data=self.synchronized_data,
            context=self.context_mock,
        )
        self._complete_run(
            self._test_round(
                test_round=test_round,
                round_payloads=get_participant_to_action_preparation(
                    self.participants, tx_hash=None
                ),
                synchronized_data_update_fn=lambda sd, _: sd,
                synchronized_data_attr_checks=[],
                most_voted_payload=None,
                exit_event=Event.ERROR,
            )
        )

    def test_round_done(self) -> None:
        """Test round reaches DONE when tx_hash is empty string."""
        test_round = ActionPreparationRound(
            synchronized_data=self.synchronized_data,
            context=self.context_mock,
        )
        self._complete_run(
            self._test_round(
                test_round=test_round,
                round_payloads=get_participant_to_action_preparation(
                    self.participants, tx_hash=""
                ),
                synchronized_data_update_fn=lambda sd, _: sd.update(
                    synchronized_data_class=SynchronizedData,
                    **{
                        get_name(SynchronizedData.most_voted_tx_hash): "",
                        get_name(SynchronizedData.tx_submitter): "submitter",
                    },
                ),
                synchronized_data_attr_checks=[
                    lambda sd: sd.db.get("most_voted_tx_hash"),
                    lambda sd: sd.db.get("tx_submitter"),
                ],
                most_voted_payload="",
                exit_event=Event.DONE,
            )
        )

    def test_round_settle(self) -> None:
        """Test round reaches SETTLE when tx_hash is non-empty."""
        test_round = ActionPreparationRound(
            synchronized_data=self.synchronized_data,
            context=self.context_mock,
        )
        self._complete_run(
            self._test_round(
                test_round=test_round,
                round_payloads=get_participant_to_action_preparation(
                    self.participants, tx_hash="0xhash123"
                ),
                synchronized_data_update_fn=lambda sd, _: sd.update(
                    synchronized_data_class=SynchronizedData,
                    **{
                        get_name(SynchronizedData.most_voted_tx_hash): "0xhash123",
                        get_name(SynchronizedData.tx_submitter): "submitter",
                    },
                ),
                synchronized_data_attr_checks=[
                    lambda sd: sd.db.get("most_voted_tx_hash"),
                    lambda sd: sd.db.get("tx_submitter"),
                ],
                most_voted_payload="0xhash123",
                exit_event=Event.SETTLE,
            )
        )


class TestActionTweetRound:
    """Tests for ActionTweetRound."""

    def test_is_subclass(self) -> None:
        """Test it is a subclass of EventRoundBase."""
        assert issubclass(ActionTweetRound, EventRoundBase)

    def test_payload_class(self) -> None:
        """Test payload class."""
        assert ActionTweetRound.payload_class is ActionTweetPayload


class TestCheckFundsRound(MemeooorrRoundTestBase):
    """Tests for CheckFundsRound.

    Note: CheckFundsRound.end_block() doesn't handle NO_MAJORITY explicitly
    (returns None when threshold not reached), so we can't use the standard
    framework _test_round which expects NO_MAJORITY handling. Tests use
    direct payload processing instead.
    """

    def _run_check_funds_round(self, event: str, check_funds_count: int) -> Any:
        """Run a CheckFundsRound to completion and return end_block result."""
        test_round = CheckFundsRound(
            synchronized_data=self.synchronized_data,
            context=self.context_mock,
        )
        payloads = get_participant_to_check_funds(
            self.participants, event=event, check_funds_count=check_funds_count
        )
        for payload in payloads.values():
            test_round.process_payload(payload)
        return test_round.end_block()

    def test_round_done(self) -> None:
        """Test round reaches DONE."""
        result = self._run_check_funds_round("done", 0)
        assert result is not None
        assert result[1] == Event.DONE

    def test_round_no_funds_below_max(self) -> None:
        """Test round reaches NO_FUNDS when count below max."""
        result = self._run_check_funds_round("no_funds", 5)
        assert result is not None
        _, event = result
        assert event == Event.NO_FUNDS

    def test_round_skip_at_max(self) -> None:
        """Test round reaches SKIP when count >= max."""
        result = self._run_check_funds_round("no_funds", MAX_CHECK_FUNDS_COUNT)
        assert result is not None
        assert result[1] == Event.SKIP

    def test_round_skip_above_max(self) -> None:
        """Test round reaches SKIP when count > max."""
        result = self._run_check_funds_round("no_funds", MAX_CHECK_FUNDS_COUNT + 5)
        assert result is not None
        assert result[1] == Event.SKIP

    def test_round_none_threshold_not_reached(self) -> None:
        """Test round returns None when threshold not reached."""
        test_round = CheckFundsRound(
            synchronized_data=self.synchronized_data,
            context=self.context_mock,
        )
        assert test_round.end_block() is None


class TestPostTxDecisionMakingRound:
    """Tests for PostTxDecisionMakingRound."""

    def test_is_subclass(self) -> None:
        """Test it is a subclass of EventRoundBase."""
        assert issubclass(PostTxDecisionMakingRound, EventRoundBase)

    def test_payload_class(self) -> None:
        """Test payload class."""
        assert PostTxDecisionMakingRound.payload_class is PostTxDecisionMakingPayload


class TestCallCheckpointRound(MemeooorrRoundTestBase):
    """Tests for CallCheckpointRound using framework pattern."""

    def test_round_done(self) -> None:
        """Test round reaches DONE when tx_hash is None."""
        test_round = CallCheckpointRound(
            synchronized_data=self.synchronized_data,
            context=self.context_mock,
        )
        self._complete_run(
            self._test_round(
                test_round=test_round,
                round_payloads=get_participant_to_call_checkpoint(
                    self.participants, tx_hash=None
                ),
                synchronized_data_update_fn=lambda sd, _: sd,
                synchronized_data_attr_checks=[],
                most_voted_payload="submitter",
                exit_event=Event.DONE,
            )
        )

    def test_round_settle(self) -> None:
        """Test round reaches SETTLE when tx_hash is present."""
        test_round = CallCheckpointRound(
            synchronized_data=self.synchronized_data,
            context=self.context_mock,
        )
        self._complete_run(
            self._test_round(
                test_round=test_round,
                round_payloads=get_participant_to_call_checkpoint(
                    self.participants, tx_hash="0xhash"
                ),
                synchronized_data_update_fn=lambda sd, _: sd.update(
                    synchronized_data_class=SynchronizedData,
                    **{
                        get_name(SynchronizedData.most_voted_tx_hash): "0xhash",
                        get_name(SynchronizedData.tx_submitter): "submitter",
                    },
                ),
                synchronized_data_attr_checks=[
                    lambda sd: sd.db.get("most_voted_tx_hash"),
                    lambda sd: sd.db.get("tx_submitter"),
                ],
                most_voted_payload="submitter",
                exit_event=Event.SETTLE,
            )
        )


class TestTransactionLoopCheckRound(MemeooorrRoundTestBase):
    """Tests for TransactionLoopCheckRound using framework pattern."""

    def test_round_done(self) -> None:
        """Test round reaches DONE when counter >= max."""
        self.context_mock.params.tx_loop_breaker_count = 5
        test_round = TransactionLoopCheckRound(
            synchronized_data=self.synchronized_data,
            context=self.context_mock,
        )
        self._complete_run(
            self._test_round(
                test_round=test_round,
                round_payloads=get_participant_to_tx_loop_check(
                    self.participants, counter=5
                ),
                synchronized_data_update_fn=lambda sd, _: sd,
                synchronized_data_attr_checks=[],
                most_voted_payload=5,
                exit_event=Event.DONE,
            )
        )

    def test_round_retry(self) -> None:
        """Test round reaches RETRY when counter < max."""
        self.context_mock.params.tx_loop_breaker_count = 5
        test_round = TransactionLoopCheckRound(
            synchronized_data=self.synchronized_data,
            context=self.context_mock,
        )
        self._complete_run(
            self._test_round(
                test_round=test_round,
                round_payloads=get_participant_to_tx_loop_check(
                    self.participants, counter=3
                ),
                synchronized_data_update_fn=lambda sd, _: sd.update(
                    synchronized_data_class=SynchronizedData,
                    **{
                        get_name(SynchronizedData.tx_loop_count): 3,
                    },
                ),
                synchronized_data_attr_checks=[
                    lambda sd: sd.db.get("tx_loop_count"),
                ],
                most_voted_payload=3,
                exit_event=Event.RETRY,
            )
        )


class TestEventRoundBase(MemeooorrRoundTestBase):
    """Tests for EventRoundBase (tested via ActionTweetRound)."""

    def test_round_done(self) -> None:
        """Test round reaches DONE via threshold."""
        test_round = ActionTweetRound(
            synchronized_data=self.synchronized_data,
            context=self.context_mock,
        )
        self._complete_run(
            self._test_round(
                test_round=test_round,
                round_payloads=get_participant_to_action_tweet(
                    self.participants, event="done"
                ),
                synchronized_data_update_fn=lambda sd, _: sd,
                synchronized_data_attr_checks=[],
                most_voted_payload="done",
                exit_event=Event.DONE,
            )
        )

    def test_round_error(self) -> None:
        """Test round reaches ERROR."""
        test_round = ActionTweetRound(
            synchronized_data=self.synchronized_data,
            context=self.context_mock,
        )
        self._complete_run(
            self._test_round(
                test_round=test_round,
                round_payloads=get_participant_to_action_tweet(
                    self.participants, event="ERROR"
                ),
                synchronized_data_update_fn=lambda sd, _: sd,
                synchronized_data_attr_checks=[],
                most_voted_payload="ERROR",
                exit_event=Event.ERROR,
            )
        )


# --- EventRoundBase / PostTxDecisionMaking round tests ---


class TestPostTxDecisionMakingRoundEndBlock(MemeooorrRoundTestBase):
    """Tests for PostTxDecisionMakingRound end_block."""

    def test_round_done(self) -> None:
        """Test round reaches DONE."""
        test_round = PostTxDecisionMakingRound(
            synchronized_data=self.synchronized_data,
            context=self.context_mock,
        )
        self._complete_run(
            self._test_round(
                test_round=test_round,
                round_payloads=get_participant_to_post_tx_decision(
                    self.participants, event="done"
                ),
                synchronized_data_update_fn=lambda sd, _: sd,
                synchronized_data_attr_checks=[],
                most_voted_payload="done",
                exit_event=Event.DONE,
            )
        )

    def test_round_action(self) -> None:
        """Test round reaches ACTION."""
        test_round = PostTxDecisionMakingRound(
            synchronized_data=self.synchronized_data,
            context=self.context_mock,
        )
        self._complete_run(
            self._test_round(
                test_round=test_round,
                round_payloads=get_participant_to_post_tx_decision(
                    self.participants, event="action"
                ),
                synchronized_data_update_fn=lambda sd, _: sd,
                synchronized_data_attr_checks=[],
                most_voted_payload="action",
                exit_event=Event.ACTION,
            )
        )

    def test_round_mech(self) -> None:
        """Test round reaches MECH."""
        test_round = PostTxDecisionMakingRound(
            synchronized_data=self.synchronized_data,
            context=self.context_mock,
        )
        self._complete_run(
            self._test_round(
                test_round=test_round,
                round_payloads=get_participant_to_post_tx_decision(
                    self.participants, event="mech"
                ),
                synchronized_data_update_fn=lambda sd, _: sd,
                synchronized_data_attr_checks=[],
                most_voted_payload="mech",
                exit_event=Event.MECH,
            )
        )


# --- Degenerate round tests ---


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


# --- MemeooorrAbciApp tests (kept as-is) ---


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
