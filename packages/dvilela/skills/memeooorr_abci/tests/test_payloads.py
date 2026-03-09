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

"""Tests for payloads.py."""

from dataclasses import FrozenInstanceError

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
from packages.valory.skills.abstract_round_abci.base import BaseTxPayload


SENDER = "test_sender"


class TestLoadDatabasePayload:
    """Tests for LoadDatabasePayload."""

    def test_construction(self) -> None:
        """Test payload construction."""
        payload = LoadDatabasePayload(
            sender=SENDER,
            persona="test_persona",
            heart_cooldown_hours=24,
            summon_cooldown_seconds=3600,
            agent_details='{"key": "value"}',
        )
        assert payload.sender == SENDER
        assert payload.persona == "test_persona"
        assert payload.heart_cooldown_hours == 24
        assert payload.summon_cooldown_seconds == 3600
        assert payload.agent_details == '{"key": "value"}'

    def test_is_subclass(self) -> None:
        """Test it is a subclass of BaseTxPayload."""
        assert issubclass(LoadDatabasePayload, BaseTxPayload)

    def test_frozen(self) -> None:
        """Test that the dataclass is frozen."""
        payload = LoadDatabasePayload(
            sender=SENDER,
            persona="p",
            heart_cooldown_hours=1,
            summon_cooldown_seconds=1,
            agent_details="{}",
        )
        with pytest.raises(FrozenInstanceError):
            payload.persona = "new"  # type: ignore


class TestCheckStakingPayload:
    """Tests for CheckStakingPayload."""

    def test_construction_true(self) -> None:
        """Test payload construction with True."""
        payload = CheckStakingPayload(sender=SENDER, is_staking_kpi_met=True)
        assert payload.is_staking_kpi_met is True

    def test_construction_false(self) -> None:
        """Test payload construction with False."""
        payload = CheckStakingPayload(sender=SENDER, is_staking_kpi_met=False)
        assert payload.is_staking_kpi_met is False

    def test_construction_none(self) -> None:
        """Test payload construction with None."""
        payload = CheckStakingPayload(sender=SENDER, is_staking_kpi_met=None)
        assert payload.is_staking_kpi_met is None

    def test_frozen(self) -> None:
        """Test that the dataclass is frozen."""
        payload = CheckStakingPayload(sender=SENDER, is_staking_kpi_met=True)
        with pytest.raises(FrozenInstanceError):
            payload.is_staking_kpi_met = False  # type: ignore


class TestPullMemesPayload:
    """Tests for PullMemesPayload."""

    def test_construction_with_data(self) -> None:
        """Test payload construction with meme coins."""
        payload = PullMemesPayload(
            sender=SENDER, meme_coins='[{"name": "test"}]', event="done"
        )
        assert payload.meme_coins == '[{"name": "test"}]'
        assert payload.event == "done"

    def test_construction_defaults(self) -> None:
        """Test payload construction with defaults."""
        payload = PullMemesPayload(sender=SENDER)
        assert payload.meme_coins is None
        assert payload.event is None

    def test_frozen(self) -> None:
        """Test that the dataclass is frozen."""
        payload = PullMemesPayload(sender=SENDER, meme_coins="[]")
        with pytest.raises(FrozenInstanceError):
            payload.meme_coins = "new"  # type: ignore


class TestCollectFeedbackPayload:
    """Tests for CollectFeedbackPayload."""

    def test_construction(self) -> None:
        """Test payload construction."""
        payload = CollectFeedbackPayload(sender=SENDER, feedback='["good"]')
        assert payload.feedback == '["good"]'

    def test_construction_none(self) -> None:
        """Test payload construction with None."""
        payload = CollectFeedbackPayload(sender=SENDER, feedback=None)
        assert payload.feedback is None

    def test_frozen(self) -> None:
        """Test that the dataclass is frozen."""
        payload = CollectFeedbackPayload(sender=SENDER, feedback="test")
        with pytest.raises(FrozenInstanceError):
            payload.feedback = "new"  # type: ignore


class TestEngageTwitterPayload:
    """Tests for EngageTwitterPayload."""

    def test_construction(self) -> None:
        """Test payload construction."""
        payload = EngageTwitterPayload(
            sender=SENDER,
            event="done",
            mech_request=None,
            tx_submitter="submitter",
            failed_mech=False,
        )
        assert payload.event == "done"
        assert payload.mech_request is None
        assert payload.tx_submitter == "submitter"
        assert payload.failed_mech is False

    def test_construction_with_mech_request(self) -> None:
        """Test payload construction with mech request."""
        payload = EngageTwitterPayload(
            sender=SENDER,
            event="mech",
            mech_request='[{"key": "val"}]',
            tx_submitter="sub",
        )
        assert payload.mech_request == '[{"key": "val"}]'

    def test_default_failed_mech(self) -> None:
        """Test default value of failed_mech."""
        payload = EngageTwitterPayload(
            sender=SENDER,
            event="done",
            mech_request=None,
            tx_submitter="sub",
        )
        assert payload.failed_mech is False

    def test_frozen(self) -> None:
        """Test that the dataclass is frozen."""
        payload = EngageTwitterPayload(
            sender=SENDER, event="done", mech_request=None, tx_submitter="sub"
        )
        with pytest.raises(FrozenInstanceError):
            payload.event = "new"  # type: ignore


class TestActionDecisionPayload:
    """Tests for ActionDecisionPayload."""

    def test_construction_full(self) -> None:
        """Test payload construction with all fields."""
        payload = ActionDecisionPayload(
            sender=SENDER,
            event="done",
            action="summon",
            token_address="0xabc",
            token_nonce=1,
            token_name="TestToken",
            token_ticker="TT",
            token_supply=1000000,
            amount=0.5,
            tweet="Launching!",
            new_persona="new persona",
            timestamp=1234567890.0,
        )
        assert payload.event == "done"
        assert payload.action == "summon"
        assert payload.token_address == "0xabc"
        assert payload.token_nonce == 1
        assert payload.token_name == "TestToken"
        assert payload.token_ticker == "TT"
        assert payload.token_supply == 1000000
        assert payload.amount == 0.5
        assert payload.tweet == "Launching!"
        assert payload.new_persona == "new persona"
        assert payload.timestamp == 1234567890.0

    def test_construction_none_fields(self) -> None:
        """Test payload with None optional fields."""
        payload = ActionDecisionPayload(
            sender=SENDER,
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
        )
        assert payload.action is None
        assert payload.token_address is None
        assert payload.timestamp == 0

    def test_default_timestamp(self) -> None:
        """Test default timestamp value."""
        payload = ActionDecisionPayload(
            sender=SENDER,
            event="done",
            action=None,
            token_address=None,
            token_nonce=None,
            token_name=None,
            token_ticker=None,
            token_supply=None,
            amount=None,
            tweet=None,
            new_persona=None,
        )
        assert payload.timestamp == 0

    def test_frozen(self) -> None:
        """Test that the dataclass is frozen."""
        payload = ActionDecisionPayload(
            sender=SENDER,
            event="done",
            action=None,
            token_address=None,
            token_nonce=None,
            token_name=None,
            token_ticker=None,
            token_supply=None,
            amount=None,
            tweet=None,
            new_persona=None,
        )
        with pytest.raises(FrozenInstanceError):
            payload.event = "new"  # type: ignore


class TestActionPreparationPayload:
    """Tests for ActionPreparationPayload."""

    def test_construction(self) -> None:
        """Test payload construction."""
        payload = ActionPreparationPayload(
            sender=SENDER, tx_hash="0xhash", tx_submitter="submitter"
        )
        assert payload.tx_hash == "0xhash"
        assert payload.tx_submitter == "submitter"

    def test_construction_none_hash(self) -> None:
        """Test payload construction with None hash."""
        payload = ActionPreparationPayload(
            sender=SENDER, tx_hash=None, tx_submitter="submitter"
        )
        assert payload.tx_hash is None

    def test_frozen(self) -> None:
        """Test that the dataclass is frozen."""
        payload = ActionPreparationPayload(
            sender=SENDER, tx_hash="0x", tx_submitter="sub"
        )
        with pytest.raises(FrozenInstanceError):
            payload.tx_hash = "new"  # type: ignore


class TestActionTweetPayload:
    """Tests for ActionTweetPayload."""

    def test_construction(self) -> None:
        """Test payload construction."""
        payload = ActionTweetPayload(sender=SENDER, event="done")
        assert payload.event == "done"

    def test_frozen(self) -> None:
        """Test that the dataclass is frozen."""
        payload = ActionTweetPayload(sender=SENDER, event="done")
        with pytest.raises(FrozenInstanceError):
            payload.event = "new"  # type: ignore


class TestCheckFundsPayload:
    """Tests for CheckFundsPayload."""

    def test_construction(self) -> None:
        """Test payload construction."""
        payload = CheckFundsPayload(
            sender=SENDER, event="done", check_funds_count=5
        )
        assert payload.event == "done"
        assert payload.check_funds_count == 5

    def test_frozen(self) -> None:
        """Test that the dataclass is frozen."""
        payload = CheckFundsPayload(sender=SENDER, event="done", check_funds_count=0)
        with pytest.raises(FrozenInstanceError):
            payload.event = "new"  # type: ignore


class TestPostTxDecisionMakingPayload:
    """Tests for PostTxDecisionMakingPayload."""

    def test_construction(self) -> None:
        """Test payload construction."""
        payload = PostTxDecisionMakingPayload(sender=SENDER, event="done")
        assert payload.event == "done"

    def test_frozen(self) -> None:
        """Test that the dataclass is frozen."""
        payload = PostTxDecisionMakingPayload(sender=SENDER, event="done")
        with pytest.raises(FrozenInstanceError):
            payload.event = "new"  # type: ignore


class TestCallCheckpointPayload:
    """Tests for CallCheckpointPayload."""

    def test_construction(self) -> None:
        """Test payload construction."""
        payload = CallCheckpointPayload(
            sender=SENDER, tx_submitter="submitter", tx_hash="0xhash"
        )
        assert payload.tx_submitter == "submitter"
        assert payload.tx_hash == "0xhash"

    def test_construction_none_hash(self) -> None:
        """Test payload with None hash."""
        payload = CallCheckpointPayload(
            sender=SENDER, tx_submitter="submitter", tx_hash=None
        )
        assert payload.tx_hash is None

    def test_frozen(self) -> None:
        """Test that the dataclass is frozen."""
        payload = CallCheckpointPayload(
            sender=SENDER, tx_submitter="sub", tx_hash="0x"
        )
        with pytest.raises(FrozenInstanceError):
            payload.tx_hash = "new"  # type: ignore


class TestMechPayload:
    """Tests for MechPayload."""

    def test_construction(self) -> None:
        """Test payload construction."""
        payload = MechPayload(sender=SENDER, mech_for_twitter=True, failed_mech=False)
        assert payload.mech_for_twitter is True
        assert payload.failed_mech is False

    def test_default_failed_mech(self) -> None:
        """Test default failed_mech value."""
        payload = MechPayload(sender=SENDER, mech_for_twitter=True)
        assert payload.failed_mech is False

    def test_frozen(self) -> None:
        """Test that the dataclass is frozen."""
        payload = MechPayload(sender=SENDER, mech_for_twitter=True)
        with pytest.raises(FrozenInstanceError):
            payload.mech_for_twitter = False  # type: ignore


class TestTransactionLoopCheckPayload:
    """Tests for TransactionLoopCheckPayload."""

    def test_construction(self) -> None:
        """Test payload construction."""
        payload = TransactionLoopCheckPayload(sender=SENDER, counter=5)
        assert payload.counter == 5

    def test_frozen(self) -> None:
        """Test that the dataclass is frozen."""
        payload = TransactionLoopCheckPayload(sender=SENDER, counter=0)
        with pytest.raises(FrozenInstanceError):
            payload.counter = 1  # type: ignore
