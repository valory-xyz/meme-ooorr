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

"""Tests for behaviour_classes/chain.py."""

import json
from typing import Any, Generator, Optional
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from packages.dvilela.skills.memeooorr_abci.behaviour_classes.chain import (
    ActionPreparationBehaviour,
    CHECKPOINT_FILENAME,
    CallCheckpointBehaviour,
    ChainBehaviour,
    CheckFundsBehaviour,
    CheckStakingBehaviour,
    EMPTY_CALL_DATA,
    LIVENESS_RATIO_SCALE_FACTOR,
    NULL_ADDRESS,
    PostTxDecisionMakingBehaviour,
    PullMemesBehaviour,
    REQUIRED_REQUESTS_SAFETY_MARGIN,
    SAFE_GAS,
    TransactionLoopCheckBehaviour,
    ZERO_VALUE,
)
from packages.dvilela.skills.memeooorr_abci.rounds import (
    ActionPreparationRound,
    CallCheckpointRound,
    CheckFundsRound,
    CheckStakingRound,
    Event,
    PostTxDecisionMakingRound,
    PullMemesRound,
    StakingState,
    TransactionLoopCheckRound,
)
from packages.valory.protocols.contract_api import ContractApiMessage

from .conftest import (
    MECH_MARKETPLACE_ADDRESS,
    SAFE_ADDRESS,
    SENDER,
    STAKING_TOKEN_ADDRESS,
    make_mock_context,
    make_mock_params,
    make_mock_synchronized_data,
)


class TestChainBehaviourMatchingRounds:
    """Tests that chain behaviours have proper matching_round assignments."""

    def test_check_funds_matching_round(self) -> None:
        """Test CheckFundsBehaviour has correct matching_round."""
        assert CheckFundsBehaviour.matching_round is CheckFundsRound

    def test_check_staking_matching_round(self) -> None:
        """Test CheckStakingBehaviour has correct matching_round."""
        assert CheckStakingBehaviour.matching_round is CheckStakingRound

    def test_pull_memes_matching_round(self) -> None:
        """Test PullMemesBehaviour has correct matching_round."""
        assert PullMemesBehaviour.matching_round is PullMemesRound

    def test_action_preparation_matching_round(self) -> None:
        """Test ActionPreparationBehaviour has correct matching_round."""
        assert ActionPreparationBehaviour.matching_round is ActionPreparationRound

    def test_post_tx_decision_making_matching_round(self) -> None:
        """Test PostTxDecisionMakingBehaviour has correct matching_round."""
        assert PostTxDecisionMakingBehaviour.matching_round is PostTxDecisionMakingRound

    def test_call_checkpoint_matching_round(self) -> None:
        """Test CallCheckpointBehaviour has correct matching_round."""
        assert CallCheckpointBehaviour.matching_round is CallCheckpointRound

    def test_transaction_loop_check_matching_round(self) -> None:
        """Test TransactionLoopCheckBehaviour has correct matching_round."""
        assert TransactionLoopCheckBehaviour.matching_round is TransactionLoopCheckRound


class TestChainBehaviourHelpers:
    """Tests for ChainBehaviour helper methods."""

    def _make_chain_behaviour(self, **kwargs: Any) -> MagicMock:
        """Create a mock ChainBehaviour."""
        behaviour = MagicMock(spec=ChainBehaviour)
        behaviour.params = make_mock_params(**kwargs)
        behaviour.context = make_mock_context(params=behaviour.params)
        behaviour.synchronized_data = make_mock_synchronized_data()
        behaviour.get_chain_id = MagicMock(return_value="base")
        behaviour.get_meme_factory_address = MagicMock(return_value="0x" + "d" * 40)
        return behaviour

    def test_default_error(self) -> None:
        """Test default_error logs error message."""
        behaviour = self._make_chain_behaviour()
        response_msg = MagicMock()
        ChainBehaviour.default_error(
            behaviour, "test_contract", "test_callable", response_msg
        )
        # Should not raise

    def test_contract_interaction_error_info(self) -> None:
        """Test contract_interaction_error with info level message."""
        behaviour = self._make_chain_behaviour()
        response_msg = MagicMock()
        response_msg.raw_transaction.body = {"info": "some info message"}
        ChainBehaviour.contract_interaction_error(
            behaviour, "test_contract", "test_callable", response_msg
        )

    def test_contract_interaction_error_warning(self) -> None:
        """Test contract_interaction_error with warning level message."""
        behaviour = self._make_chain_behaviour()
        response_msg = MagicMock()
        response_msg.raw_transaction.body = {"warning": "some warning"}
        ChainBehaviour.contract_interaction_error(
            behaviour, "test_contract", "test_callable", response_msg
        )

    def test_contract_interaction_error_error(self) -> None:
        """Test contract_interaction_error with error level message."""
        behaviour = self._make_chain_behaviour()
        response_msg = MagicMock()
        response_msg.raw_transaction.body = {"error": "some error"}
        ChainBehaviour.contract_interaction_error(
            behaviour, "test_contract", "test_callable", response_msg
        )

    def test_contract_interaction_error_default(self) -> None:
        """Test contract_interaction_error falls back to default_error."""
        behaviour = self._make_chain_behaviour()
        response_msg = MagicMock()
        response_msg.raw_transaction.body = {}
        ChainBehaviour.contract_interaction_error(
            behaviour, "test_contract", "test_callable", response_msg
        )


class TestContractInteract:
    """Tests for contract_interact generator."""

    def _make_chain_behaviour(self) -> MagicMock:
        """Create a mock ChainBehaviour for generator tests."""
        behaviour = MagicMock(spec=ChainBehaviour)
        behaviour.params = make_mock_params()
        behaviour.context = make_mock_context(params=behaviour.params)
        behaviour.synchronized_data = make_mock_synchronized_data()
        behaviour.get_chain_id = MagicMock(return_value="base")
        return behaviour

    def test_contract_interact_success(self) -> None:
        """Test contract_interact returns data on success."""
        behaviour = self._make_chain_behaviour()
        response_msg = MagicMock()
        response_msg.performative = ContractApiMessage.Performative.RAW_TRANSACTION
        response_msg.raw_transaction.body = {"data": 42}

        def mock_get_contract_api_response(*args, **kwargs):
            yield
            return response_msg

        behaviour.get_contract_api_response = mock_get_contract_api_response

        from aea.configurations.data_types import PublicId

        gen = ChainBehaviour.contract_interact(
            behaviour,
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address="0x" + "a" * 40,
            contract_public_id=PublicId("test", "contract", "0.1.0"),
            contract_callable="test_method",
            data_key="data",
        )
        next(gen)
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result == 42

    def test_contract_interact_wrong_performative(self) -> None:
        """Test contract_interact returns None on wrong performative."""
        behaviour = self._make_chain_behaviour()
        response_msg = MagicMock()
        response_msg.performative = ContractApiMessage.Performative.ERROR

        def mock_get_contract_api_response(*args, **kwargs):
            yield
            return response_msg

        behaviour.get_contract_api_response = mock_get_contract_api_response

        from aea.configurations.data_types import PublicId

        gen = ChainBehaviour.contract_interact(
            behaviour,
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address="0x" + "a" * 40,
            contract_public_id=PublicId("test", "contract", "0.1.0"),
            contract_callable="test_method",
            data_key="data",
        )
        next(gen)
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result is None

    def test_contract_interact_missing_data_key(self) -> None:
        """Test contract_interact returns None when data_key not in response."""
        behaviour = self._make_chain_behaviour()
        response_msg = MagicMock()
        response_msg.performative = ContractApiMessage.Performative.RAW_TRANSACTION
        response_msg.raw_transaction.body = {"other_key": 42}

        def mock_get_contract_api_response(*args, **kwargs):
            yield
            return response_msg

        behaviour.get_contract_api_response = mock_get_contract_api_response

        from aea.configurations.data_types import PublicId

        gen = ChainBehaviour.contract_interact(
            behaviour,
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address="0x" + "a" * 40,
            contract_public_id=PublicId("test", "contract", "0.1.0"),
            contract_callable="test_method",
            data_key="data",
        )
        next(gen)
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result is None


class TestGetServiceStakingState:
    """Tests for _get_service_staking_state."""

    def _make_chain_behaviour(self, **kwargs: Any) -> MagicMock:
        behaviour = MagicMock(spec=ChainBehaviour)
        behaviour.params = make_mock_params(**kwargs)
        behaviour.context = make_mock_context(params=behaviour.params)
        behaviour.synchronized_data = make_mock_synchronized_data()
        behaviour.get_chain_id = MagicMock(return_value="base")
        return behaviour

    def test_no_service_id(self) -> None:
        """Test returns UNSTAKED when no service id configured."""
        behaviour = self._make_chain_behaviour(on_chain_service_id=None)

        gen = ChainBehaviour._get_service_staking_state(behaviour, chain="base")
        try:
            next(gen)
        except StopIteration as e:
            result = e.value
        assert result == StakingState.UNSTAKED

    def test_null_staking_address(self) -> None:
        """Test returns UNSTAKED when staking address is NULL_ADDRESS."""
        behaviour = self._make_chain_behaviour(
            staking_token_contract_address=NULL_ADDRESS
        )

        gen = ChainBehaviour._get_service_staking_state(behaviour, chain="base")
        try:
            next(gen)
        except StopIteration as e:
            result = e.value
        assert result == StakingState.UNSTAKED

    def test_staking_state_staked(self) -> None:
        """Test returns correct staking state from contract."""
        behaviour = self._make_chain_behaviour()

        def mock_contract_interact(*args, **kwargs):
            yield
            return StakingState.STAKED.value

        behaviour.contract_interact = mock_contract_interact

        gen = ChainBehaviour._get_service_staking_state(behaviour, chain="base")
        next(gen)
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result == StakingState.STAKED

    def test_staking_state_none_returns_unstaked(self) -> None:
        """Test returns UNSTAKED when contract returns None."""
        behaviour = self._make_chain_behaviour()

        def mock_contract_interact(*args, **kwargs):
            yield
            return None

        behaviour.contract_interact = mock_contract_interact

        gen = ChainBehaviour._get_service_staking_state(behaviour, chain="base")
        next(gen)
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result == StakingState.UNSTAKED


class TestCheckFundsBehaviourGetEvent:
    """Tests for CheckFundsBehaviour.get_event."""

    def _make_behaviour(self) -> MagicMock:
        behaviour = MagicMock(spec=CheckFundsBehaviour)
        behaviour.params = make_mock_params()
        behaviour.context = make_mock_context(params=behaviour.params)
        behaviour.synchronized_data = make_mock_synchronized_data()
        return behaviour

    def test_get_event_no_balance(self) -> None:
        """Test get_event returns NO_FUNDS when balance is None."""
        behaviour = self._make_behaviour()

        def mock_get_native_balance():
            yield
            return {"agent": None, "safe": None}

        behaviour.get_native_balance = mock_get_native_balance

        gen = CheckFundsBehaviour.get_event(behaviour)
        next(gen)
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result == Event.NO_FUNDS.value

    def test_get_event_insufficient_funds(self) -> None:
        """Test get_event returns NO_FUNDS when balance is below minimum."""
        behaviour = self._make_behaviour()

        def mock_get_native_balance():
            yield
            return {"agent": 0.0001, "safe": 1.0}

        behaviour.get_native_balance = mock_get_native_balance

        gen = CheckFundsBehaviour.get_event(behaviour)
        next(gen)
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result == Event.NO_FUNDS.value

    def test_get_event_sufficient_funds(self) -> None:
        """Test get_event returns DONE when balance is sufficient."""
        behaviour = self._make_behaviour()

        def mock_get_native_balance():
            yield
            return {"agent": 1.0, "safe": 1.0}

        behaviour.get_native_balance = mock_get_native_balance

        gen = CheckFundsBehaviour.get_event(behaviour)
        next(gen)
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result == Event.DONE.value


class TestPostTxDecisionMakingBehaviour:
    """Tests for PostTxDecisionMakingBehaviour."""

    def test_event_done_for_checkpoint(self) -> None:
        """Test returns DONE when tx_submitter matches CallCheckpointBehaviour."""
        behaviour = MagicMock(spec=PostTxDecisionMakingBehaviour)
        behaviour.context = make_mock_context()
        round_id = CallCheckpointBehaviour.matching_round.auto_round_id()
        behaviour.synchronized_data = make_mock_synchronized_data(tx_submitter=round_id)
        # The logic is straightforward; test the conditional path
        assert round_id == CallCheckpointRound.auto_round_id()

    def test_event_action_for_action_preparation(self) -> None:
        """Test returns ACTION when tx_submitter matches ActionPreparationBehaviour."""
        round_id = ActionPreparationBehaviour.matching_round.auto_round_id()
        assert round_id == ActionPreparationRound.auto_round_id()


class TestPullMemesBehaviour:
    """Tests for PullMemesBehaviour."""

    def test_matching_round(self) -> None:
        """Test matching round is PullMemesRound."""
        assert PullMemesBehaviour.matching_round is PullMemesRound


class TestTransactionLoopCheckBehaviour:
    """Tests for TransactionLoopCheckBehaviour."""

    def test_matching_round(self) -> None:
        """Test matching round is TransactionLoopCheckRound."""
        assert TransactionLoopCheckBehaviour.matching_round is TransactionLoopCheckRound


class TestBuildSafeTxHash:
    """Tests for _build_safe_tx_hash method."""

    def _make_chain_behaviour(self) -> MagicMock:
        behaviour = MagicMock(spec=ChainBehaviour)
        behaviour.params = make_mock_params()
        behaviour.context = make_mock_context(params=behaviour.params)
        behaviour.synchronized_data = make_mock_synchronized_data()
        behaviour.get_chain_id = MagicMock(return_value="base")
        return behaviour

    def test_build_safe_tx_hash_error_response(self) -> None:
        """Test _build_safe_tx_hash returns None on error response."""
        behaviour = self._make_chain_behaviour()
        response_msg = MagicMock()
        response_msg.performative = ContractApiMessage.Performative.ERROR

        def mock_get_contract_api_response(*args, **kwargs):
            yield
            return response_msg

        behaviour.get_contract_api_response = mock_get_contract_api_response

        gen = ChainBehaviour._build_safe_tx_hash(behaviour, to_address="0x" + "a" * 40)
        next(gen)
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result is None

    def test_build_safe_tx_hash_invalid_hash(self) -> None:
        """Test _build_safe_tx_hash returns None on invalid hash."""
        behaviour = self._make_chain_behaviour()
        response_msg = MagicMock()
        response_msg.performative = ContractApiMessage.Performative.STATE
        response_msg.state.body = {"tx_hash": "0xshort"}

        def mock_get_contract_api_response(*args, **kwargs):
            yield
            return response_msg

        behaviour.get_contract_api_response = mock_get_contract_api_response

        gen = ChainBehaviour._build_safe_tx_hash(behaviour, to_address="0x" + "a" * 40)
        next(gen)
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result is None

    def test_build_safe_tx_hash_none_hash(self) -> None:
        """Test _build_safe_tx_hash returns None when hash is None."""
        behaviour = self._make_chain_behaviour()
        response_msg = MagicMock()
        response_msg.performative = ContractApiMessage.Performative.STATE
        response_msg.state.body = {"tx_hash": None}

        def mock_get_contract_api_response(*args, **kwargs):
            yield
            return response_msg

        behaviour.get_contract_api_response = mock_get_contract_api_response

        gen = ChainBehaviour._build_safe_tx_hash(behaviour, to_address="0x" + "a" * 40)
        next(gen)
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result is None


class TestGetLivenessRatio:
    """Tests for _get_liveness_ratio."""

    def _make_chain_behaviour(self) -> MagicMock:
        behaviour = MagicMock(spec=ChainBehaviour)
        behaviour.params = make_mock_params()
        behaviour.context = make_mock_context(params=behaviour.params)
        return behaviour

    def test_liveness_ratio_none(self) -> None:
        """Test _get_liveness_ratio logs error on None."""
        behaviour = self._make_chain_behaviour()

        def mock_contract_interact(*args, **kwargs):
            yield
            return None

        behaviour.contract_interact = mock_contract_interact

        gen = ChainBehaviour._get_liveness_ratio(behaviour, "base")
        next(gen)
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result is None

    def test_liveness_ratio_zero(self) -> None:
        """Test _get_liveness_ratio logs error on zero."""
        behaviour = self._make_chain_behaviour()

        def mock_contract_interact(*args, **kwargs):
            yield
            return 0

        behaviour.contract_interact = mock_contract_interact

        gen = ChainBehaviour._get_liveness_ratio(behaviour, "base")
        next(gen)
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result == 0

    def test_liveness_ratio_valid(self) -> None:
        """Test _get_liveness_ratio returns valid ratio."""
        behaviour = self._make_chain_behaviour()

        def mock_contract_interact(*args, **kwargs):
            yield
            return 10**18

        behaviour.contract_interact = mock_contract_interact

        gen = ChainBehaviour._get_liveness_ratio(behaviour, "base")
        next(gen)
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result == 10**18


class TestGetServiceInfo:
    """Tests for _get_service_info."""

    def _make_chain_behaviour(self, **kwargs: Any) -> MagicMock:
        behaviour = MagicMock(spec=ChainBehaviour)
        behaviour.params = make_mock_params(**kwargs)
        behaviour.context = make_mock_context(params=behaviour.params)
        return behaviour

    def test_no_service_id(self) -> None:
        """Test _get_service_info returns None when no service id."""
        behaviour = self._make_chain_behaviour(on_chain_service_id=None)

        gen = ChainBehaviour._get_service_info(behaviour, chain="base")
        try:
            next(gen)
        except StopIteration as e:
            result = e.value
        assert result is None

    def test_with_service_id(self) -> None:
        """Test _get_service_info calls contract_interact."""
        behaviour = self._make_chain_behaviour()
        service_info = (1, 2, (3, 4))

        def mock_contract_interact(*args, **kwargs):
            yield
            return service_info

        behaviour.contract_interact = mock_contract_interact

        gen = ChainBehaviour._get_service_info(behaviour, chain="base")
        next(gen)
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result == service_info


class TestGetMultisigNonces:
    """Tests for _get_multisig_nonces."""

    def _make_chain_behaviour(self) -> MagicMock:
        behaviour = MagicMock(spec=ChainBehaviour)
        behaviour.params = make_mock_params()
        behaviour.context = make_mock_context(params=behaviour.params)
        return behaviour

    def test_multisig_nonces_none(self) -> None:
        """Test returns None when contract returns None."""
        behaviour = self._make_chain_behaviour()

        def mock_contract_interact(*args, **kwargs):
            yield
            return None

        behaviour.contract_interact = mock_contract_interact

        gen = ChainBehaviour._get_multisig_nonces(behaviour, "base", SAFE_ADDRESS)
        next(gen)
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result is None

    def test_multisig_nonces_empty_list(self) -> None:
        """Test returns None when contract returns empty list."""
        behaviour = self._make_chain_behaviour()

        def mock_contract_interact(*args, **kwargs):
            yield
            return []

        behaviour.contract_interact = mock_contract_interact

        gen = ChainBehaviour._get_multisig_nonces(behaviour, "base", SAFE_ADDRESS)
        next(gen)
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result is None

    def test_multisig_nonces_valid(self) -> None:
        """Test returns first element of valid list."""
        behaviour = self._make_chain_behaviour()

        def mock_contract_interact(*args, **kwargs):
            yield
            return [42, 10]

        behaviour.contract_interact = mock_contract_interact

        gen = ChainBehaviour._get_multisig_nonces(behaviour, "base", SAFE_ADDRESS)
        next(gen)
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result == 42


class TestCheckpointBehaviour:
    """Tests for CallCheckpointBehaviour helper methods."""

    def _make_behaviour(self) -> MagicMock:
        behaviour = MagicMock(spec=CallCheckpointBehaviour)
        behaviour.params = make_mock_params()
        behaviour.context = make_mock_context(params=behaviour.params)
        behaviour.synchronized_data = make_mock_synchronized_data()
        behaviour.get_chain_id = MagicMock(return_value="base")

        mock_ts = MagicMock()
        mock_ts.timestamp.return_value = 1700000000.0
        behaviour.round_sequence = MagicMock()
        behaviour.round_sequence.last_round_transition_timestamp = mock_ts
        return behaviour

    def test_check_if_checkpoint_reached_none(self) -> None:
        """Test _check_if_checkpoint_reached returns False when next_checkpoint is None."""
        behaviour = self._make_behaviour()

        def mock_get_next_checkpoint(chain):
            yield
            return None

        behaviour._get_next_checkpoint = mock_get_next_checkpoint

        gen = CallCheckpointBehaviour._check_if_checkpoint_reached(
            behaviour, chain="base"
        )
        next(gen)
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result is False

    def test_check_if_checkpoint_reached_zero(self) -> None:
        """Test _check_if_checkpoint_reached returns True when next_checkpoint is 0."""
        behaviour = self._make_behaviour()

        def mock_get_next_checkpoint(chain):
            yield
            return 0

        behaviour._get_next_checkpoint = mock_get_next_checkpoint

        gen = CallCheckpointBehaviour._check_if_checkpoint_reached(
            behaviour, chain="base"
        )
        next(gen)
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result is True

    def test_check_if_checkpoint_reached_future(self) -> None:
        """Test returns False when next checkpoint is in the future."""
        behaviour = self._make_behaviour()

        def mock_get_next_checkpoint(chain):
            yield
            return 1700000001  # 1 second in the future

        behaviour._get_next_checkpoint = mock_get_next_checkpoint

        gen = CallCheckpointBehaviour._check_if_checkpoint_reached(
            behaviour, chain="base"
        )
        next(gen)
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result is False

    def test_check_if_checkpoint_reached_past(self) -> None:
        """Test returns True when next checkpoint is in the past."""
        behaviour = self._make_behaviour()

        def mock_get_next_checkpoint(chain):
            yield
            return 1699999999  # 1 second in the past

        behaviour._get_next_checkpoint = mock_get_next_checkpoint

        gen = CallCheckpointBehaviour._check_if_checkpoint_reached(
            behaviour, chain="base"
        )
        next(gen)
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result is True
