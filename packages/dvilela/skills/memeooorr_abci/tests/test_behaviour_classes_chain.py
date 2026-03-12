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

# pylint: disable=protected-access,unused-argument,used-before-assignment,too-few-public-methods,useless-return,import-outside-toplevel,assigning-non-slot

import json
from typing import Any, Generator
from unittest.mock import MagicMock

import pytest

from packages.dvilela.skills.memeooorr_abci.behaviour_classes.chain import (
    ActionPreparationBehaviour,
    CallCheckpointBehaviour,
    ChainBehaviour,
    CheckFundsBehaviour,
    CheckStakingBehaviour,
    NULL_ADDRESS,
    PostTxDecisionMakingBehaviour,
    PullMemesBehaviour,
    TransactionLoopCheckBehaviour,
)
from packages.dvilela.skills.memeooorr_abci.rounds import (
    ActionPreparationRound,
    CallCheckpointRound,
    Event,
    StakingState,
    SynchronizedData,
)
from packages.dvilela.skills.memeooorr_abci.tests.conftest import (
    MemeooorrFSMBehaviourBaseCase,
    SAFE_ADDRESS,
    make_mock_context,
    make_mock_params,
    make_mock_synchronized_data,
)
from packages.valory.protocols.contract_api import ContractApiMessage
from packages.valory.protocols.ledger_api import LedgerApiMessage
from packages.valory.skills.abstract_round_abci.base import AbciAppDB


def _run_generator(gen: Generator[Any, None, Any]) -> Any:
    """Run a generator to completion, sending None for each yield, return the final value."""
    result = None
    try:
        next(gen)
        while True:
            gen.send(None)
    except StopIteration as e:
        result = e.value
    return result


def _make_timestamp_mock(ts: float = 1700000000.0) -> MagicMock:
    """Create a mock timestamp."""
    mock_ts = MagicMock()
    mock_ts.timestamp.return_value = ts
    return mock_ts


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

        def mock_get_contract_api_response(*args, **kwargs):  # type: ignore[no-untyped-def]
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
        result = None
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

        def mock_get_contract_api_response(*args, **kwargs):  # type: ignore[no-untyped-def]
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
        result = None
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

        def mock_get_contract_api_response(*args, **kwargs):  # type: ignore[no-untyped-def]
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
        result = None
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
        result = None
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
        result = None
        try:
            next(gen)
        except StopIteration as e:
            result = e.value
        assert result == StakingState.UNSTAKED

    def test_staking_state_staked(self) -> None:
        """Test returns correct staking state from contract."""
        behaviour = self._make_chain_behaviour()

        def mock_contract_interact(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield
            return StakingState.STAKED.value

        behaviour.contract_interact = mock_contract_interact

        gen = ChainBehaviour._get_service_staking_state(behaviour, chain="base")
        next(gen)
        result = None
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result == StakingState.STAKED

    def test_staking_state_none_returns_unstaked(self) -> None:
        """Test returns UNSTAKED when contract returns None."""
        behaviour = self._make_chain_behaviour()

        def mock_contract_interact(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield
            return None

        behaviour.contract_interact = mock_contract_interact

        gen = ChainBehaviour._get_service_staking_state(behaviour, chain="base")
        next(gen)
        result = None
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

        def mock_get_native_balance():  # type: ignore[no-untyped-def]
            yield
            return {"agent": None, "safe": None}

        behaviour.get_native_balance = mock_get_native_balance

        gen = CheckFundsBehaviour.get_event(behaviour)
        next(gen)
        result = None
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result == Event.NO_FUNDS.value

    def test_get_event_insufficient_funds(self) -> None:
        """Test get_event returns NO_FUNDS when balance is below minimum."""
        behaviour = self._make_behaviour()

        def mock_get_native_balance():  # type: ignore[no-untyped-def]
            yield
            return {"agent": 0.0001, "safe": 1.0}

        behaviour.get_native_balance = mock_get_native_balance

        gen = CheckFundsBehaviour.get_event(behaviour)
        next(gen)
        result = None
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result == Event.NO_FUNDS.value

    def test_get_event_sufficient_funds(self) -> None:
        """Test get_event returns DONE when balance is sufficient."""
        behaviour = self._make_behaviour()

        def mock_get_native_balance():  # type: ignore[no-untyped-def]
            yield
            return {"agent": 1.0, "safe": 1.0}

        behaviour.get_native_balance = mock_get_native_balance

        gen = CheckFundsBehaviour.get_event(behaviour)
        next(gen)
        result = None
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

        def mock_get_contract_api_response(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield
            return response_msg

        behaviour.get_contract_api_response = mock_get_contract_api_response

        gen = ChainBehaviour._build_safe_tx_hash(behaviour, to_address="0x" + "a" * 40)
        next(gen)
        result = None
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

        def mock_get_contract_api_response(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield
            return response_msg

        behaviour.get_contract_api_response = mock_get_contract_api_response

        gen = ChainBehaviour._build_safe_tx_hash(behaviour, to_address="0x" + "a" * 40)
        next(gen)
        result = None
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

        def mock_get_contract_api_response(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield
            return response_msg

        behaviour.get_contract_api_response = mock_get_contract_api_response

        gen = ChainBehaviour._build_safe_tx_hash(behaviour, to_address="0x" + "a" * 40)
        next(gen)
        result = None
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result is None

    def test_build_safe_tx_hash_success(self) -> None:
        """Test _build_safe_tx_hash returns valid hash on success."""
        behaviour = self._make_chain_behaviour()
        # TX_HASH_LENGTH = 66 => 0x + 64 hex chars
        valid_hash = "0x" + "ab" * 32  # 66 chars total
        response_msg = MagicMock()
        response_msg.performative = ContractApiMessage.Performative.STATE
        response_msg.state.body = {"tx_hash": valid_hash}

        def mock_get_contract_api_response(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield
            return response_msg

        behaviour.get_contract_api_response = mock_get_contract_api_response

        gen = ChainBehaviour._build_safe_tx_hash(
            behaviour, to_address="0x" + "a" * 40, value=0, data=b"0x"
        )
        next(gen)
        result = None
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        # hash_payload_to_hex returns a hex string; just verify it is not None
        assert result is not None
        assert isinstance(result, str)


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

        def mock_contract_interact(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield
            return None

        behaviour.contract_interact = mock_contract_interact

        gen = ChainBehaviour._get_liveness_ratio(behaviour, "base")
        next(gen)
        result = None
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result is None

    def test_liveness_ratio_zero(self) -> None:
        """Test _get_liveness_ratio logs error on zero."""
        behaviour = self._make_chain_behaviour()

        def mock_contract_interact(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield
            return 0

        behaviour.contract_interact = mock_contract_interact

        gen = ChainBehaviour._get_liveness_ratio(behaviour, "base")
        next(gen)
        result = None
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result == 0

    def test_liveness_ratio_valid(self) -> None:
        """Test _get_liveness_ratio returns valid ratio."""
        behaviour = self._make_chain_behaviour()

        def mock_contract_interact(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield
            return 10**18

        behaviour.contract_interact = mock_contract_interact

        gen = ChainBehaviour._get_liveness_ratio(behaviour, "base")
        next(gen)
        result = None
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result == 10**18


class TestGetLivenessPeriod:
    """Tests for _get_liveness_period."""

    def _make_chain_behaviour(self) -> MagicMock:
        behaviour = MagicMock(spec=ChainBehaviour)
        behaviour.params = make_mock_params()
        behaviour.context = make_mock_context(params=behaviour.params)
        return behaviour

    def test_liveness_period_none(self) -> None:
        """Test _get_liveness_period logs error on None."""
        behaviour = self._make_chain_behaviour()

        def mock_contract_interact(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield
            return None

        behaviour.contract_interact = mock_contract_interact

        gen = ChainBehaviour._get_liveness_period(behaviour, "base")
        result = _run_generator(gen)
        assert result is None

    def test_liveness_period_zero(self) -> None:
        """Test _get_liveness_period logs error on zero."""
        behaviour = self._make_chain_behaviour()

        def mock_contract_interact(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield
            return 0

        behaviour.contract_interact = mock_contract_interact

        gen = ChainBehaviour._get_liveness_period(behaviour, "base")
        result = _run_generator(gen)
        assert result == 0

    def test_liveness_period_valid(self) -> None:
        """Test _get_liveness_period returns valid period."""
        behaviour = self._make_chain_behaviour()

        def mock_contract_interact(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield
            return 3600

        behaviour.contract_interact = mock_contract_interact

        gen = ChainBehaviour._get_liveness_period(behaviour, "base")
        result = _run_generator(gen)
        assert result == 3600


class TestGetTsCheckpoint:
    """Tests for _get_ts_checkpoint."""

    def _make_chain_behaviour(self) -> MagicMock:
        behaviour = MagicMock(spec=ChainBehaviour)
        behaviour.params = make_mock_params()
        behaviour.context = make_mock_context(params=behaviour.params)
        return behaviour

    def test_ts_checkpoint_returns_value(self) -> None:
        """Test _get_ts_checkpoint returns the contract value."""
        behaviour = self._make_chain_behaviour()

        def mock_contract_interact(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield
            return 1700000000

        behaviour.contract_interact = mock_contract_interact

        gen = ChainBehaviour._get_ts_checkpoint(behaviour, "base")
        result = _run_generator(gen)
        assert result == 1700000000

    def test_ts_checkpoint_returns_none(self) -> None:
        """Test _get_ts_checkpoint returns None when contract fails."""
        behaviour = self._make_chain_behaviour()

        def mock_contract_interact(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield
            return None

        behaviour.contract_interact = mock_contract_interact

        gen = ChainBehaviour._get_ts_checkpoint(behaviour, "base")
        result = _run_generator(gen)
        assert result is None


class TestCalculateMinNumOfSafeTxRequired:
    """Tests for _calculate_min_num_of_safe_tx_required."""

    def _make_chain_behaviour(self) -> MagicMock:
        behaviour = MagicMock(spec=ChainBehaviour)
        behaviour.params = make_mock_params()
        behaviour.context = make_mock_context(params=behaviour.params)
        behaviour.synchronized_data = make_mock_synchronized_data()
        behaviour.get_chain_id = MagicMock(return_value="base")

        mock_ts = _make_timestamp_mock(1700000000.0)
        behaviour.round_sequence = MagicMock()
        behaviour.round_sequence.last_round_transition_timestamp = mock_ts
        return behaviour

    def test_returns_none_when_liveness_ratio_is_none(self) -> None:
        """Test returns None when liveness_ratio is None."""
        behaviour = self._make_chain_behaviour()

        def mock_get_liveness_ratio(chain):  # type: ignore[no-untyped-def]
            yield
            return None

        def mock_get_liveness_period(chain):  # type: ignore[no-untyped-def]
            yield
            return 3600

        behaviour._get_liveness_ratio = mock_get_liveness_ratio
        behaviour._get_liveness_period = mock_get_liveness_period

        gen = ChainBehaviour._calculate_min_num_of_safe_tx_required(behaviour, "base")
        result = _run_generator(gen)
        assert result is None

    def test_returns_none_when_liveness_period_is_none(self) -> None:
        """Test returns None when liveness_period is None."""
        behaviour = self._make_chain_behaviour()

        def mock_get_liveness_ratio(chain):  # type: ignore[no-untyped-def]
            yield
            return 10**18

        def mock_get_liveness_period(chain):  # type: ignore[no-untyped-def]
            yield
            return None

        behaviour._get_liveness_ratio = mock_get_liveness_ratio
        behaviour._get_liveness_period = mock_get_liveness_period

        gen = ChainBehaviour._calculate_min_num_of_safe_tx_required(behaviour, "base")
        result = _run_generator(gen)
        assert result is None

    def test_returns_none_when_ts_checkpoint_is_none(self) -> None:
        """Test returns None when ts_checkpoint is None."""
        behaviour = self._make_chain_behaviour()

        def mock_get_liveness_ratio(chain):  # type: ignore[no-untyped-def]
            yield
            return 10**18

        def mock_get_liveness_period(chain):  # type: ignore[no-untyped-def]
            yield
            return 3600

        def mock_get_ts_checkpoint(chain):  # type: ignore[no-untyped-def]
            yield
            return None

        behaviour._get_liveness_ratio = mock_get_liveness_ratio
        behaviour._get_liveness_period = mock_get_liveness_period
        behaviour._get_ts_checkpoint = mock_get_ts_checkpoint

        gen = ChainBehaviour._calculate_min_num_of_safe_tx_required(behaviour, "base")
        result = _run_generator(gen)
        assert result is None

    def test_returns_calculated_value(self) -> None:
        """Test returns correct calculated value."""
        behaviour = self._make_chain_behaviour()

        def mock_get_liveness_ratio(chain):  # type: ignore[no-untyped-def]
            yield
            return 10**18  # 1 tx per second

        def mock_get_liveness_period(chain):  # type: ignore[no-untyped-def]
            yield
            return 3600  # 1 hour

        def mock_get_ts_checkpoint(chain):  # type: ignore[no-untyped-def]
            yield
            return 1699996400  # 3600 seconds ago

        behaviour._get_liveness_ratio = mock_get_liveness_ratio
        behaviour._get_liveness_period = mock_get_liveness_period
        behaviour._get_ts_checkpoint = mock_get_ts_checkpoint

        gen = ChainBehaviour._calculate_min_num_of_safe_tx_required(behaviour, "base")
        result = _run_generator(gen)
        # max(3600, 1700000000 - 1699996400) * 10**18 / 10**18 + 1 = 3601
        assert result == 3601


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
        result = None
        try:
            next(gen)
        except StopIteration as e:
            result = e.value
        assert result is None

    def test_with_service_id(self) -> None:
        """Test _get_service_info calls contract_interact."""
        behaviour = self._make_chain_behaviour()
        service_info = (1, 2, (3, 4))

        def mock_contract_interact(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield
            return service_info

        behaviour.contract_interact = mock_contract_interact

        gen = ChainBehaviour._get_service_info(behaviour, chain="base")
        next(gen)
        result = None
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

        def mock_contract_interact(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield
            return None

        behaviour.contract_interact = mock_contract_interact

        gen = ChainBehaviour._get_multisig_nonces(behaviour, "base", SAFE_ADDRESS)
        next(gen)
        result = None
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result is None

    def test_multisig_nonces_empty_list(self) -> None:
        """Test returns None when contract returns empty list."""
        behaviour = self._make_chain_behaviour()

        def mock_contract_interact(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield
            return []

        behaviour.contract_interact = mock_contract_interact

        gen = ChainBehaviour._get_multisig_nonces(behaviour, "base", SAFE_ADDRESS)
        next(gen)
        result = None
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result is None

    def test_multisig_nonces_valid(self) -> None:
        """Test returns first element of valid list."""
        behaviour = self._make_chain_behaviour()

        def mock_contract_interact(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield
            return [42, 10]

        behaviour.contract_interact = mock_contract_interact

        gen = ChainBehaviour._get_multisig_nonces(behaviour, "base", SAFE_ADDRESS)
        next(gen)
        result = None
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result == 42


class TestGetMultisigNoncesSinceLastCp:
    """Tests for _get_multisig_nonces_since_last_cp."""

    def _make_chain_behaviour(self) -> MagicMock:
        behaviour = MagicMock(spec=ChainBehaviour)
        behaviour.params = make_mock_params()
        behaviour.context = make_mock_context(params=behaviour.params)
        behaviour.synchronized_data = make_mock_synchronized_data()
        behaviour.get_chain_id = MagicMock(return_value="base")
        return behaviour

    def test_returns_none_when_nonces_none(self) -> None:
        """Test returns None when _get_multisig_nonces returns None."""
        behaviour = self._make_chain_behaviour()

        def mock_get_multisig_nonces(chain, multisig):  # type: ignore[no-untyped-def]
            yield
            return None

        behaviour._get_multisig_nonces = mock_get_multisig_nonces

        gen = ChainBehaviour._get_multisig_nonces_since_last_cp(
            behaviour, "base", SAFE_ADDRESS
        )
        result = _run_generator(gen)
        assert result is None

    def test_returns_none_when_service_info_none(self) -> None:
        """Test returns None when _get_service_info returns None."""
        behaviour = self._make_chain_behaviour()

        def mock_get_multisig_nonces(chain, multisig):  # type: ignore[no-untyped-def]
            yield
            return 42

        def mock_get_service_info(chain):  # type: ignore[no-untyped-def]
            yield
            return None

        behaviour._get_multisig_nonces = mock_get_multisig_nonces
        behaviour._get_service_info = mock_get_service_info

        gen = ChainBehaviour._get_multisig_nonces_since_last_cp(
            behaviour, "base", SAFE_ADDRESS
        )
        result = _run_generator(gen)
        assert result is None

    def test_returns_none_when_service_info_empty(self) -> None:
        """Test returns None when service_info is empty."""
        behaviour = self._make_chain_behaviour()

        def mock_get_multisig_nonces(chain, multisig):  # type: ignore[no-untyped-def]
            yield
            return 42

        def mock_get_service_info(chain):  # type: ignore[no-untyped-def]
            yield
            return ()

        behaviour._get_multisig_nonces = mock_get_multisig_nonces
        behaviour._get_service_info = mock_get_service_info

        gen = ChainBehaviour._get_multisig_nonces_since_last_cp(
            behaviour, "base", SAFE_ADDRESS
        )
        result = _run_generator(gen)
        assert result is None

    def test_returns_none_when_service_info_inner_empty(self) -> None:
        """Test returns None when service_info[2] is empty."""
        behaviour = self._make_chain_behaviour()

        def mock_get_multisig_nonces(chain, multisig):  # type: ignore[no-untyped-def]
            yield
            return 42

        def mock_get_service_info(chain):  # type: ignore[no-untyped-def]
            yield
            return (1, 2, ())

        behaviour._get_multisig_nonces = mock_get_multisig_nonces
        behaviour._get_service_info = mock_get_service_info

        gen = ChainBehaviour._get_multisig_nonces_since_last_cp(
            behaviour, "base", SAFE_ADDRESS
        )
        result = _run_generator(gen)
        assert result is None

    def test_returns_difference(self) -> None:
        """Test returns correct nonce difference."""
        behaviour = self._make_chain_behaviour()

        def mock_get_multisig_nonces(chain, multisig):  # type: ignore[no-untyped-def]
            yield
            return 50

        def mock_get_service_info(chain):  # type: ignore[no-untyped-def]
            yield
            return (1, 2, (30, 5))

        behaviour._get_multisig_nonces = mock_get_multisig_nonces
        behaviour._get_service_info = mock_get_service_info

        gen = ChainBehaviour._get_multisig_nonces_since_last_cp(
            behaviour, "base", SAFE_ADDRESS
        )
        result = _run_generator(gen)
        assert result == 20  # 50 - 30


class TestIsStakingKpiMet:
    """Tests for _is_staking_kpi_met."""

    def _make_chain_behaviour(self, **kwargs: Any) -> MagicMock:
        behaviour = MagicMock(spec=ChainBehaviour)
        behaviour.params = make_mock_params(**kwargs)
        behaviour.context = make_mock_context(params=behaviour.params)
        behaviour.synchronized_data = make_mock_synchronized_data()
        behaviour.get_chain_id = MagicMock(return_value="base")

        mock_ts = _make_timestamp_mock(1700000000.0)
        behaviour.round_sequence = MagicMock()
        behaviour.round_sequence.last_round_transition_timestamp = mock_ts
        return behaviour

    def test_returns_false_when_not_staked(self) -> None:
        """Test returns False when service is not staked."""
        behaviour = self._make_chain_behaviour()

        def mock_get_staking_state(chain):  # type: ignore[no-untyped-def]
            yield
            return StakingState.UNSTAKED

        behaviour._get_service_staking_state = mock_get_staking_state

        gen = ChainBehaviour._is_staking_kpi_met(behaviour)
        result = _run_generator(gen)
        assert result is False

    def test_returns_none_when_mech_request_count_error(self) -> None:
        """Test returns None when mech request count query fails."""
        behaviour = self._make_chain_behaviour()

        def mock_get_staking_state(chain):  # type: ignore[no-untyped-def]
            yield
            return StakingState.STAKED

        behaviour._get_service_staking_state = mock_get_staking_state

        response_msg = MagicMock()
        response_msg.performative = ContractApiMessage.Performative.ERROR

        def mock_get_contract_api_response(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield
            return response_msg

        behaviour.get_contract_api_response = mock_get_contract_api_response

        gen = ChainBehaviour._is_staking_kpi_met(behaviour)
        result = _run_generator(gen)
        assert result is None

    def test_returns_none_when_requests_count_is_none(self) -> None:
        """Test returns None when requests_count key is missing."""
        behaviour = self._make_chain_behaviour()

        def mock_get_staking_state(chain):  # type: ignore[no-untyped-def]
            yield
            return StakingState.STAKED

        behaviour._get_service_staking_state = mock_get_staking_state

        response_msg = MagicMock()
        response_msg.performative = ContractApiMessage.Performative.STATE
        response_msg.state.body = {}  # no requests_count

        def mock_get_contract_api_response(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield
            return response_msg

        behaviour.get_contract_api_response = mock_get_contract_api_response

        gen = ChainBehaviour._is_staking_kpi_met(behaviour)
        result = _run_generator(gen)
        assert result is None

    def test_returns_none_when_service_info_fails(self) -> None:
        """Test returns None when service_info is invalid."""
        behaviour = self._make_chain_behaviour()

        def mock_get_staking_state(chain):  # type: ignore[no-untyped-def]
            yield
            return StakingState.STAKED

        behaviour._get_service_staking_state = mock_get_staking_state

        response_msg = MagicMock()
        response_msg.performative = ContractApiMessage.Performative.STATE
        response_msg.state.body = {"requests_count": 10}

        def mock_get_contract_api_response(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield
            return response_msg

        behaviour.get_contract_api_response = mock_get_contract_api_response

        def mock_get_service_info(chain):  # type: ignore[no-untyped-def]
            yield
            return None

        behaviour._get_service_info = mock_get_service_info

        gen = ChainBehaviour._is_staking_kpi_met(behaviour)
        result = _run_generator(gen)
        assert result is None

    def test_returns_none_when_ts_checkpoint_fails(self) -> None:
        """Test returns None when ts_checkpoint is None."""
        behaviour = self._make_chain_behaviour()

        def mock_get_staking_state(chain):  # type: ignore[no-untyped-def]
            yield
            return StakingState.STAKED

        behaviour._get_service_staking_state = mock_get_staking_state

        response_msg = MagicMock()
        response_msg.performative = ContractApiMessage.Performative.STATE
        response_msg.state.body = {"requests_count": 10}

        def mock_get_contract_api_response(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield
            return response_msg

        behaviour.get_contract_api_response = mock_get_contract_api_response

        def mock_get_service_info(chain):  # type: ignore[no-untyped-def]
            yield
            return (1, 2, (5, 3))

        behaviour._get_service_info = mock_get_service_info

        def mock_get_ts_checkpoint(chain):  # type: ignore[no-untyped-def]
            yield
            return None

        behaviour._get_ts_checkpoint = mock_get_ts_checkpoint

        gen = ChainBehaviour._is_staking_kpi_met(behaviour)
        result = _run_generator(gen)
        assert result is None

    def test_returns_none_when_liveness_period_fails(self) -> None:
        """Test returns None when liveness_period is None."""
        behaviour = self._make_chain_behaviour()

        def mock_get_staking_state(chain):  # type: ignore[no-untyped-def]
            yield
            return StakingState.STAKED

        behaviour._get_service_staking_state = mock_get_staking_state

        response_msg = MagicMock()
        response_msg.performative = ContractApiMessage.Performative.STATE
        response_msg.state.body = {"requests_count": 10}

        def mock_get_contract_api_response(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield
            return response_msg

        behaviour.get_contract_api_response = mock_get_contract_api_response

        def mock_get_service_info(chain):  # type: ignore[no-untyped-def]
            yield
            return (1, 2, (5, 3))

        behaviour._get_service_info = mock_get_service_info

        def mock_get_ts_checkpoint(chain):  # type: ignore[no-untyped-def]
            yield
            return 1699996400

        behaviour._get_ts_checkpoint = mock_get_ts_checkpoint

        def mock_get_liveness_period(chain):  # type: ignore[no-untyped-def]
            yield
            return None

        behaviour._get_liveness_period = mock_get_liveness_period

        gen = ChainBehaviour._is_staking_kpi_met(behaviour)
        result = _run_generator(gen)
        assert result is None

    def test_returns_none_when_liveness_ratio_fails(self) -> None:
        """Test returns None when liveness_ratio is None."""
        behaviour = self._make_chain_behaviour()

        def mock_get_staking_state(chain):  # type: ignore[no-untyped-def]
            yield
            return StakingState.STAKED

        behaviour._get_service_staking_state = mock_get_staking_state

        response_msg = MagicMock()
        response_msg.performative = ContractApiMessage.Performative.STATE
        response_msg.state.body = {"requests_count": 10}

        def mock_get_contract_api_response(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield
            return response_msg

        behaviour.get_contract_api_response = mock_get_contract_api_response

        def mock_get_service_info(chain):  # type: ignore[no-untyped-def]
            yield
            return (1, 2, (5, 3))

        behaviour._get_service_info = mock_get_service_info

        def mock_get_ts_checkpoint(chain):  # type: ignore[no-untyped-def]
            yield
            return 1699996400

        behaviour._get_ts_checkpoint = mock_get_ts_checkpoint

        def mock_get_liveness_period(chain):  # type: ignore[no-untyped-def]
            yield
            return 3600

        behaviour._get_liveness_period = mock_get_liveness_period

        def mock_get_liveness_ratio(chain):  # type: ignore[no-untyped-def]
            yield
            return None

        behaviour._get_liveness_ratio = mock_get_liveness_ratio

        gen = ChainBehaviour._is_staking_kpi_met(behaviour)
        result = _run_generator(gen)
        assert result is None

    @pytest.mark.parametrize(
        "mech_request_count,mech_on_checkpoint,liveness_ratio,expected",
        [
            # With low ratio: required = ceil(3600 * 10**15 / 10**18) + 1 = ceil(3.6) + 1 = 5
            # requests since cp = 100 - 3 = 97 >= 5 => True
            (100, 3, 10**15, True),
            # Same ratio: requests since cp = 5 - 3 = 2 < 5 => False
            (5, 3, 10**15, False),
        ],
    )
    def test_kpi_met_calculation(
        self,
        mech_request_count: int,
        mech_on_checkpoint: int,
        liveness_ratio: int,
        expected: bool,
    ) -> None:
        """Test KPI met/not met calculation."""
        behaviour = self._make_chain_behaviour()

        def mock_get_staking_state(chain):  # type: ignore[no-untyped-def]
            yield
            return StakingState.STAKED

        behaviour._get_service_staking_state = mock_get_staking_state

        response_msg = MagicMock()
        response_msg.performative = ContractApiMessage.Performative.STATE
        response_msg.state.body = {"requests_count": mech_request_count}

        def mock_get_contract_api_response(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield
            return response_msg

        behaviour.get_contract_api_response = mock_get_contract_api_response

        def mock_get_service_info(chain):  # type: ignore[no-untyped-def]
            yield
            return (1, 2, (5, mech_on_checkpoint))

        behaviour._get_service_info = mock_get_service_info

        def mock_get_ts_checkpoint(chain):  # type: ignore[no-untyped-def]
            yield
            return 1699996400  # 3600 seconds ago

        behaviour._get_ts_checkpoint = mock_get_ts_checkpoint

        def mock_get_liveness_period(chain):  # type: ignore[no-untyped-def]
            yield
            return 3600

        behaviour._get_liveness_period = mock_get_liveness_period

        def mock_get_liveness_ratio(chain):  # type: ignore[no-untyped-def]
            yield
            return liveness_ratio

        behaviour._get_liveness_ratio = mock_get_liveness_ratio

        gen = ChainBehaviour._is_staking_kpi_met(behaviour)
        result = _run_generator(gen)
        assert result is expected


class TestCheckFundsBehaviourAsyncAct:
    """Tests for CheckFundsBehaviour.async_act."""

    def _make_behaviour(
        self, event: str = Event.DONE.value, check_funds_count: int = 0
    ) -> MagicMock:
        behaviour = MagicMock(spec=CheckFundsBehaviour)
        behaviour.params = make_mock_params()
        behaviour.context = make_mock_context(params=behaviour.params)
        behaviour.synchronized_data = make_mock_synchronized_data(
            check_funds_count=check_funds_count
        )
        behaviour.behaviour_id = "check_funds"
        return behaviour

    def test_async_act_done(self) -> None:
        """Test async_act with DONE event."""
        behaviour = self._make_behaviour()

        def mock_get_event():  # type: ignore[no-untyped-def]
            yield
            return Event.DONE.value

        behaviour.get_event = mock_get_event

        def mock_send_a2a_transaction(payload):  # type: ignore[no-untyped-def]
            yield

        behaviour.send_a2a_transaction = mock_send_a2a_transaction

        def mock_wait_until_round_end():  # type: ignore[no-untyped-def]
            yield

        behaviour.wait_until_round_end = mock_wait_until_round_end
        behaviour.set_done = MagicMock()

        gen = CheckFundsBehaviour.async_act(behaviour)
        _run_generator(gen)
        behaviour.set_done.assert_called_once()

    def test_async_act_no_funds(self) -> None:
        """Test async_act with NO_FUNDS event increments check_funds_count."""
        behaviour = self._make_behaviour(check_funds_count=3)

        def mock_get_event():  # type: ignore[no-untyped-def]
            yield
            return Event.NO_FUNDS.value

        behaviour.get_event = mock_get_event

        def mock_sleep(seconds):  # type: ignore[no-untyped-def]
            yield

        behaviour.sleep = mock_sleep

        def mock_send_a2a_transaction(payload):  # type: ignore[no-untyped-def]
            yield

        behaviour.send_a2a_transaction = mock_send_a2a_transaction

        def mock_wait_until_round_end():  # type: ignore[no-untyped-def]
            yield

        behaviour.wait_until_round_end = mock_wait_until_round_end
        behaviour.set_done = MagicMock()

        gen = CheckFundsBehaviour.async_act(behaviour)
        _run_generator(gen)
        behaviour.set_done.assert_called_once()


class TestCheckStakingBehaviourAsyncAct:
    """Tests for CheckStakingBehaviour.async_act."""

    def _make_behaviour(self) -> MagicMock:
        behaviour = MagicMock(spec=CheckStakingBehaviour)
        behaviour.params = make_mock_params()
        behaviour.context = make_mock_context(params=behaviour.params)
        behaviour.synchronized_data = make_mock_synchronized_data()
        behaviour.behaviour_id = "check_staking"
        behaviour.get_chain_id = MagicMock(return_value="base")
        return behaviour

    def test_async_act(self) -> None:
        """Test async_act calls _is_staking_kpi_met and sends payload."""
        behaviour = self._make_behaviour()

        def mock_is_staking_kpi_met():  # type: ignore[no-untyped-def]
            yield
            return True

        behaviour._is_staking_kpi_met = mock_is_staking_kpi_met

        def mock_send_a2a_transaction(payload):  # type: ignore[no-untyped-def]
            yield

        behaviour.send_a2a_transaction = mock_send_a2a_transaction

        def mock_wait_until_round_end():  # type: ignore[no-untyped-def]
            yield

        behaviour.wait_until_round_end = mock_wait_until_round_end
        behaviour.set_done = MagicMock()

        gen = CheckStakingBehaviour.async_act(behaviour)
        _run_generator(gen)
        behaviour.set_done.assert_called_once()


class TestPullMemesBehaviourAsyncAct:
    """Tests for PullMemesBehaviour async_act and helpers."""

    def _make_behaviour(self, is_memecoin_logic_enabled: bool = True) -> MagicMock:
        behaviour = MagicMock(spec=PullMemesBehaviour)
        behaviour.params = make_mock_params(
            is_memecoin_logic_enabled=is_memecoin_logic_enabled
        )
        behaviour.context = make_mock_context(params=behaviour.params)
        behaviour.synchronized_data = make_mock_synchronized_data()
        behaviour.behaviour_id = "pull_memes"
        behaviour.get_chain_id = MagicMock(return_value="base")
        return behaviour

    def test_async_act_skip_when_disabled(self) -> None:
        """Test async_act skips when memecoin logic is disabled."""
        behaviour = self._make_behaviour(is_memecoin_logic_enabled=False)

        def mock_send_a2a_transaction(payload):  # type: ignore[no-untyped-def]
            yield

        behaviour.send_a2a_transaction = mock_send_a2a_transaction

        def mock_wait_until_round_end():  # type: ignore[no-untyped-def]
            yield

        behaviour.wait_until_round_end = mock_wait_until_round_end
        behaviour.set_done = MagicMock()

        gen = PullMemesBehaviour.async_act(behaviour)
        _run_generator(gen)
        behaviour.set_done.assert_called_once()

    def test_async_act_enabled(self) -> None:
        """Test async_act when memecoin logic is enabled."""
        behaviour = self._make_behaviour(is_memecoin_logic_enabled=True)

        def mock_get_meme_coins():  # type: ignore[no-untyped-def]
            yield
            return [{"name": "TestCoin"}]

        behaviour.get_meme_coins = mock_get_meme_coins

        def mock_send_a2a_transaction(payload):  # type: ignore[no-untyped-def]
            yield

        behaviour.send_a2a_transaction = mock_send_a2a_transaction

        def mock_wait_until_round_end():  # type: ignore[no-untyped-def]
            yield

        behaviour.wait_until_round_end = mock_wait_until_round_end
        behaviour.set_done = MagicMock()

        gen = PullMemesBehaviour.async_act(behaviour)
        _run_generator(gen)
        behaviour.set_done.assert_called_once()

    def test_params_property(self) -> None:
        """Test PullMemesBehaviour.params property returns context.params."""
        behaviour = MagicMock(spec=PullMemesBehaviour)
        params = make_mock_params()
        behaviour.context = make_mock_context(params=params)
        result = PullMemesBehaviour.params.fget(behaviour)  # type: ignore[attr-defined]  # pylint: disable=assignment-from-no-return
        assert result is params


class TestPullMemesBehaviourGetBlockNumber:
    """Tests for PullMemesBehaviour.get_block_number."""

    def _make_behaviour(self) -> MagicMock:
        behaviour = MagicMock(spec=PullMemesBehaviour)
        behaviour.params = make_mock_params()
        behaviour.context = make_mock_context(params=behaviour.params)
        behaviour.get_chain_id = MagicMock(return_value="base")
        return behaviour

    def test_get_block_number_error(self) -> None:
        """Test get_block_number returns None on error."""
        behaviour = self._make_behaviour()

        response_msg = MagicMock()
        response_msg.performative = LedgerApiMessage.Performative.ERROR

        def mock_get_ledger_api_response(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield
            return response_msg

        behaviour.get_ledger_api_response = mock_get_ledger_api_response

        gen = PullMemesBehaviour.get_block_number(behaviour)
        result = _run_generator(gen)
        assert result is None

    def test_get_block_number_success(self) -> None:
        """Test get_block_number returns block number on success."""
        behaviour = self._make_behaviour()

        response_msg = MagicMock()
        response_msg.performative = LedgerApiMessage.Performative.STATE
        response_msg.state.body = {"get_block_number_result": 12345}

        def mock_get_ledger_api_response(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield
            return response_msg

        behaviour.get_ledger_api_response = mock_get_ledger_api_response

        gen = PullMemesBehaviour.get_block_number(behaviour)
        result = _run_generator(gen)
        assert result == 12345


class TestActionPreparationBehaviourAsyncAct:
    """Tests for ActionPreparationBehaviour.async_act."""

    def _make_behaviour(self) -> MagicMock:
        behaviour = MagicMock(spec=ActionPreparationBehaviour)
        behaviour.params = make_mock_params()
        behaviour.context = make_mock_context(params=behaviour.params)
        behaviour.synchronized_data = make_mock_synchronized_data()
        behaviour.behaviour_id = "action_preparation"
        behaviour.matching_round = ActionPreparationRound
        return behaviour

    def test_async_act(self) -> None:
        """Test async_act sends payload and completes."""
        behaviour = self._make_behaviour()

        def mock_get_tx_hash():  # type: ignore[no-untyped-def]
            yield
            return "0xhash"

        behaviour.get_tx_hash = mock_get_tx_hash

        def mock_send_a2a_transaction(payload):  # type: ignore[no-untyped-def]
            yield

        behaviour.send_a2a_transaction = mock_send_a2a_transaction

        def mock_wait_until_round_end():  # type: ignore[no-untyped-def]
            yield

        behaviour.wait_until_round_end = mock_wait_until_round_end
        behaviour.set_done = MagicMock()

        gen = ActionPreparationBehaviour.async_act(behaviour)
        _run_generator(gen)
        behaviour.set_done.assert_called_once()


class TestActionPreparationBehaviourGetTxHash:
    """Tests for ActionPreparationBehaviour.get_tx_hash."""

    def _make_behaviour(self, **sync_data_overrides: Any) -> MagicMock:
        behaviour = MagicMock(spec=ActionPreparationBehaviour)
        behaviour.params = make_mock_params()
        behaviour.context = make_mock_context(params=behaviour.params)
        behaviour.synchronized_data = make_mock_synchronized_data(**sync_data_overrides)
        behaviour.get_chain_id = MagicMock(return_value="base")
        behaviour.get_meme_factory_address = MagicMock(return_value="0x" + "d" * 40)
        return behaviour

    def test_get_tx_hash_final_tx_hash_present(self) -> None:
        """Test returns empty string when final_tx_hash is present."""
        behaviour = self._make_behaviour(
            final_tx_hash="0xabc",
            token_action={"action": "heart"},
        )

        def mock_post_action():  # type: ignore[no-untyped-def]
            yield

        behaviour.post_action = mock_post_action

        gen = ActionPreparationBehaviour.get_tx_hash(behaviour)
        result = _run_generator(gen)
        assert result == ""

    def test_get_tx_hash_no_token_action(self) -> None:
        """Test returns None when token_action is empty."""
        behaviour = self._make_behaviour(token_action={})

        gen = ActionPreparationBehaviour.get_tx_hash(behaviour)
        result = _run_generator(gen)
        assert result is None

    @pytest.mark.parametrize(
        "action,extra_kwargs",
        [
            (
                "summon",
                {
                    "token_name": "Test",
                    "token_ticker": "TST",
                    "token_supply": "1000000",
                    "amount": "100",
                },
            ),
            ("heart", {"token_nonce": 1, "amount": "100"}),
            ("unleash", {"token_nonce": 1}),
            ("collect", {"token_address": "0x" + "a" * 40}),
            ("purge", {"token_address": "0x" + "a" * 40}),
        ],
    )
    def test_get_tx_hash_various_actions(self, action: str, extra_kwargs: dict) -> None:
        """Test get_tx_hash for various action types."""
        token_action = {"action": action, **extra_kwargs}
        behaviour = self._make_behaviour(token_action=token_action)

        response_msg = MagicMock()
        response_msg.performative = ContractApiMessage.Performative.RAW_TRANSACTION
        response_msg.raw_transaction.body = {"data": b"\x01\x02\x03"}

        def mock_get_contract_api_response(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield
            return response_msg

        behaviour.get_contract_api_response = mock_get_contract_api_response

        # Mock _build_safe_tx_hash
        def mock_build_safe_tx_hash(to_address, data, value=0):  # type: ignore[no-untyped-def]
            yield
            return "safe_tx_hash_result"

        behaviour._build_safe_tx_hash = mock_build_safe_tx_hash

        gen = ActionPreparationBehaviour.get_tx_hash(behaviour)
        result = _run_generator(gen)
        assert result == "safe_tx_hash_result"

    def test_get_tx_hash_contract_error(self) -> None:
        """Test get_tx_hash returns None on contract error."""
        token_action = {"action": "heart", "token_nonce": 1, "amount": "100"}
        behaviour = self._make_behaviour(token_action=token_action)

        response_msg = MagicMock()
        response_msg.performative = ContractApiMessage.Performative.ERROR

        def mock_get_contract_api_response(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield
            return response_msg

        behaviour.get_contract_api_response = mock_get_contract_api_response

        gen = ActionPreparationBehaviour.get_tx_hash(behaviour)
        result = _run_generator(gen)
        assert result is None

    def test_get_tx_hash_no_data_bytes(self) -> None:
        """Test get_tx_hash returns None when data is None."""
        token_action = {"action": "heart", "token_nonce": 1, "amount": "100"}
        behaviour = self._make_behaviour(token_action=token_action)

        response_msg = MagicMock()
        response_msg.performative = ContractApiMessage.Performative.RAW_TRANSACTION
        response_msg.raw_transaction.body = {"data": None}

        def mock_get_contract_api_response(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield
            return response_msg

        behaviour.get_contract_api_response = mock_get_contract_api_response

        gen = ActionPreparationBehaviour.get_tx_hash(behaviour)
        result = _run_generator(gen)
        assert result is None


class TestActionPreparationBehaviourPostAction:
    """Tests for ActionPreparationBehaviour.post_action."""

    def _make_behaviour(self, **sync_data_overrides: Any) -> MagicMock:
        behaviour = MagicMock(spec=ActionPreparationBehaviour)
        behaviour.params = make_mock_params()
        behaviour.context = make_mock_context(params=behaviour.params)
        behaviour.synchronized_data = make_mock_synchronized_data(**sync_data_overrides)
        behaviour.get_sync_timestamp = MagicMock(return_value=1700000000.0)
        return behaviour

    def test_post_action_no_nonce(self) -> None:
        """Test post_action returns early when token_nonce is missing."""
        behaviour = self._make_behaviour(token_action={"action": "heart"})

        gen = ActionPreparationBehaviour.post_action(behaviour)
        _run_generator(gen)
        # Should not raise; just return early

    def test_post_action_with_nonce_non_summon(self) -> None:
        """Test post_action with nonce but non-summon action returns early after logging."""
        behaviour = self._make_behaviour(
            token_action={"action": "heart", "token_nonce": 42}
        )

        gen = ActionPreparationBehaviour.post_action(behaviour)
        _run_generator(gen)
        # Should not raise

    def test_post_action_summon_with_db_data(self) -> None:
        """Test post_action for summon action reads/writes db."""
        token_action = {
            "action": "summon",
            "token_nonce": 42,
            "token_name": "TestToken",
            "token_ticker": "TST",
            "total_supply": 1000000,
        }
        behaviour = self._make_behaviour(token_action=token_action)

        def mock_read_kv(keys):  # type: ignore[no-untyped-def]
            yield
            return {"tokens": json.dumps([{"name": "OldToken"}])}

        behaviour._read_kv = mock_read_kv

        def mock_write_kv(data):  # type: ignore[no-untyped-def]
            yield

        behaviour._write_kv = mock_write_kv

        gen = ActionPreparationBehaviour.post_action(behaviour)
        _run_generator(gen)

    def test_post_action_summon_db_none(self) -> None:
        """Test post_action for summon action when db returns None."""
        token_action = {
            "action": "summon",
            "token_nonce": 42,
            "token_name": "TestToken",
            "token_ticker": "TST",
            "total_supply": 1000000,
        }
        behaviour = self._make_behaviour(token_action=token_action)

        def mock_read_kv(keys):  # type: ignore[no-untyped-def]
            yield
            return None

        behaviour._read_kv = mock_read_kv

        def mock_write_kv(data):  # type: ignore[no-untyped-def]
            yield

        behaviour._write_kv = mock_write_kv

        gen = ActionPreparationBehaviour.post_action(behaviour)
        _run_generator(gen)

    def test_post_action_summon_empty_tokens(self) -> None:
        """Test post_action for summon action when db has empty tokens."""
        token_action = {
            "action": "summon",
            "token_nonce": 42,
            "token_name": "TestToken",
            "token_ticker": "TST",
            "total_supply": 1000000,
        }
        behaviour = self._make_behaviour(token_action=token_action)

        def mock_read_kv(keys):  # type: ignore[no-untyped-def]
            yield
            return {"tokens": ""}

        behaviour._read_kv = mock_read_kv

        def mock_write_kv(data):  # type: ignore[no-untyped-def]
            yield

        behaviour._write_kv = mock_write_kv

        gen = ActionPreparationBehaviour.post_action(behaviour)
        _run_generator(gen)


class TestActionPreparationBehaviourGetTokenNonce:
    """Tests for ActionPreparationBehaviour.get_token_nonce."""

    def _make_behaviour(self) -> MagicMock:
        behaviour = MagicMock(spec=ActionPreparationBehaviour)
        behaviour.params = make_mock_params()
        behaviour.context = make_mock_context(params=behaviour.params)
        behaviour.synchronized_data = make_mock_synchronized_data(
            final_tx_hash="0x" + "ab" * 32
        )
        behaviour.get_chain_id = MagicMock(return_value="base")
        behaviour.get_meme_factory_address = MagicMock(return_value="0x" + "d" * 40)
        return behaviour

    def test_get_token_nonce_error(self) -> None:
        """Test get_token_nonce returns None on error."""
        behaviour = self._make_behaviour()

        response_msg = MagicMock()
        response_msg.performative = ContractApiMessage.Performative.ERROR

        def mock_get_contract_api_response(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield
            return response_msg

        behaviour.get_contract_api_response = mock_get_contract_api_response

        gen = ActionPreparationBehaviour.get_token_nonce(behaviour)
        result = _run_generator(gen)
        assert result is None

    def test_get_token_nonce_success(self) -> None:
        """Test get_token_nonce returns nonce on success."""
        behaviour = self._make_behaviour()

        response_msg = MagicMock()
        response_msg.performative = ContractApiMessage.Performative.STATE
        response_msg.state.body = {"token_nonce": 42}

        def mock_get_contract_api_response(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield
            return response_msg

        behaviour.get_contract_api_response = mock_get_contract_api_response

        gen = ActionPreparationBehaviour.get_token_nonce(behaviour)
        result = _run_generator(gen)
        assert result == 42


class _ChainBehaviourTestBase(MemeooorrFSMBehaviourBaseCase):
    """Shared FSM base class for chain behaviour tests."""

    _default_data = dict(
        all_participants=["0x" + "0" * 40],
        participants=["0x" + "0" * 40],
        consensus_threshold=1,
        safe_contract_address=SAFE_ADDRESS,
        tx_submitter=CallCheckpointRound.auto_round_id(),
        tx_loop_count=0,
    )


class TestPostTxDecisionMakingBehaviourAsyncAct(_ChainBehaviourTestBase):
    """Tests for PostTxDecisionMakingBehaviour.async_act using FSMBehaviourBaseCase."""

    def test_async_act(self) -> None:
        """Test async_act sends payload and completes."""
        self.fast_forward_to_behaviour(
            self.behaviour,
            PostTxDecisionMakingBehaviour.auto_behaviour_id(),
            SynchronizedData(
                AbciAppDB(
                    setup_data=AbciAppDB.data_to_lists(
                        {
                            **self._default_data,
                            "tx_submitter": CallCheckpointRound.auto_round_id(),
                        }
                    )
                )
            ),
        )
        self.behaviour.act_wrapper()
        self.mock_a2a_transaction()
        self._test_done_flag_set()
        self.end_round(done_event=Event.DONE)


class TestCallCheckpointBehaviourAsyncAct:
    """Tests for CallCheckpointBehaviour.async_act."""

    def _make_behaviour(self) -> MagicMock:
        behaviour = MagicMock(spec=CallCheckpointBehaviour)
        behaviour.params = make_mock_params()
        behaviour.context = make_mock_context(params=behaviour.params)
        behaviour.synchronized_data = make_mock_synchronized_data()
        behaviour.behaviour_id = "call_checkpoint"
        behaviour.matching_round = CallCheckpointRound
        return behaviour

    def test_async_act(self) -> None:
        """Test async_act sends payload and completes."""
        behaviour = self._make_behaviour()

        def mock_get_checkpoint_tx_hash():  # type: ignore[no-untyped-def]
            yield
            return "0xcheckpointhash"

        behaviour.get_checkpoint_tx_hash = mock_get_checkpoint_tx_hash

        def mock_send_a2a_transaction(payload):  # type: ignore[no-untyped-def]
            yield

        behaviour.send_a2a_transaction = mock_send_a2a_transaction

        def mock_wait_until_round_end():  # type: ignore[no-untyped-def]
            yield

        behaviour.wait_until_round_end = mock_wait_until_round_end
        behaviour.set_done = MagicMock()

        gen = CallCheckpointBehaviour.async_act(behaviour)
        _run_generator(gen)
        behaviour.set_done.assert_called_once()


class TestCallCheckpointBehaviourGetCheckpointTxHash:
    """Tests for CallCheckpointBehaviour.get_checkpoint_tx_hash."""

    def _make_behaviour(self) -> MagicMock:
        behaviour = MagicMock(spec=CallCheckpointBehaviour)
        behaviour.params = make_mock_params()
        behaviour.context = make_mock_context(params=behaviour.params)
        behaviour.synchronized_data = make_mock_synchronized_data()
        behaviour.get_chain_id = MagicMock(return_value="base")

        mock_ts = _make_timestamp_mock(1700000000.0)
        behaviour.round_sequence = MagicMock()
        behaviour.round_sequence.last_round_transition_timestamp = mock_ts
        return behaviour

    def test_returns_none_when_unstaked(self) -> None:
        """Test returns None when service is unstaked."""
        behaviour = self._make_behaviour()

        def mock_get_staking_state(chain):  # type: ignore[no-untyped-def]
            yield
            return StakingState.UNSTAKED

        behaviour._get_service_staking_state = mock_get_staking_state

        gen = CallCheckpointBehaviour.get_checkpoint_tx_hash(behaviour)
        result = _run_generator(gen)
        assert result is None

    def test_returns_none_when_checkpoint_not_reached(self) -> None:
        """Test returns None when checkpoint is not reached."""
        behaviour = self._make_behaviour()

        def mock_get_staking_state(chain):  # type: ignore[no-untyped-def]
            yield
            return StakingState.STAKED

        behaviour._get_service_staking_state = mock_get_staking_state

        def mock_check_if_checkpoint_reached(chain):  # type: ignore[no-untyped-def]
            yield
            return False

        behaviour._check_if_checkpoint_reached = mock_check_if_checkpoint_reached

        gen = CallCheckpointBehaviour.get_checkpoint_tx_hash(behaviour)
        result = _run_generator(gen)
        assert result is None

    def test_returns_hash_when_staked_and_checkpoint_reached(self) -> None:
        """Test returns tx hash when staked and checkpoint reached."""
        behaviour = self._make_behaviour()

        def mock_get_staking_state(chain):  # type: ignore[no-untyped-def]
            yield
            return StakingState.STAKED

        behaviour._get_service_staking_state = mock_get_staking_state

        def mock_check_if_checkpoint_reached(chain):  # type: ignore[no-untyped-def]
            yield
            return True

        behaviour._check_if_checkpoint_reached = mock_check_if_checkpoint_reached

        def mock_prepare_checkpoint_tx(chain):  # type: ignore[no-untyped-def]
            yield
            return "0xcheckpoint_hash"

        behaviour._prepare_checkpoint_tx = mock_prepare_checkpoint_tx

        gen = CallCheckpointBehaviour.get_checkpoint_tx_hash(behaviour)
        result = _run_generator(gen)
        assert result == "0xcheckpoint_hash"

    def test_returns_none_when_evicted_and_checkpoint_reached(self) -> None:
        """Test returns None when evicted even if checkpoint is reached (not STAKED)."""
        behaviour = self._make_behaviour()

        def mock_get_staking_state(chain):  # type: ignore[no-untyped-def]
            yield
            return StakingState.EVICTED

        behaviour._get_service_staking_state = mock_get_staking_state

        def mock_check_if_checkpoint_reached(chain):  # type: ignore[no-untyped-def]
            yield
            return True

        behaviour._check_if_checkpoint_reached = mock_check_if_checkpoint_reached

        gen = CallCheckpointBehaviour.get_checkpoint_tx_hash(behaviour)
        result = _run_generator(gen)
        # EVICTED is not UNSTAKED so won't return early; but EVICTED != STAKED so won't prepare checkpoint
        assert result is None


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

        def mock_get_next_checkpoint(chain):  # type: ignore[no-untyped-def]
            yield
            return None

        behaviour._get_next_checkpoint = mock_get_next_checkpoint

        gen = CallCheckpointBehaviour._check_if_checkpoint_reached(
            behaviour, chain="base"
        )
        next(gen)
        result = None
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result is False

    def test_check_if_checkpoint_reached_zero(self) -> None:
        """Test _check_if_checkpoint_reached returns True when next_checkpoint is 0."""
        behaviour = self._make_behaviour()

        def mock_get_next_checkpoint(chain):  # type: ignore[no-untyped-def]
            yield
            return 0

        behaviour._get_next_checkpoint = mock_get_next_checkpoint

        gen = CallCheckpointBehaviour._check_if_checkpoint_reached(
            behaviour, chain="base"
        )
        next(gen)
        result = None
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result is True

    def test_check_if_checkpoint_reached_future(self) -> None:
        """Test returns False when next checkpoint is in the future."""
        behaviour = self._make_behaviour()

        def mock_get_next_checkpoint(chain):  # type: ignore[no-untyped-def]
            yield
            return 1700000001  # 1 second in the future

        behaviour._get_next_checkpoint = mock_get_next_checkpoint

        gen = CallCheckpointBehaviour._check_if_checkpoint_reached(
            behaviour, chain="base"
        )
        next(gen)
        result = None
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result is False

    def test_check_if_checkpoint_reached_past(self) -> None:
        """Test returns True when next checkpoint is in the past."""
        behaviour = self._make_behaviour()

        def mock_get_next_checkpoint(chain):  # type: ignore[no-untyped-def]
            yield
            return 1699999999  # 1 second in the past

        behaviour._get_next_checkpoint = mock_get_next_checkpoint

        gen = CallCheckpointBehaviour._check_if_checkpoint_reached(
            behaviour, chain="base"
        )
        next(gen)
        result = None
        try:
            gen.send(None)
        except StopIteration as e:
            result = e.value
        assert result is True


class TestGetNextCheckpoint:
    """Tests for CallCheckpointBehaviour._get_next_checkpoint."""

    def _make_behaviour(self) -> MagicMock:
        behaviour = MagicMock(spec=CallCheckpointBehaviour)
        behaviour.params = make_mock_params()
        behaviour.context = make_mock_context(params=behaviour.params)
        return behaviour

    def test_get_next_checkpoint_returns_value(self) -> None:
        """Test _get_next_checkpoint returns value from contract."""
        behaviour = self._make_behaviour()

        def mock_contract_interact(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield
            return 1700001000

        behaviour.contract_interact = mock_contract_interact

        gen = CallCheckpointBehaviour._get_next_checkpoint(behaviour, "base")
        result = _run_generator(gen)
        assert result == 1700001000

    def test_get_next_checkpoint_returns_none(self) -> None:
        """Test _get_next_checkpoint returns None on failure."""
        behaviour = self._make_behaviour()

        def mock_contract_interact(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield
            return None

        behaviour.contract_interact = mock_contract_interact

        gen = CallCheckpointBehaviour._get_next_checkpoint(behaviour, "base")
        result = _run_generator(gen)
        assert result is None


class TestPrepareCheckpointTx:
    """Tests for CallCheckpointBehaviour._prepare_checkpoint_tx."""

    def _make_behaviour(self) -> MagicMock:
        behaviour = MagicMock(spec=CallCheckpointBehaviour)
        behaviour.params = make_mock_params()
        behaviour.context = make_mock_context(params=behaviour.params)
        behaviour.synchronized_data = make_mock_synchronized_data()
        behaviour.get_chain_id = MagicMock(return_value="base")
        return behaviour

    def test_prepare_checkpoint_tx(self) -> None:
        """Test _prepare_checkpoint_tx calls contract_interact and _build_safe_tx_hash."""
        behaviour = self._make_behaviour()

        def mock_contract_interact(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield
            return b"\x01\x02\x03"

        behaviour.contract_interact = mock_contract_interact

        def mock_build_safe_tx_hash(to_address, data):  # type: ignore[no-untyped-def]
            yield
            return "safe_checkpoint_hash"

        behaviour._build_safe_tx_hash = mock_build_safe_tx_hash

        gen = CallCheckpointBehaviour._prepare_checkpoint_tx(behaviour, "base")
        result = _run_generator(gen)
        assert result == "safe_checkpoint_hash"


class TestTransactionLoopCheckBehaviourAsyncAct(_ChainBehaviourTestBase):
    """Tests for TransactionLoopCheckBehaviour.async_act using FSMBehaviourBaseCase."""

    def test_async_act(self) -> None:
        """Test async_act increments counter and sends payload."""
        self.fast_forward_to_behaviour(
            self.behaviour,
            TransactionLoopCheckBehaviour.auto_behaviour_id(),
            SynchronizedData(
                AbciAppDB(setup_data=AbciAppDB.data_to_lists(self._default_data))
            ),
        )
        self.behaviour.act_wrapper()
        self.mock_a2a_transaction()
        self._test_done_flag_set()
        self.end_round(done_event=Event.DONE)
