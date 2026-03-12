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

"""Tests for packages.dvilela.skills.memeooorr_abci.behaviour_classes.base"""

# pylint: disable=unsupported-membership-test,unsubscriptable-object

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from packages.dvilela.protocols.kv_store.message import KvStoreMessage
from packages.dvilela.skills.memeooorr_abci.behaviour_classes.base import (
    AttributeDefinitionParams,
    HTTP_OK,
    LIST_COUNT_TO_KEEP,
    MemeooorrBaseBehaviour,
    is_tweet_valid,
)
from packages.valory.protocols.contract_api.message import ContractApiMessage
from packages.valory.protocols.ledger_api.message import LedgerApiMessage
from packages.valory.protocols.srr.message import SrrMessage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _exhaust(gen: Generator[Any, None, Any]) -> Any:
    """Run a generator to completion and return its result."""
    try:
        next(gen)
        while True:
            gen.send(None)
    except StopIteration as exc:
        return exc.value


def _make_behaviour(**overrides: Any) -> MagicMock:
    """Create a MagicMock with the right attributes for MemeooorrBaseBehaviour.

    We do NOT use spec= so that when unbound real methods call self.xxx(),
    the mock auto-creates attributes without AttributeError.
    We bind the real methods we want to test explicitly.

    :param overrides: keyword arguments to override default attributes.
    :return: a MagicMock configured for MemeooorrBaseBehaviour tests.
    """
    b = MagicMock()

    # Context
    b.context = MagicMock()
    b.context.logger = MagicMock()
    b.context.outbox = MagicMock()
    b.context.agent_address = "0xagent"

    # State
    b.context.state = MagicMock()
    b.context.state.twitter_username = None
    b.context.state.twitter_id = None
    b.context.state.twitter_display_name = None
    b.context.state.env_var_status = {"needs_update": False, "env_vars": {}}

    # Params
    b.params = MagicMock()
    b.params.home_chain_id = "base"
    b.params.persona = "test_persona"
    b.params.heart_cooldown_hours = 24
    b.params.summon_cooldown_seconds = 3600
    b.params.meme_factory_address_base = "0xfactory_base"
    b.params.meme_factory_address_celo = "0xfactory_celo"
    b.params.meme_factory_deployment_block_base = 100
    b.params.meme_factory_deployment_block_celo = 200
    b.params.olas_token_address_base = "0xolas_base"
    b.params.olas_token_address_celo = "0xolas_celo"
    b.params.service_registry_address_base = "0xsr_base"
    b.params.service_registry_address_celo = "0xsr_celo"
    b.params.olas_subgraph_url = "https://subgraph.example.com"
    b.params.meme_subgraph_url = "https://meme-subgraph.example.com"

    # Alternative model
    alt = MagicMock()
    alt.use = False
    b.params.alternative_model_for_tweets = alt

    # Synchronized data
    b.synchronized_data = MagicMock()
    b.synchronized_data.safe_contract_address = "0xsafe"
    b.synchronized_data.meme_coins = None

    # shared_state
    b.shared_state = MagicMock()

    for k, v in overrides.items():
        setattr(b, k, v)

    return b


def _bind(method: Any) -> Any:
    """Return a helper that calls an unbound method on a mock."""

    def wrapper(b: Any, *args: Any, **kwargs: Any) -> Any:
        """Test wrapper."""
        return method(b, *args, **kwargs)

    return wrapper


def _fake_do_conn(response: Any) -> Any:
    """Return a do_connection_request side_effect that yields once and returns response."""

    def fn(msg: Any, dlg: Any, timeout: Any = None) -> Generator[Any, None, Any]:
        """Test fn."""
        yield
        return response

    return fn


def _fake_read_kv(return_value: Any) -> Any:
    """Return a _read_kv side_effect."""

    def fn(keys: Any) -> Generator[Any, None, Any]:
        """Test fn."""
        yield
        return return_value

    return fn


def _fake_write_kv(return_value: Any = True) -> Any:
    """Return a _write_kv side_effect."""

    def fn(data: Any) -> Generator[Any, None, Any]:
        """Test fn."""
        yield
        return return_value

    return fn


def _fake_read_json_from_kv(return_value: Any) -> Any:
    """Return a _read_json_from_kv side_effect."""

    def fn(key: Any, default_value: Any) -> Generator[Any, None, Any]:
        """Test fn."""
        yield
        return return_value

    return fn


def _fake_read_value_from_kv(return_value: Any) -> Any:
    """Return a _read_value_from_kv side_effect."""

    def fn(key: Any, default_value: Any) -> Generator[Any, None, Any]:
        """Test fn."""
        yield
        return return_value

    return fn


def _setup_sync_timestamp(b: Any, ts: float = 1700000000.0) -> None:
    """Make get_sync_timestamp return a known value."""
    ts_mock = MagicMock()
    ts_mock.timestamp.return_value = ts
    b.context.state.round_sequence.last_round_transition_timestamp = ts_mock
    # Also bind the real get_sync_timestamp so chained calls work
    b.get_sync_timestamp = lambda: MemeooorrBaseBehaviour.get_sync_timestamp(b)


# ---------------------------------------------------------------------------
# is_tweet_valid tests
# ---------------------------------------------------------------------------


class TestIsTweetValid:
    """Tests for the is_tweet_valid function."""

    def test_short_tweet_valid(self) -> None:
        """Test test_short_tweet_valid."""
        assert is_tweet_valid("Hello world") is True

    def test_long_tweet_invalid(self) -> None:
        """Test test_long_tweet_invalid."""
        assert is_tweet_valid("a" * 500) is False

    def test_empty_tweet_valid(self) -> None:
        """Test test_empty_tweet_valid."""
        assert is_tweet_valid("") is True


# ---------------------------------------------------------------------------
# AttributeDefinitionParams tests
# ---------------------------------------------------------------------------


class TestAttributeDefinitionParams:  # pylint: disable=too-few-public-methods
    """Test TestAttributeDefinitionParams."""

    def test_creation(self) -> None:
        """Test test_creation."""
        p = AttributeDefinitionParams("n", 1, 2, "str", True, "d")
        assert p.attr_def_name == "n"
        assert p.agent_type_id == 1


# ---------------------------------------------------------------------------
# Properties (lines 141, 146, 151) tests
# ---------------------------------------------------------------------------


def _make_concrete_instance() -> Any:
    """Create a concrete (non-abstract) instance that inherits from MemeooorrBaseBehaviour."""
    from packages.valory.skills.agent_db_abci.behaviours import (  # pylint: disable=import-outside-toplevel
        AgentDBBehaviour,
    )

    class _TestBehaviour(MemeooorrBaseBehaviour):
        """Test _TestBehaviour."""

        matching_round = MagicMock()

        def async_act(self) -> Generator:  # type: ignore[override]
            """Test async_act."""
            yield

    # Use object.__new__ to skip __init__ (which needs full framework setup)
    return _TestBehaviour, AgentDBBehaviour


class TestProperties:
    """Test TestProperties."""

    def test_synchronized_data(self) -> None:
        """Test that synchronized_data property delegates to super and casts."""
        TestCls, ParentCls = _make_concrete_instance()
        mock_sd = MagicMock()
        with patch.object(
            ParentCls,
            "synchronized_data",
            new_callable=PropertyMock,
            return_value=mock_sd,
        ):
            inst = object.__new__(TestCls)
            result = inst.synchronized_data
            assert result is mock_sd

    def test_params(self) -> None:
        """Test that params property delegates to super and casts."""
        TestCls, ParentCls = _make_concrete_instance()
        mock_p = MagicMock()
        with patch.object(
            ParentCls, "params", new_callable=PropertyMock, return_value=mock_p
        ):
            inst = object.__new__(TestCls)
            result = inst.params
            assert result is mock_p

    def test_local_state(self) -> None:
        """Test test_local_state."""
        b = _make_behaviour()
        result: Any = MemeooorrBaseBehaviour.local_state.fget(b)  # type: ignore[attr-defined]
        assert result is not None


# ---------------------------------------------------------------------------
# _do_connection_request / do_connection_request (lines 161-167, 184) tests
# ---------------------------------------------------------------------------


class TestDoConnectionRequest:
    """Test TestDoConnectionRequest."""

    def test_do_connection_request(self) -> None:
        """Test test_do_connection_request."""
        b = _make_behaviour()
        msg, dlg = MagicMock(), MagicMock()
        mock_resp = MagicMock()
        b._get_request_nonce_from_dialogue = MagicMock(return_value="n1")
        b.get_callback_request = MagicMock(return_value="cb")
        b.context.requests = MagicMock()
        b.context.requests.request_id_to_callback = {}

        def fake_wait(timeout: Any = None) -> Generator[Any, None, Any]:
            """Test fake_wait."""
            yield
            return mock_resp

        b.wait_for_message = MagicMock(side_effect=fake_wait)

        gen = MemeooorrBaseBehaviour._do_connection_request(b, msg, dlg, timeout=10.0)
        result = _exhaust(gen)
        b.context.outbox.put_message.assert_called_once_with(message=msg)
        assert result == mock_resp

    def test_do_connection_request_public(self) -> None:
        """Test test_do_connection_request_public."""
        b = _make_behaviour()
        mock_resp = MagicMock()

        def fake_inner(m: Any, d: Any, t: Any = None) -> Generator[Any, None, Any]:
            """Test fake_inner."""
            yield
            return mock_resp

        b._do_connection_request = MagicMock(side_effect=fake_inner)
        gen = MemeooorrBaseBehaviour.do_connection_request(b, MagicMock(), MagicMock())
        assert _exhaust(gen) == mock_resp


# ---------------------------------------------------------------------------
# _call_tweepy (lines 193-247) tests
# ---------------------------------------------------------------------------


class TestCallTweepy:
    """Test TestCallTweepy."""

    def _run(self, response_msg: Any, **tweepy_kwargs: Any) -> Any:
        """Test _run."""
        b = _make_behaviour()
        b.do_connection_request = MagicMock(side_effect=_fake_do_conn(response_msg))
        b.context.srr_dialogues = MagicMock()
        b.context.srr_dialogues.create = MagicMock(
            return_value=(MagicMock(), MagicMock())
        )
        gen = MemeooorrBaseBehaviour._call_tweepy(b, method="get_me", **tweepy_kwargs)
        return b, _exhaust(gen)

    def test_success(self) -> None:
        """Test test_success."""
        resp = MagicMock(performative=SrrMessage.Performative.RESPONSE)
        resp.payload = json.dumps({"response": {"user_id": "1"}})
        _, result = self._run(resp)
        assert result == {"user_id": "1"}

    def test_unexpected_performative(self) -> None:
        """Test test_unexpected_performative."""
        resp = MagicMock(performative=SrrMessage.Performative.REQUEST)
        _, result = self._run(resp)
        assert result is None

    def test_error_forbidden(self) -> None:
        """Test test_error_forbidden."""
        resp = MagicMock(performative=SrrMessage.Performative.RESPONSE)
        resp.payload = json.dumps({"response": {"error": "Forbidden 403"}})
        b, _result = self._run(resp)
        assert b.context.state.env_var_status["needs_update"] is True

    def test_error_403_in_string(self) -> None:
        """Test test_error_403_in_string."""
        resp = MagicMock(performative=SrrMessage.Performative.RESPONSE)
        resp.payload = json.dumps({"response": {"error": "Got a 403 response"}})
        b, _ = self._run(resp)
        assert b.context.state.env_var_status["needs_update"] is True

    def test_error_credentials(self) -> None:
        """Test test_error_credentials."""
        resp = MagicMock(performative=SrrMessage.Performative.RESPONSE)
        resp.payload = json.dumps(
            {"response": {"error": "Invalid credentials provided"}}
        )
        b, _ = self._run(resp)
        assert b.context.state.env_var_status["needs_update"] is True

    def test_error_generic_no_update(self) -> None:
        """Test test_error_generic_no_update."""
        resp = MagicMock(performative=SrrMessage.Performative.RESPONSE)
        resp.payload = json.dumps({"response": {"error": "Unknown error"}})
        b, _ = self._run(resp)
        assert b.context.state.env_var_status["needs_update"] is False


# ---------------------------------------------------------------------------
# _call_genai (lines 257-281) tests
# ---------------------------------------------------------------------------


class TestCallGenai:
    """Test TestCallGenai."""

    def _run(
        self, response_msg: Any, schema: Any = None, temperature: Any = None
    ) -> Any:
        """Test _run."""
        b = _make_behaviour()
        b.do_connection_request = MagicMock(side_effect=_fake_do_conn(response_msg))
        b.context.srr_dialogues = MagicMock()
        b.context.srr_dialogues.create = MagicMock(
            return_value=(MagicMock(), MagicMock())
        )
        gen = MemeooorrBaseBehaviour._call_genai(
            b, "prompt", schema=schema, temperature=temperature
        )
        return b, _exhaust(gen)

    def test_success(self) -> None:
        """Test test_success."""
        resp = MagicMock()
        resp.payload = json.dumps({"response": "hello"})
        _, result = self._run(resp)
        assert result == "hello"

    def test_with_schema_and_temperature(self) -> None:
        """Test test_with_schema_and_temperature."""
        resp = MagicMock()
        resp.payload = json.dumps({"response": "ok"})
        _, result = self._run(resp, schema={"type": "object"}, temperature=0.5)
        assert result == "ok"

    def test_error(self) -> None:
        """Test test_error."""
        resp = MagicMock()
        resp.payload = json.dumps({"error": "overloaded"})
        _, result = self._run(resp)
        assert result is None


# ---------------------------------------------------------------------------
# _read_kv / _write_kv / read_kv / write_kv (lines 288-357) tests
# ---------------------------------------------------------------------------


class TestKvStore:
    """Test TestKvStore."""

    def _setup(self, b: Any, response_msg: Any) -> None:
        """Test _setup."""
        b.do_connection_request = MagicMock(side_effect=_fake_do_conn(response_msg))
        b.context.kv_store_dialogues = MagicMock()
        b.context.kv_store_dialogues.create = MagicMock(
            return_value=(MagicMock(), MagicMock())
        )

    def test_read_kv_success(self) -> None:
        """Test test_read_kv_success."""
        b = _make_behaviour()
        resp = MagicMock(performative=KvStoreMessage.Performative.READ_RESPONSE)
        resp.data = {"k1": "v1"}
        self._setup(b, resp)
        assert _exhaust(MemeooorrBaseBehaviour._read_kv(b, keys=("k1",))) == {
            "k1": "v1"
        }

    def test_read_kv_wrong_performative(self) -> None:
        """Test test_read_kv_wrong_performative."""
        b = _make_behaviour()
        resp = MagicMock(performative=KvStoreMessage.Performative.ERROR)
        self._setup(b, resp)
        assert _exhaust(MemeooorrBaseBehaviour._read_kv(b, keys=("k1",))) is None

    def test_write_kv_success(self) -> None:
        """Test test_write_kv_success."""
        b = _make_behaviour()
        resp = MagicMock(performative=KvStoreMessage.Performative.SUCCESS)
        self._setup(b, resp)
        assert _exhaust(MemeooorrBaseBehaviour._write_kv(b, data={"k": "v"})) is True

    def test_write_kv_failure(self) -> None:
        """Test test_write_kv_failure."""
        b = _make_behaviour()
        resp = MagicMock(performative=KvStoreMessage.Performative.ERROR)
        self._setup(b, resp)
        assert _exhaust(MemeooorrBaseBehaviour._write_kv(b, data={"k": "v"})) is False

    def test_write_kv_none_response(self) -> None:
        """Test test_write_kv_none_response."""
        b = _make_behaviour()
        b.do_connection_request = MagicMock(side_effect=_fake_do_conn(None))
        b.context.kv_store_dialogues = MagicMock()
        b.context.kv_store_dialogues.create = MagicMock(
            return_value=(MagicMock(), MagicMock())
        )
        assert _exhaust(MemeooorrBaseBehaviour._write_kv(b, data={"k": "v"})) is False

    def test_read_kv_wrapper(self) -> None:
        """Test test_read_kv_wrapper."""
        b = _make_behaviour()
        b._read_kv = MagicMock(side_effect=_fake_read_kv({"k": "v"}))
        assert _exhaust(MemeooorrBaseBehaviour.read_kv(b, keys=("k",))) == {"k": "v"}

    def test_write_kv_wrapper(self) -> None:
        """Test test_write_kv_wrapper."""
        b = _make_behaviour()
        b._write_kv = MagicMock(side_effect=_fake_write_kv(True))
        assert _exhaust(MemeooorrBaseBehaviour.write_kv(b, data={"k": "v"})) is True


# ---------------------------------------------------------------------------
# Time helpers (lines 360-373, covers get_sync_*) tests
# ---------------------------------------------------------------------------


class TestTimeHelpers:
    """Test TestTimeHelpers."""

    def test_get_sync_timestamp(self) -> None:
        """Test test_get_sync_timestamp."""
        b = _make_behaviour()
        _setup_sync_timestamp(b, 1700000000.0)
        assert MemeooorrBaseBehaviour.get_sync_timestamp(b) == 1700000000.0

    def test_get_sync_datetime(self) -> None:
        """Test test_get_sync_datetime."""
        b = _make_behaviour()
        _setup_sync_timestamp(b, 1700000000.0)
        b.get_sync_datetime = lambda: MemeooorrBaseBehaviour.get_sync_datetime(b)
        result = MemeooorrBaseBehaviour.get_sync_datetime(b)
        assert result == datetime.fromtimestamp(1700000000.0)

    def test_get_sync_time_str(self) -> None:
        """Test test_get_sync_time_str."""
        b = _make_behaviour()
        _setup_sync_timestamp(b, 1700000000.0)
        b.get_sync_datetime = lambda: MemeooorrBaseBehaviour.get_sync_datetime(b)
        result = MemeooorrBaseBehaviour.get_sync_time_str(b)
        assert result == datetime.fromtimestamp(1700000000.0).strftime(
            "%Y-%m-%d %H:%M:%S"
        )


# ---------------------------------------------------------------------------
# _get_configurable_param (lines 392-425) tests
# ---------------------------------------------------------------------------


class TestGetConfigurableParam:
    """Test TestGetConfigurableParam."""

    def _setup(self, b: Any, read_return: Any) -> None:
        """Test _setup."""
        b.read_kv = MagicMock(side_effect=_fake_read_kv(read_return))
        b.write_kv = MagicMock(side_effect=_fake_write_kv(True))

    def test_db_returns_none_fallback(self) -> None:
        """Test test_db_returns_none_fallback."""
        b = _make_behaviour()
        b.params.persona = "cfg"
        self._setup(b, None)
        result = _exhaust(
            MemeooorrBaseBehaviour._get_configurable_param(
                b, "persona", "initial_persona", str
            )
        )
        assert result == "cfg"

    def test_initial_db_none_writes_it(self) -> None:
        """Test test_initial_db_none_writes_it."""
        b = _make_behaviour()
        b.params.persona = "cfg"
        self._setup(b, {"initial_persona": None, "persona": "db_val"})
        result = _exhaust(
            MemeooorrBaseBehaviour._get_configurable_param(
                b, "persona", "initial_persona", str
            )
        )
        # initial_db=None -> write, initial_db=cfg. cfg!=None(cfg now)? No, cfg==cfg. Return db_val
        assert result == "db_val"

    def test_param_db_none_writes_it(self) -> None:
        """Test test_param_db_none_writes_it."""
        b = _make_behaviour()
        b.params.persona = "cfg"
        self._setup(b, {"initial_persona": "cfg", "persona": None})
        result = _exhaust(
            MemeooorrBaseBehaviour._get_configurable_param(
                b, "persona", "initial_persona", str
            )
        )
        assert result == "cfg"

    def test_reconfiguration(self) -> None:
        """Test test_reconfiguration."""
        b = _make_behaviour()
        b.params.persona = "new"
        self._setup(b, {"initial_persona": "old", "persona": "old"})
        result = _exhaust(
            MemeooorrBaseBehaviour._get_configurable_param(
                b, "persona", "initial_persona", str
            )
        )
        assert result == "new"

    def test_no_reconfiguration(self) -> None:
        """Test test_no_reconfiguration."""
        b = _make_behaviour()
        b.params.persona = "same"
        self._setup(b, {"initial_persona": "same", "persona": "same"})
        result = _exhaust(
            MemeooorrBaseBehaviour._get_configurable_param(
                b, "persona", "initial_persona", str
            )
        )
        assert result == "same"

    def test_int_type(self) -> None:
        """Test test_int_type."""
        b = _make_behaviour()
        b.params.heart_cooldown_hours = 48
        self._setup(
            b, {"initial_heart_cooldown_hours": "48", "heart_cooldown_hours": "48"}
        )
        result = _exhaust(
            MemeooorrBaseBehaviour._get_configurable_param(
                b, "heart_cooldown_hours", "initial_heart_cooldown_hours", int
            )
        )
        assert result == 48


# ---------------------------------------------------------------------------
# get_persona / get_heart_cooldown_hours / get_summon_cooldown_seconds (lines 430-457) tests
# ---------------------------------------------------------------------------


class TestConfigurableWrappers:
    """Test TestConfigurableWrappers."""

    def _fake_configurable(self, return_value: Any) -> Any:
        """Test _fake_configurable."""

        def fn(*args: Any, **kwargs: Any) -> Generator[Any, None, Any]:
            """Test fn."""
            yield
            return return_value

        return fn

    def test_get_persona(self) -> None:
        """Test test_get_persona."""
        b = _make_behaviour()
        b._get_configurable_param = MagicMock(
            side_effect=self._fake_configurable("my_persona")
        )
        result = _exhaust(MemeooorrBaseBehaviour.get_persona(b))
        assert result == "my_persona"
        b.shared_state.update_agent_behavior.assert_called_once_with("my_persona")

    def test_get_heart_cooldown_hours(self) -> None:
        """Test test_get_heart_cooldown_hours."""
        b = _make_behaviour()
        b._get_configurable_param = MagicMock(side_effect=self._fake_configurable(24))
        result = _exhaust(MemeooorrBaseBehaviour.get_heart_cooldown_hours(b))
        assert result == 24

    def test_get_summon_cooldown_seconds(self) -> None:
        """Test test_get_summon_cooldown_seconds."""
        b = _make_behaviour()
        b._get_configurable_param = MagicMock(side_effect=self._fake_configurable(3600))
        result = _exhaust(MemeooorrBaseBehaviour.get_summon_cooldown_seconds(b))
        assert result == 3600


# ---------------------------------------------------------------------------
# get_native_balance (lines 463-512) tests
# ---------------------------------------------------------------------------


class TestGetNativeBalance:
    """Test TestGetNativeBalance."""

    def _setup(self, b: Any, safe_resp: Any, agent_resp: Any) -> None:
        """Test _setup."""
        call_idx = [0]

        def fake_ledger(**kw: Any) -> Generator[Any, None, Any]:
            """Test fake_ledger."""
            call_idx[0] += 1
            yield
            return safe_resp if call_idx[0] == 1 else agent_resp

        b.get_ledger_api_response = MagicMock(side_effect=fake_ledger)
        b.get_chain_id = MagicMock(return_value="base")

    def _make_state_resp(self, balance: Any) -> MagicMock:
        """Test _make_state_resp."""
        r = MagicMock(performative=LedgerApiMessage.Performative.STATE)
        r.state.body = {"get_balance_result": balance}
        return r

    def _make_error_resp(self) -> MagicMock:
        """Test _make_error_resp."""
        return MagicMock(performative=LedgerApiMessage.Performative.ERROR)

    def test_both_ok(self) -> None:
        """Test test_both_ok."""
        b = _make_behaviour()
        self._setup(b, self._make_state_resp(1e18), self._make_state_resp(2e18))
        assert _exhaust(MemeooorrBaseBehaviour.get_native_balance(b)) == {
            "safe": 1.0,
            "agent": 2.0,
        }

    def test_safe_error(self) -> None:
        """Test test_safe_error."""
        b = _make_behaviour()
        self._setup(b, self._make_error_resp(), self._make_state_resp(2e18))
        assert _exhaust(MemeooorrBaseBehaviour.get_native_balance(b)) == {
            "safe": None,
            "agent": 2.0,
        }

    def test_agent_error(self) -> None:
        """Test test_agent_error."""
        b = _make_behaviour()
        self._setup(b, self._make_state_resp(1e18), self._make_error_resp())
        assert _exhaust(MemeooorrBaseBehaviour.get_native_balance(b)) == {
            "safe": 1.0,
            "agent": None,
        }


# ---------------------------------------------------------------------------
# get_meme_available_actions (lines 523-585) tests
# ---------------------------------------------------------------------------


class TestGetMemeAvailableActions:
    """Test TestGetMemeAvailableActions."""

    NOW_TS = 1700000000

    def _make_meme(self, **overrides: Any) -> dict:
        """Test _make_meme."""
        d: dict = {
            "summon_time": self.NOW_TS - 100000,
            "unleash_time": 0,
            "is_purged": False,
            "hearters": {},
            "token_nonce": 5,
        }
        d.update(overrides)
        return d

    def _setup(
        self,
        b: Any,
        collectable: int = 0,
        last_heart_ts: int = 0,
        heart_cooldown: int = 24,
    ) -> None:
        """Test _setup."""
        _setup_sync_timestamp(b, float(self.NOW_TS))

        def fake_collectable(nonce: Any) -> Generator[Any, None, Any]:
            """Test fake_collectable."""
            yield
            return collectable

        b.get_collectable_amount = MagicMock(side_effect=fake_collectable)

        b._read_json_from_kv = MagicMock(
            side_effect=_fake_read_json_from_kv(last_heart_ts)
        )

        def fake_hc() -> Generator[Any, None, Any]:
            """Test fake_hc."""
            yield
            return heart_cooldown

        b.get_heart_cooldown_hours = MagicMock(side_effect=fake_hc)

    def test_heart(self) -> None:
        """Test test_heart."""
        b = _make_behaviour()
        self._setup(b, last_heart_ts=0, heart_cooldown=0)
        result: Any = _exhaust(
            MemeooorrBaseBehaviour.get_meme_available_actions(
                b, self._make_meme(), 0, False
            )
        )
        assert "heart" in result

    def test_heart_nonce1_excluded(self) -> None:
        """Test test_heart_nonce1_excluded."""
        b = _make_behaviour()
        self._setup(b, last_heart_ts=0, heart_cooldown=0)
        result: Any = _exhaust(
            MemeooorrBaseBehaviour.get_meme_available_actions(
                b, self._make_meme(token_nonce=1), 0, False
            )
        )
        assert "heart" not in result

    def test_heart_cooldown_not_passed(self) -> None:
        """Test test_heart_cooldown_not_passed."""
        b = _make_behaviour()
        self._setup(b, last_heart_ts=self.NOW_TS - 100, heart_cooldown=24)
        result: Any = _exhaust(
            MemeooorrBaseBehaviour.get_meme_available_actions(
                b, self._make_meme(), 0, False
            )
        )
        assert "heart" not in result

    def test_unleash(self) -> None:
        """Test test_unleash."""
        b = _make_behaviour()
        self._setup(b, last_heart_ts=self.NOW_TS, heart_cooldown=24)
        result: Any = _exhaust(
            MemeooorrBaseBehaviour.get_meme_available_actions(
                b, self._make_meme(summon_time=self.NOW_TS - 100000), 0, False
            )
        )
        assert "unleash" in result

    def test_unleash_nonce1_excluded(self) -> None:
        """Test test_unleash_nonce1_excluded."""
        b = _make_behaviour()
        self._setup(b, last_heart_ts=self.NOW_TS, heart_cooldown=24)
        result: Any = _exhaust(
            MemeooorrBaseBehaviour.get_meme_available_actions(
                b,
                self._make_meme(summon_time=self.NOW_TS - 100000, token_nonce=1),
                0,
                False,
            )
        )
        assert "unleash" not in result

    def test_unleash_too_recent(self) -> None:
        """Test test_unleash_too_recent."""
        b = _make_behaviour()
        self._setup(b, last_heart_ts=self.NOW_TS, heart_cooldown=24)
        result: Any = _exhaust(
            MemeooorrBaseBehaviour.get_meme_available_actions(
                b, self._make_meme(summon_time=self.NOW_TS - 100), 0, False
            )
        )
        assert "unleash" not in result

    def test_collect(self) -> None:
        """Test test_collect."""
        b = _make_behaviour()
        self._setup(b, collectable=100, last_heart_ts=self.NOW_TS, heart_cooldown=24)
        meme = self._make_meme(
            unleash_time=self.NOW_TS - 1000, hearters={"0xsafe": True}
        )
        result: Any = _exhaust(
            MemeooorrBaseBehaviour.get_meme_available_actions(b, meme, 0, False)
        )
        assert "collect" in result

    def test_collect_not_hearted(self) -> None:
        """Test test_collect_not_hearted."""
        b = _make_behaviour()
        self._setup(b, collectable=100, last_heart_ts=self.NOW_TS, heart_cooldown=24)
        meme = self._make_meme(unleash_time=self.NOW_TS - 1000, hearters={})
        result: Any = _exhaust(
            MemeooorrBaseBehaviour.get_meme_available_actions(b, meme, 0, False)
        )
        assert "collect" not in result

    def test_purge(self) -> None:
        """Test test_purge."""
        b = _make_behaviour()
        self._setup(b, last_heart_ts=self.NOW_TS, heart_cooldown=24)
        meme = self._make_meme(unleash_time=self.NOW_TS - 100000, is_purged=False)
        result: Any = _exhaust(
            MemeooorrBaseBehaviour.get_meme_available_actions(b, meme, 0, False)
        )
        assert "purge" in result

    def test_purge_already_purged(self) -> None:
        """Test test_purge_already_purged."""
        b = _make_behaviour()
        self._setup(b, last_heart_ts=self.NOW_TS, heart_cooldown=24)
        meme = self._make_meme(unleash_time=self.NOW_TS - 100000, is_purged=True)
        result: Any = _exhaust(
            MemeooorrBaseBehaviour.get_meme_available_actions(b, meme, 0, False)
        )
        assert "purge" not in result

    def test_burn(self) -> None:
        """Test test_burn."""
        b = _make_behaviour()
        self._setup(b, last_heart_ts=self.NOW_TS, heart_cooldown=24)
        result: Any = _exhaust(
            MemeooorrBaseBehaviour.get_meme_available_actions(
                b, self._make_meme(), 100, True
            )
        )
        assert "burn" in result

    def test_no_actions(self) -> None:
        """Test test_no_actions."""
        b = _make_behaviour()
        self._setup(b, last_heart_ts=self.NOW_TS - 100, heart_cooldown=24)
        meme = self._make_meme(unleash_time=self.NOW_TS - 1000, hearters={})
        result: Any = _exhaust(
            MemeooorrBaseBehaviour.get_meme_available_actions(b, meme, 0, False)
        )
        assert not result


# ---------------------------------------------------------------------------
# get_chain_id / get_native_ticker (lines 587-605) tests
# ---------------------------------------------------------------------------


class TestChainHelpers:
    """Test TestChainHelpers."""

    @pytest.mark.parametrize(
        "chain,expected",
        [
            ("base", "base"),
            ("Base", "base"),
            ("celo", "celo"),
            ("Celo", "celo"),
            ("other", ""),
        ],
    )
    def test_get_chain_id(self, chain: str, expected: str) -> None:
        """Test test_get_chain_id."""
        b = _make_behaviour()
        b.params.home_chain_id = chain
        assert MemeooorrBaseBehaviour.get_chain_id(b) == expected

    @pytest.mark.parametrize(
        "chain,expected", [("base", "ETH"), ("celo", "CELO"), ("other", "")]
    )
    def test_get_native_ticker(self, chain: str, expected: str) -> None:
        """Test test_get_native_ticker."""
        b = _make_behaviour()
        b.params.home_chain_id = chain
        assert MemeooorrBaseBehaviour.get_native_ticker(b) == expected


# ---------------------------------------------------------------------------
# get_packages (lines 610-648) tests
# ---------------------------------------------------------------------------


class TestGetPackages:
    """Test TestGetPackages."""

    def _setup(self, b: Any, status_code: int, body_dict: Any) -> None:
        """Test _setup."""

        def fake_http(**kw: Any) -> Generator[Any, None, Any]:
            """Test fake_http."""
            r = MagicMock()
            r.status_code = status_code
            r.body = (
                json.dumps(body_dict).encode()
                if isinstance(body_dict, dict)
                else body_dict
            )
            yield
            return r

        b.get_http_response = MagicMock(side_effect=fake_http)

    def test_success(self) -> None:
        """Test test_success."""
        b = _make_behaviour()
        self._setup(b, HTTP_OK, {"data": {"units": []}})
        assert _exhaust(MemeooorrBaseBehaviour.get_packages(b, "service")) == {
            "units": []
        }

    def test_http_error(self) -> None:
        """Test test_http_error."""
        b = _make_behaviour()
        self._setup(b, 500, {})
        assert _exhaust(MemeooorrBaseBehaviour.get_packages(b, "service")) is None

    def test_no_data_key(self) -> None:
        """Test test_no_data_key."""
        b = _make_behaviour()
        self._setup(b, HTTP_OK, {"error": "x"})
        assert _exhaust(MemeooorrBaseBehaviour.get_packages(b, "service")) is None


# ---------------------------------------------------------------------------
# get_memeooorr_handles_from_subgraph (lines 652-672) tests
# ---------------------------------------------------------------------------


class TestGetMemeoorrHandlesFromSubgraph:
    """Test TestGetMemeoorrHandlesFromSubgraph."""

    def test_no_services(self) -> None:
        """Test test_no_services."""
        b = _make_behaviour()

        def fake(pt: Any) -> Generator[Any, None, None]:
            """Test fake."""
            yield

        b.get_packages = MagicMock(side_effect=fake)
        assert not _exhaust(
            MemeooorrBaseBehaviour.get_memeooorr_handles_from_subgraph(b)
        )

    def test_matching(self) -> None:
        """Test test_matching."""
        b = _make_behaviour()
        b.context.state.twitter_username = "mybot"

        def fake(pt: Any) -> Generator[Any, None, Any]:
            """Test fake."""
            yield
            return {
                "units": [
                    {"description": "Memeooorr @alice"},
                    {"description": "Memeooorr @mybot"},  # own
                    {"description": "Other service"},
                    {"description": "Memeooorr @bob"},
                ]
            }

        b.get_packages = MagicMock(side_effect=fake)
        assert _exhaust(
            MemeooorrBaseBehaviour.get_memeooorr_handles_from_subgraph(b)
        ) == ["alice", "bob"]


# ---------------------------------------------------------------------------
# address getters (lines 674-704) tests
# ---------------------------------------------------------------------------


class TestAddressGetters:
    """Test TestAddressGetters."""

    @pytest.mark.parametrize(
        "chain,expected", [("base", "0xsr_base"), ("celo", "0xsr_celo")]
    )
    def test_service_registry(self, chain: str, expected: str) -> None:
        """Test test_service_registry."""
        b = _make_behaviour()
        b.params.home_chain_id = chain
        # Bind get_chain_id so the real method works
        b.get_chain_id = lambda: MemeooorrBaseBehaviour.get_chain_id(b)
        assert MemeooorrBaseBehaviour.get_service_registry_address(b) == expected

    @pytest.mark.parametrize(
        "chain,expected", [("base", "0xolas_base"), ("celo", "0xolas_celo")]
    )
    def test_olas(self, chain: str, expected: str) -> None:
        """Test test_olas."""
        b = _make_behaviour()
        b.params.home_chain_id = chain
        b.get_chain_id = lambda: MemeooorrBaseBehaviour.get_chain_id(b)
        assert MemeooorrBaseBehaviour.get_olas_address(b) == expected

    @pytest.mark.parametrize(
        "chain,expected", [("base", "0xfactory_base"), ("celo", "0xfactory_celo")]
    )
    def test_meme_factory(self, chain: str, expected: str) -> None:
        """Test test_meme_factory."""
        b = _make_behaviour()
        b.params.home_chain_id = chain
        b.get_chain_id = lambda: MemeooorrBaseBehaviour.get_chain_id(b)
        assert MemeooorrBaseBehaviour.get_meme_factory_address(b) == expected

    @pytest.mark.parametrize("chain,expected", [("base", 100), ("celo", 200)])
    def test_deployment_block(self, chain: str, expected: int) -> None:
        """Test test_deployment_block."""
        b = _make_behaviour()
        b.params.home_chain_id = chain
        b.get_chain_id = lambda: MemeooorrBaseBehaviour.get_chain_id(b)
        assert MemeooorrBaseBehaviour.get_meme_factory_deployment_block(b) == expected


# ---------------------------------------------------------------------------
# get_memeooorr_handles_from_chain (lines 709-755) tests
# ---------------------------------------------------------------------------


class TestGetMemeoorrHandlesFromChain:
    """Test TestGetMemeoorrHandlesFromChain."""

    def test_contract_error(self) -> None:
        """Test test_contract_error."""
        b = _make_behaviour()
        b.get_chain_id = MagicMock(return_value="base")
        b.get_service_registry_address = MagicMock(return_value="0xsr")

        def fake(**kw: Any) -> Generator[Any, None, Any]:
            """Test fake."""
            r = MagicMock(performative=ContractApiMessage.Performative.ERROR)
            yield
            return r

        b.get_contract_api_response = MagicMock(side_effect=fake)
        assert not _exhaust(MemeooorrBaseBehaviour.get_memeooorr_handles_from_chain(b))

    def test_success(self) -> None:
        """Test test_success."""
        b = _make_behaviour()
        b.get_chain_id = MagicMock(return_value="base")
        b.get_service_registry_address = MagicMock(return_value="0xsr")
        b.context.state.twitter_username = "mybot"

        def fake_contract(**kw: Any) -> Generator[Any, None, Any]:
            """Test fake_contract."""
            r = MagicMock(performative=ContractApiMessage.Performative.STATE)
            r.state.body = {
                "services_data": [
                    {"ipfs_hash": "h1"},
                    {"ipfs_hash": "h2"},
                    {"ipfs_hash": "h3"},
                    {"ipfs_hash": "h4"},
                ]
            }
            yield
            return r

        b.get_contract_api_response = MagicMock(side_effect=fake_contract)

        http_data = [
            (HTTP_OK, json.dumps({"description": "Memeooorr @alice"}).encode()),
            (500, b"err"),
            (HTTP_OK, json.dumps({"description": "Not memeooorr"}).encode()),
            (HTTP_OK, json.dumps({"description": "Memeooorr @mybot"}).encode()),
        ]
        idx = [0]

        def fake_http(**kw: Any) -> Generator[Any, None, Any]:
            """Test fake_http."""
            i = idx[0]
            idx[0] += 1
            r = MagicMock()
            r.status_code = http_data[i][0]
            r.body = http_data[i][1]
            yield
            return r

        b.get_http_response = MagicMock(side_effect=fake_http)
        assert _exhaust(MemeooorrBaseBehaviour.get_memeooorr_handles_from_chain(b)) == [
            "alice"
        ]


# ---------------------------------------------------------------------------
# get_meme_coins / get_meme_coins_from_subgraph (lines 760-833) tests
# ---------------------------------------------------------------------------


class TestGetMemeCoins:
    """Test TestGetMemeCoins."""

    def test_cached(self) -> None:
        """Test test_cached."""
        b = _make_behaviour()
        b.synchronized_data.meme_coins = [{"name": "x"}]
        # get_meme_coins is not a generator when cached; it just returns
        gen = MemeooorrBaseBehaviour.get_meme_coins(b)
        assert _exhaust(gen) == [{"name": "x"}]

    def test_not_cached(self) -> None:
        """Test test_not_cached."""
        b = _make_behaviour()
        b.synchronized_data.meme_coins = None

        def fake() -> Generator[Any, None, Any]:
            """Test fake."""
            yield
            return [{"name": "fetched"}]

        b.get_meme_coins_from_subgraph = MagicMock(side_effect=fake)
        assert _exhaust(MemeooorrBaseBehaviour.get_meme_coins(b)) == [
            {"name": "fetched"}
        ]

    def test_subgraph_http_error(self) -> None:
        """Test test_subgraph_http_error."""
        b = _make_behaviour()
        b.get_chain_id = MagicMock(return_value="base")

        def fake_http(**kw: Any) -> Generator[Any, None, Any]:
            """Test fake_http."""
            r = MagicMock()
            r.status_code = 500
            yield
            return r

        b.get_http_response = MagicMock(side_effect=fake_http)
        assert not _exhaust(MemeooorrBaseBehaviour.get_meme_coins_from_subgraph(b))

    def test_subgraph_success(self) -> None:
        """Test test_subgraph_success."""
        b = _make_behaviour()
        b.get_chain_id = MagicMock(return_value="base")

        items = [
            {
                "name": "T1",
                "symbol": "T",
                "blockNumber": "1",
                "chain": "base",
                "memeToken": "0x1",
                "liquidity": "10",
                "heartCount": "1",
                "isUnleashed": False,
                "isPurged": False,
                "lpPairAddress": "0xlp",
                "owner": "0xo",
                "timestamp": "1",
                "memeNonce": "2",
                "summonTime": "100",
                "unleashTime": "0",
                "hearters": {},
            },
            # nonce 0 - filtered
            {
                "name": "F",
                "symbol": "F",
                "blockNumber": "2",
                "chain": "base",
                "memeToken": "0x2",
                "liquidity": "0",
                "heartCount": "0",
                "isUnleashed": False,
                "isPurged": False,
                "lpPairAddress": "0x",
                "owner": "0x",
                "timestamp": "2",
                "memeNonce": "0",
                "summonTime": "101",
                "unleashTime": "0",
                "hearters": {},
            },
            # wrong chain - filtered
            {
                "name": "C",
                "symbol": "C",
                "blockNumber": "3",
                "chain": "celo",
                "memeToken": "0x3",
                "liquidity": "0",
                "heartCount": "0",
                "isUnleashed": False,
                "isPurged": False,
                "lpPairAddress": "0x",
                "owner": "0x",
                "timestamp": "3",
                "memeNonce": "3",
                "summonTime": "102",
                "unleashTime": "0",
                "hearters": {},
            },
        ]

        def fake_http(**kw: Any) -> Generator[Any, None, Any]:
            """Test fake_http."""
            r = MagicMock()
            r.status_code = HTTP_OK
            r.body = json.dumps({"data": {"memeTokens": {"items": items}}})
            yield
            return r

        b.get_http_response = MagicMock(side_effect=fake_http)

        def fake_burn() -> Generator[Any, None, Any]:
            """Test fake_burn."""
            yield
            return 0

        b.get_burnable_amount = MagicMock(side_effect=fake_burn)

        def fake_actions(md: Any, ba: Any, ml: Any) -> Generator[Any, None, Any]:
            """Test fake_actions."""
            yield
            return []

        b.get_meme_available_actions = MagicMock(side_effect=fake_actions)

        result: Any = _exhaust(MemeooorrBaseBehaviour.get_meme_coins_from_subgraph(b))
        assert len(result) == 1
        assert result[0]["token_name"] == "T1"

    def test_subgraph_maga_launched(self) -> None:
        """Test test_subgraph_maga_launched."""
        b = _make_behaviour()
        b.get_chain_id = MagicMock(return_value="base")

        items = [
            {
                "name": "AG3NT",
                "symbol": "A",
                "blockNumber": "1",
                "chain": "base",
                "memeToken": "0x1",
                "liquidity": "10",
                "heartCount": "1",
                "isUnleashed": True,
                "isPurged": False,
                "lpPairAddress": "0xlp",
                "owner": "0xo",
                "timestamp": "1",
                "memeNonce": "1",
                "summonTime": "100",
                "unleashTime": "200",
                "hearters": {},
            },
        ]

        def fake_http(**kw: Any) -> Generator[Any, None, Any]:
            """Test fake_http."""
            r = MagicMock()
            r.status_code = HTTP_OK
            r.body = json.dumps({"data": {"memeTokens": {"items": items}}})
            yield
            return r

        b.get_http_response = MagicMock(side_effect=fake_http)

        def fake_burn() -> Generator[Any, None, Any]:
            """Test fake_burn."""
            yield
            return 100

        b.get_burnable_amount = MagicMock(side_effect=fake_burn)

        actions_calls: list = []

        def fake_actions(md: Any, ba: Any, ml: Any) -> Generator[Any, None, Any]:
            """Test fake_actions."""
            actions_calls.append(ml)
            yield
            return ["burn"] if ml else []

        b.get_meme_available_actions = MagicMock(side_effect=fake_actions)

        _exhaust(MemeooorrBaseBehaviour.get_meme_coins_from_subgraph(b))
        assert actions_calls == [True]  # maga_launched should be True


# ---------------------------------------------------------------------------
# get_min_deploy_value (lines 835-844) tests
# ---------------------------------------------------------------------------


class TestGetMinDeployValue:  # pylint: disable=too-few-public-methods
    """Test TestGetMinDeployValue."""

    @pytest.mark.parametrize(
        "chain,expected", [("base", int(0.01 * 1e18)), ("celo", 10), ("other", 0)]
    )
    def test_get_min_deploy_value(self, chain: str, expected: int) -> None:
        """Test test_get_min_deploy_value."""
        b = _make_behaviour()
        b.params.home_chain_id = chain
        b.get_chain_id = lambda: MemeooorrBaseBehaviour.get_chain_id(b)
        assert MemeooorrBaseBehaviour.get_min_deploy_value(b) == expected


# ---------------------------------------------------------------------------
# get_tweets_from_db (lines 846-855) tests
# ---------------------------------------------------------------------------


class TestGetTweetsFromDb:
    """Test TestGetTweetsFromDb."""

    def test_none(self) -> None:
        """Test test_none."""
        b = _make_behaviour()
        b.read_kv = MagicMock(side_effect=_fake_read_kv(None))
        assert not _exhaust(MemeooorrBaseBehaviour.get_tweets_from_db(b))

    def test_empty(self) -> None:
        """Test test_empty."""
        b = _make_behaviour()
        b.read_kv = MagicMock(side_effect=_fake_read_kv({"tweets": None}))
        assert not _exhaust(MemeooorrBaseBehaviour.get_tweets_from_db(b))

    def test_with_data(self) -> None:
        """Test test_with_data."""
        b = _make_behaviour()
        b.read_kv = MagicMock(
            side_effect=_fake_read_kv({"tweets": json.dumps([{"t": 1}])})
        )
        assert _exhaust(MemeooorrBaseBehaviour.get_tweets_from_db(b)) == [{"t": 1}]


# ---------------------------------------------------------------------------
# get_burnable_amount (lines 859-875) tests
# ---------------------------------------------------------------------------


class TestGetBurnableAmount:
    """Test TestGetBurnableAmount."""

    def test_success(self) -> None:
        """Test test_success."""
        b = _make_behaviour()
        b.get_chain_id = MagicMock(return_value="base")
        b.get_meme_factory_address = MagicMock(return_value="0xf")

        def fake(**kw: Any) -> Generator[Any, None, Any]:
            """Test fake."""
            r = MagicMock(performative=ContractApiMessage.Performative.STATE)
            r.state.body = {"burnable_amount": 42}
            yield
            return r

        b.get_contract_api_response = MagicMock(side_effect=fake)
        assert _exhaust(MemeooorrBaseBehaviour.get_burnable_amount(b)) == 42

    def test_error(self) -> None:
        """Test test_error."""
        b = _make_behaviour()
        b.get_chain_id = MagicMock(return_value="base")
        b.get_meme_factory_address = MagicMock(return_value="0xf")

        def fake(**kw: Any) -> Generator[Any, None, Any]:
            """Test fake."""
            r = MagicMock(performative=ContractApiMessage.Performative.ERROR)
            yield
            return r

        b.get_contract_api_response = MagicMock(side_effect=fake)
        assert _exhaust(MemeooorrBaseBehaviour.get_burnable_amount(b)) == 0


# ---------------------------------------------------------------------------
# get_collectable_amount (lines 879-902) tests
# ---------------------------------------------------------------------------


class TestGetCollectableAmount:
    """Test TestGetCollectableAmount."""

    def test_success(self) -> None:
        """Test test_success."""
        b = _make_behaviour()
        b.get_chain_id = MagicMock(return_value="base")
        b.get_meme_factory_address = MagicMock(return_value="0xf")

        def fake(**kw: Any) -> Generator[Any, None, Any]:
            """Test fake."""
            r = MagicMock(performative=ContractApiMessage.Performative.STATE)
            r.state.body = {"collectable_amount": 99}
            yield
            return r

        b.get_contract_api_response = MagicMock(side_effect=fake)
        assert _exhaust(MemeooorrBaseBehaviour.get_collectable_amount(b, 5)) == 99

    def test_error(self) -> None:
        """Test test_error."""
        b = _make_behaviour()
        b.get_chain_id = MagicMock(return_value="base")
        b.get_meme_factory_address = MagicMock(return_value="0xf")

        def fake(**kw: Any) -> Generator[Any, None, Any]:
            """Test fake."""
            r = MagicMock(performative=ContractApiMessage.Performative.ERROR)
            yield
            return r

        b.get_contract_api_response = MagicMock(side_effect=fake)
        assert _exhaust(MemeooorrBaseBehaviour.get_collectable_amount(b, 5)) == 0


# ---------------------------------------------------------------------------
# get_purged_memes_from_chain (lines 906-924) tests
# ---------------------------------------------------------------------------


class TestGetPurgedMemes:
    """Test TestGetPurgedMemes."""

    def test_success(self) -> None:
        """Test test_success."""
        b = _make_behaviour()
        b.get_chain_id = MagicMock(return_value="base")
        b.get_meme_factory_address = MagicMock(return_value="0xf")

        def fake(**kw: Any) -> Generator[Any, None, Any]:
            """Test fake."""
            r = MagicMock(performative=ContractApiMessage.Performative.STATE)
            r.state.body = {"purged_addresses": ["0x1"]}
            yield
            return r

        b.get_contract_api_response = MagicMock(side_effect=fake)
        assert _exhaust(MemeooorrBaseBehaviour.get_purged_memes_from_chain(b)) == [
            "0x1"
        ]

    def test_error(self) -> None:
        """Test test_error."""
        b = _make_behaviour()
        b.get_chain_id = MagicMock(return_value="base")
        b.get_meme_factory_address = MagicMock(return_value="0xf")

        def fake(**kw: Any) -> Generator[Any, None, Any]:
            """Test fake."""
            r = MagicMock(performative=ContractApiMessage.Performative.ERROR)
            yield
            return r

        b.get_contract_api_response = MagicMock(side_effect=fake)
        assert not _exhaust(MemeooorrBaseBehaviour.get_purged_memes_from_chain(b))


# ---------------------------------------------------------------------------
# replace_tweet_with_alternative_model (lines 931-994) tests
# ---------------------------------------------------------------------------


class TestReplaceTweet:
    """Test TestReplaceTweet."""

    def _make_alt_config(self, use: bool = True) -> MagicMock:
        """Test _make_alt_config."""
        alt = MagicMock()
        alt.use = use
        alt.model = "m"
        alt.max_tokens = 10
        alt.top_p = 1
        alt.top_k = 50
        alt.presence_penalty = 0
        alt.frequency_penalty = 0
        alt.temperature = 0.7
        alt.api_key = "key"
        alt.url = "https://api.example.com"
        return alt

    def test_disabled(self) -> None:
        """Test test_disabled."""
        b = _make_behaviour()
        b.params.alternative_model_for_tweets = self._make_alt_config(use=False)
        assert (
            _exhaust(
                MemeooorrBaseBehaviour.replace_tweet_with_alternative_model(b, "p")
            )
            is None
        )

    def test_success(self) -> None:
        """Test test_success."""
        b = _make_behaviour()
        b.params.alternative_model_for_tweets = self._make_alt_config()

        def fake_http(**kw: Any) -> Generator[Any, None, Any]:
            """Test fake_http."""
            r = MagicMock()
            r.status_code = HTTP_OK
            r.body = json.dumps({"choices": [{"message": {"content": "Short tweet"}}]})
            yield
            return r

        b.get_http_response = MagicMock(side_effect=fake_http)
        assert (
            _exhaust(
                MemeooorrBaseBehaviour.replace_tweet_with_alternative_model(b, "p")
            )
            == "Short tweet"
        )

    def test_http_error_with_api_error(self) -> None:
        """Test test_http_error_with_api_error."""
        b = _make_behaviour()
        b.params.alternative_model_for_tweets = self._make_alt_config()

        def fake_http(**kw: Any) -> Generator[Any, None, Any]:
            """Test fake_http."""
            r = MagicMock()
            r.status_code = 500
            r.body = json.dumps({"error": "Server error"})
            yield
            return r

        b.get_http_response = MagicMock(side_effect=fake_http)
        assert (
            _exhaust(
                MemeooorrBaseBehaviour.replace_tweet_with_alternative_model(b, "p")
            )
            is None
        )

    def test_invalid_format(self) -> None:
        """Test test_invalid_format."""
        b = _make_behaviour()
        b.params.alternative_model_for_tweets = self._make_alt_config()

        def fake_http(**kw: Any) -> Generator[Any, None, Any]:
            """Test fake_http."""
            r = MagicMock()
            r.status_code = HTTP_OK
            r.body = json.dumps({"unexpected": True})
            yield
            return r

        b.get_http_response = MagicMock(side_effect=fake_http)
        assert (
            _exhaust(
                MemeooorrBaseBehaviour.replace_tweet_with_alternative_model(b, "p")
            )
            is None
        )

    def test_tweet_too_long(self) -> None:
        """Test test_tweet_too_long."""
        b = _make_behaviour()
        b.params.alternative_model_for_tweets = self._make_alt_config()

        def fake_http(**kw: Any) -> Generator[Any, None, Any]:
            """Test fake_http."""
            r = MagicMock()
            r.status_code = HTTP_OK
            r.body = json.dumps({"choices": [{"message": {"content": "x" * 500}}]})
            yield
            return r

        b.get_http_response = MagicMock(side_effect=fake_http)
        assert (
            _exhaust(
                MemeooorrBaseBehaviour.replace_tweet_with_alternative_model(b, "p")
            )
            is None
        )

    def test_http_error_no_api_error(self) -> None:
        """HTTP error but body has no 'error' key -- hits the status_code branch then continues."""
        b = _make_behaviour()
        b.params.alternative_model_for_tweets = self._make_alt_config()

        def fake_http(**kw: Any) -> Generator[Any, None, Any]:
            """Test fake_http."""
            r = MagicMock()
            r.status_code = 500
            r.body = json.dumps({"choices": [{"message": {"content": "ok"}}]})
            yield
            return r

        b.get_http_response = MagicMock(side_effect=fake_http)
        assert (
            _exhaust(
                MemeooorrBaseBehaviour.replace_tweet_with_alternative_model(b, "p")
            )
            == "ok"
        )


# ---------------------------------------------------------------------------
# init_own_twitter_details (lines 999-1037) tests
# ---------------------------------------------------------------------------


class TestInitOwnTwitterDetails:
    """Test TestInitOwnTwitterDetails."""

    def test_case1_state_ok_db_missing_details(self) -> None:
        """Test test_case1_state_ok_db_missing_details."""
        b = _make_behaviour()
        b.context.state.twitter_username = "mybot"
        b.context.state.twitter_id = "123"
        db_agent = MagicMock()
        db_agent.twitter_username = None
        db_agent.twitter_user_id = None

        def fake_update() -> Generator[Any, None, None]:
            """Test fake_update."""
            yield

        db_agent.update_twitter_details = MagicMock(side_effect=fake_update)
        b.context.agents_fun_db = MagicMock()
        b.context.agents_fun_db.my_agent = db_agent
        _exhaust(MemeooorrBaseBehaviour.init_own_twitter_details(b))
        assert db_agent.twitter_username == "mybot"

    def test_case1_state_ok_db_has_details(self) -> None:
        """Test test_case1_state_ok_db_has_details."""
        b = _make_behaviour()
        b.context.state.twitter_username = "mybot"
        b.context.state.twitter_id = "123"
        db_agent = MagicMock()
        db_agent.twitter_username = "mybot"
        db_agent.twitter_user_id = "123"
        b.context.agents_fun_db = MagicMock()
        b.context.agents_fun_db.my_agent = db_agent
        _exhaust(MemeooorrBaseBehaviour.init_own_twitter_details(b))
        db_agent.update_twitter_details.assert_not_called()

    def test_case1_state_ok_db_not_initialized(self) -> None:
        """State has details but DB is None."""
        b = _make_behaviour()
        b.context.state.twitter_username = "mybot"
        b.context.state.twitter_id = "123"
        b.context.agents_fun_db = MagicMock()
        b.context.agents_fun_db.my_agent = None
        _exhaust(MemeooorrBaseBehaviour.init_own_twitter_details(b))
        # Should return early without error

    def test_case2_fetch_fails(self) -> None:
        """Test test_case2_fetch_fails."""
        b = _make_behaviour()
        b.context.state.twitter_username = None
        b.context.state.twitter_id = None
        b.context.agents_fun_db = MagicMock()
        b.context.agents_fun_db.my_agent = MagicMock()

        def fake(method: Any) -> Generator[Any, None, None]:
            """Test fake."""
            yield

        b._call_tweepy = MagicMock(side_effect=fake)
        _exhaust(MemeooorrBaseBehaviour.init_own_twitter_details(b))
        b.context.logger.error.assert_called()

    def test_case3_fetch_ok_db_not_initialized(self) -> None:
        """Test test_case3_fetch_ok_db_not_initialized."""
        b = _make_behaviour()
        b.context.state.twitter_username = None
        b.context.state.twitter_id = None
        b.context.agents_fun_db = MagicMock()
        b.context.agents_fun_db.my_agent = None

        def fake(method: Any) -> Generator[Any, None, Any]:
            """Test fake."""
            yield
            return {"username": "bot", "user_id": "456", "display_name": "Bot"}

        b._call_tweepy = MagicMock(side_effect=fake)
        _exhaust(MemeooorrBaseBehaviour.init_own_twitter_details(b))
        assert b.context.state.twitter_username == "bot"
        assert b.context.state.twitter_id == "456"

    def test_case4_fetch_ok_db_initialized(self) -> None:
        """Test test_case4_fetch_ok_db_initialized."""
        b = _make_behaviour()
        b.context.state.twitter_username = None
        b.context.state.twitter_id = None
        db_agent = MagicMock()

        def fake_update() -> Generator[Any, None, None]:
            """Test fake_update."""
            yield

        db_agent.update_twitter_details = MagicMock(side_effect=fake_update)
        b.context.agents_fun_db = MagicMock()
        b.context.agents_fun_db.my_agent = db_agent

        def fake(method: Any) -> Generator[Any, None, Any]:
            """Test fake."""
            yield
            return {"username": "bot", "user_id": "456", "display_name": "Bot"}

        b._call_tweepy = MagicMock(side_effect=fake)
        _exhaust(MemeooorrBaseBehaviour.init_own_twitter_details(b))
        assert db_agent.twitter_username == "bot"
        assert db_agent.twitter_user_id == "456"


# ---------------------------------------------------------------------------
# _store_agent_action (lines 1039-1069) tests
# ---------------------------------------------------------------------------


class TestStoreAgentAction:
    """Test TestStoreAgentAction."""

    def _setup(self, b: Any, read_value: Any) -> None:
        """Test _setup."""
        b._read_json_from_kv = MagicMock(
            side_effect=_fake_read_json_from_kv(read_value)
        )
        b._write_kv = MagicMock(side_effect=_fake_write_kv(True))
        _setup_sync_timestamp(b)

    def test_new_action_dict(self) -> None:
        """Test test_new_action_dict."""
        b = _make_behaviour()
        self._setup(b, {})
        _exhaust(
            MemeooorrBaseBehaviour._store_agent_action(
                b, "tweet_action", {"text": "hi"}
            )
        )
        b._write_kv.assert_called_once()

    def test_non_list_type_warning(self) -> None:
        """Test test_non_list_type_warning."""
        b = _make_behaviour()
        self._setup(
            b, {"tool_action": "not_list", "tweet_action": [], "token_action": []}
        )
        _exhaust(MemeooorrBaseBehaviour._store_agent_action(b, "tool_action", {"t": 1}))
        b.context.logger.warning.assert_called()

    def test_non_dict_data(self) -> None:
        """Non-dict action_data should not get timestamp added."""
        b = _make_behaviour()
        self._setup(b, {})
        _exhaust(
            MemeooorrBaseBehaviour._store_agent_action(b, "tweet_action", "string_data")
        )
        b._write_kv.assert_called_once()

    def test_dict_data_gets_timestamp(self) -> None:
        """Test test_dict_data_gets_timestamp."""
        b = _make_behaviour()
        self._setup(b, {})
        action = {"text": "hello"}
        _exhaust(MemeooorrBaseBehaviour._store_agent_action(b, "tweet_action", action))
        assert "timestamp" in action


# ---------------------------------------------------------------------------
# get_latest_agent_actions (lines 1071-1092) tests
# ---------------------------------------------------------------------------


class TestGetLatestAgentActions:
    """Test TestGetLatestAgentActions."""

    def test_success(self) -> None:
        """Test test_success."""
        b = _make_behaviour()
        b._read_json_from_kv = MagicMock(
            side_effect=_fake_read_json_from_kv(
                {
                    "tool_action": [1, 2, 3],
                }
            )
        )
        assert _exhaust(
            MemeooorrBaseBehaviour.get_latest_agent_actions(b, "tool_action", 2)
        ) == [2, 3]

    def test_not_a_list(self) -> None:
        """Test test_not_a_list."""
        b = _make_behaviour()
        b._read_json_from_kv = MagicMock(
            side_effect=_fake_read_json_from_kv({"tool_action": "bad"})
        )
        assert not _exhaust(
            MemeooorrBaseBehaviour.get_latest_agent_actions(b, "tool_action", 5)
        )

    def test_missing_type(self) -> None:
        """Test test_missing_type."""
        b = _make_behaviour()
        b._read_json_from_kv = MagicMock(side_effect=_fake_read_json_from_kv({}))
        assert not _exhaust(MemeooorrBaseBehaviour.get_latest_agent_actions(b, "x", 5))


# ---------------------------------------------------------------------------
# _read_json_from_kv (lines 1094-1108) tests
# ---------------------------------------------------------------------------


class TestReadJsonFromKv:
    """Test TestReadJsonFromKv."""

    def test_success(self) -> None:
        """Test test_success."""
        b = _make_behaviour()
        b._read_kv = MagicMock(side_effect=_fake_read_kv({"k": json.dumps({"a": 1})}))
        assert _exhaust(MemeooorrBaseBehaviour._read_json_from_kv(b, "k", {})) == {
            "a": 1
        }

    def test_no_data(self) -> None:
        """Test test_no_data."""
        b = _make_behaviour()
        b._read_kv = MagicMock(side_effect=_fake_read_kv(None))
        assert _exhaust(
            MemeooorrBaseBehaviour._read_json_from_kv(b, "k", {"d": 1})
        ) == {"d": 1}

    def test_empty_value(self) -> None:
        """Test test_empty_value."""
        b = _make_behaviour()
        b._read_kv = MagicMock(side_effect=_fake_read_kv({"k": None}))
        assert _exhaust(MemeooorrBaseBehaviour._read_json_from_kv(b, "k", "fb")) == "fb"

    def test_invalid_json(self) -> None:
        """Test test_invalid_json."""
        b = _make_behaviour()
        b._read_kv = MagicMock(side_effect=_fake_read_kv({"k": "{{bad"}))
        assert (
            _exhaust(MemeooorrBaseBehaviour._read_json_from_kv(b, "k", "def")) == "def"
        )


# ---------------------------------------------------------------------------
# _read_value_from_kv (lines 1110-1119) tests
# ---------------------------------------------------------------------------


class TestReadValueFromKv:
    """Test TestReadValueFromKv."""

    def test_success(self) -> None:
        """Test test_success."""
        b = _make_behaviour()
        b._read_kv = MagicMock(side_effect=_fake_read_kv({"k": "hello"}))
        assert (
            _exhaust(MemeooorrBaseBehaviour._read_value_from_kv(b, "k", "def"))
            == "hello"
        )

    def test_no_data(self) -> None:
        """Test test_no_data."""
        b = _make_behaviour()
        b._read_kv = MagicMock(side_effect=_fake_read_kv(None))
        assert (
            _exhaust(MemeooorrBaseBehaviour._read_value_from_kv(b, "k", "def")) == "def"
        )

    def test_empty_value(self) -> None:
        """Test test_empty_value."""
        b = _make_behaviour()
        b._read_kv = MagicMock(side_effect=_fake_read_kv({"k": ""}))
        assert (
            _exhaust(MemeooorrBaseBehaviour._read_value_from_kv(b, "k", "def")) == "def"
        )


# ---------------------------------------------------------------------------
# _store_media_info_list / _read_media_info_list (lines 1121-1149) tests
# ---------------------------------------------------------------------------


class TestMediaInfoList:
    """Test TestMediaInfoList."""

    def test_store_within_limit(self) -> None:
        """Test test_store_within_limit."""
        b = _make_behaviour()
        b._read_json_from_kv = MagicMock(side_effect=_fake_read_json_from_kv([]))
        b._write_kv = MagicMock(side_effect=_fake_write_kv())
        b._cleanup_temp_file = MagicMock()
        # We need _read_media_info_list to call _read_json_from_kv
        b._read_media_info_list = MagicMock(
            side_effect=lambda: (yield from _gen_val([]))
        )
        _exhaust(MemeooorrBaseBehaviour._store_media_info_list(b, {"path": "/tmp/x"}))
        b._write_kv.assert_called_once()
        b._cleanup_temp_file.assert_not_called()

    def test_store_exceeds_limit(self) -> None:
        """Test test_store_exceeds_limit."""
        b = _make_behaviour()
        existing = [{"path": f"/tmp/img{i}.png"} for i in range(LIST_COUNT_TO_KEEP)]
        b._read_media_info_list = MagicMock(
            side_effect=lambda: (yield from _gen_val(list(existing)))
        )
        b._write_kv = MagicMock(side_effect=_fake_write_kv())
        b._cleanup_temp_file = MagicMock()
        _exhaust(
            MemeooorrBaseBehaviour._store_media_info_list(b, {"path": "/tmp/new.png"})
        )
        b._cleanup_temp_file.assert_called()

    def test_read_success(self) -> None:
        """Test test_read_success."""
        b = _make_behaviour()
        b._read_json_from_kv = MagicMock(
            side_effect=_fake_read_json_from_kv([{"path": "/tmp/a"}])
        )
        assert _exhaust(MemeooorrBaseBehaviour._read_media_info_list(b)) == [
            {"path": "/tmp/a"}
        ]

    def test_read_not_a_list(self) -> None:
        """Test test_read_not_a_list."""
        b = _make_behaviour()
        b._read_json_from_kv = MagicMock(
            side_effect=_fake_read_json_from_kv("not_list")
        )
        assert not _exhaust(MemeooorrBaseBehaviour._read_media_info_list(b))


def _gen_val(v: Any) -> Generator[Any, None, Any]:
    """A simple generator that yields once then returns v."""
    yield
    return v


# ---------------------------------------------------------------------------
# _cleanup_temp_file (lines 1151-1158) tests
# ---------------------------------------------------------------------------


class TestCleanupTempFile:
    """Test TestCleanupTempFile."""

    def test_existing_file(self) -> None:
        """Test test_existing_file."""
        b = _make_behaviour()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            p = f.name
        MemeooorrBaseBehaviour._cleanup_temp_file(b, p, "test")
        assert not Path(p).exists()

    def test_none_path(self) -> None:
        """Test test_none_path."""
        b = _make_behaviour()
        MemeooorrBaseBehaviour._cleanup_temp_file(b, None, "test")
        b.context.logger.warning.assert_called()

    def test_empty_path(self) -> None:
        """Test test_empty_path."""
        b = _make_behaviour()
        MemeooorrBaseBehaviour._cleanup_temp_file(b, "", "test")
        b.context.logger.warning.assert_called()
