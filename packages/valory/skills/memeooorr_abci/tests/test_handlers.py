# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024-2026 Valory AG
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

"""Tests for handlers.py"""

# pylint: disable=no-member,protected-access

import json
from datetime import datetime
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, mock_open, patch

import pytest

from packages.valory.connections.http_server.connection import (
    PUBLIC_ID as HTTP_SERVER_PUBLIC_ID,
)
from packages.valory.protocols.http.message import HttpMessage
from packages.valory.protocols.kv_store.message import KvStoreMessage
from packages.valory.protocols.srr.message import SrrMessage
from packages.valory.skills.abstract_round_abci.handlers import (
    HttpHandler as BaseHttpHandler,
)
from packages.valory.skills.memeooorr_abci.handlers import (
    BAD_REQUEST_CODE,
    GENAI_API_KEY_NOT_SET_ERROR,
    GENAI_RATE_LIMIT_ERROR,
    HttpHandler,
    HttpMethod,
    INTERNAL_SERVER_ERROR_CODE,
    KvStoreHandler,
    NOT_FOUND_CODE,
    OK_CODE,
    SrrHandler,
    Store,
    TOO_EARLY_CODE,
    TOO_MANY_REQUESTS_CODE,
    USDC_ADDRESS,
    camel_to_snake,
    get_password_from_args,
    load_fsm_spec,
)

# =============================================================================
# Helper / utility tests
# =============================================================================


class TestCamelToSnake:  # pylint: disable=too-few-public-methods
    """Tests for camel_to_snake utility."""

    @pytest.mark.parametrize(
        "input_str,expected",
        [
            ("CamelCase", "camel_case"),
            ("HTTPServer", "h_t_t_p_server"),
            ("SimpleTest", "simple_test"),
            ("already_snake", "already_snake"),
            ("A", "a"),
        ],
    )
    def test_camel_to_snake(self, input_str: str, expected: str) -> None:
        """Test camel to snake conversion."""
        assert camel_to_snake(input_str) == expected


class TestLoadFsmSpec:  # pylint: disable=too-few-public-methods
    """Tests for load_fsm_spec."""

    def test_load_fsm_spec(self) -> None:
        """Test loading FSM spec from file."""
        result = load_fsm_spec()
        assert isinstance(result, dict)
        assert "transition_func" in result


class TestGetPasswordFromArgs:
    """Tests for get_password_from_args."""

    def test_no_password(self) -> None:
        """Test when no password argument is provided."""
        with patch("sys.argv", ["test"]):
            result = get_password_from_args()
            assert result is None

    def test_with_password(self) -> None:
        """Test when password argument is provided."""
        with patch("sys.argv", ["test", "--password", "secret123"]):
            result = get_password_from_args()
            assert result == "secret123"


class TestHttpMethod:  # pylint: disable=too-few-public-methods
    """Tests for HttpMethod enum."""

    def test_values(self) -> None:
        """Test enum values."""
        assert HttpMethod.GET.value == "get"
        assert HttpMethod.HEAD.value == "head"
        assert HttpMethod.POST.value == "post"


# =============================================================================
# SrrHandler tests
# =============================================================================


def _make_mock_srr_message(
    performative: SrrMessage.Performative = SrrMessage.Performative.RESPONSE,  # type: ignore[assignment]
    dialogue_reference: tuple = ("nonce1", ""),
    payload: str = "{}",
) -> MagicMock:
    """Create a mock SrrMessage."""
    msg = MagicMock(spec=SrrMessage)
    msg.performative = performative
    msg.dialogue_reference = dialogue_reference
    msg.payload = payload
    return msg


class TestSrrHandler:
    """Tests for SrrHandler."""

    def _make_handler(self) -> MagicMock:
        """Create a mock handler with SrrHandler behavior."""
        handler = MagicMock(spec=SrrHandler)
        handler.context = MagicMock()
        handler.context.logger = MagicMock()
        handler.context.state = MagicMock()
        handler.context.state.req_to_callback = {}
        handler.context.srr_dialogues = MagicMock()
        handler.allowed_response_performatives = (
            SrrHandler.allowed_response_performatives
        )
        return handler

    def test_handle_unrecognized_performative(self) -> None:
        """Test handling of unrecognized performative."""
        handler = self._make_handler()
        msg = _make_mock_srr_message()
        # Use a performative not in allowed set
        msg.performative = "UNKNOWN"

        SrrHandler.handle(handler, msg)

        handler.context.logger.warning.assert_called_once()

    def test_handle_no_callback(self) -> None:
        """Test handling when no callback is registered for the nonce."""
        handler = self._make_handler()
        msg = _make_mock_srr_message(
            performative=SrrMessage.Performative.RESPONSE,  # type: ignore[arg-type]
            dialogue_reference=("unknown_nonce", ""),
        )

        SrrHandler.handle(handler, msg)

        # Should call super().handle since callback is None
        # The super() call is on the mock, so we check via the mock
        # In the actual code, it falls through to super().handle(message)

    def test_handle_with_callback(self) -> None:
        """Test handling when a callback is registered."""
        handler = self._make_handler()
        callback = MagicMock()
        handler.context.state.req_to_callback = {"nonce1": (callback, {"key": "value"})}
        msg = _make_mock_srr_message(
            performative=SrrMessage.Performative.RESPONSE,  # type: ignore[arg-type]
            dialogue_reference=("nonce1", ""),
        )
        dialogue = MagicMock()
        handler.context.srr_dialogues.update.return_value = dialogue

        SrrHandler.handle(handler, msg)

        callback.assert_called_once()
        assert "nonce1" not in handler.context.state.req_to_callback

    def test_handle_request_performative(self) -> None:
        """Test handling REQUEST performative (also allowed)."""
        handler = self._make_handler()
        msg = _make_mock_srr_message(
            performative=SrrMessage.Performative.REQUEST,  # type: ignore[arg-type]
            dialogue_reference=("nonce_req", ""),
        )

        SrrHandler.handle(handler, msg)
        # No callback registered, should go to super()


# =============================================================================
# KvStoreHandler tests
# =============================================================================


class TestKvStoreHandler:
    """Tests for KvStoreHandler."""

    def test_supported_protocol(self) -> None:
        """Test the supported protocol is KvStoreMessage."""
        assert KvStoreHandler.SUPPORTED_PROTOCOL == KvStoreMessage.protocol_id

    def test_allowed_performatives(self) -> None:
        """Test allowed response performatives."""
        expected = frozenset(
            {
                KvStoreMessage.Performative.READ_REQUEST,
                KvStoreMessage.Performative.CREATE_OR_UPDATE_REQUEST,
                KvStoreMessage.Performative.READ_RESPONSE,
                KvStoreMessage.Performative.SUCCESS,
                KvStoreMessage.Performative.ERROR,
            }
        )
        assert KvStoreHandler.allowed_response_performatives == expected


# =============================================================================
# HttpHandler tests
# =============================================================================


def _make_http_handler() -> MagicMock:
    """Create a mock HttpHandler with all necessary attributes."""
    handler = MagicMock(spec=HttpHandler)
    handler.context = MagicMock()
    handler.context.logger = MagicMock()
    handler.context.outbox = MagicMock()
    handler.context.state = MagicMock()
    handler.context.state.round_sequence = MagicMock()
    handler.context.state.env_var_status = {}
    handler.context.params = MagicMock()
    handler.context.params.service_endpoint = "http://localhost:8000"
    handler.context.params.reset_pause_duration = 10
    handler.context.params.store_path = "/tmp/test_store"
    handler.context.params.ipfs_address = "https://gateway.autonolas.tech/ipfs/"
    handler.context.params.use_x402 = False
    handler.context.params.genai_api_key = "test_key"
    handler.context.params.is_memecoin_logic_enabled = True
    handler.context.params.base_ledger_rpc = "http://localhost:8545"
    handler.context.params.lifi_quote_to_amount_url = (
        "https://li.quest/v1/quote/toAmount"
    )
    handler.context.params.x402_payment_requirements = {
        "threshold": 1000000,
        "top_up": 5000000,
    }
    handler.context.http_dialogues = MagicMock()
    handler.context.srr_dialogues = MagicMock()
    handler.context.default_ledger_id = "ethereum"
    handler.context.data_dir = "/tmp/test_data"

    handler.json_content_header = "Content-Type: application/json\n"
    handler.html_content_header = "Content-Type: text/html\n"
    handler.handler_url_regex = r".*localhost(:\d+)?\/.*"
    handler.agent_profile_path = "agentsfun-ui-build"
    handler.rounds_info = {}
    handler.executor = MagicMock()

    # Routes
    hostname_regex = r".*(localhost)(:\d+)?"
    handler.routes = {
        ("post",): [
            (
                rf"{hostname_regex}\/configure_strategies",
                handler._handle_post_process_prompt,
            ),
        ],
        ("get", "head"): [
            (rf"{hostname_regex}\/healthcheck", handler._handle_get_health),
            (rf"{hostname_regex}\/agent-info", handler._handle_get_agent_details),
            (rf"{hostname_regex}\/x-activity", handler._handle_get_recent_x_activity),
            (rf"{hostname_regex}\/memecoin-activity", handler._handle_get_meme_coins),
            (rf"{hostname_regex}\/media", handler._handle_get_media),
            (rf"{hostname_regex}\/funds-status", handler._handle_get_funds_status),
            (rf"{hostname_regex}\/features", handler._handle_get_features),
            (rf"{hostname_regex}\/(.*)", handler._handle_get_static_file),
        ],
    }

    return handler


def _make_http_msg(  # pylint: disable=too-many-arguments, too-many-positional-arguments
    method: str = "get",
    url: str = "http://localhost:8000/healthcheck",
    body: bytes = b"",
    headers: str = "",
    version: str = "HTTP/1.1",
    performative: HttpMessage.Performative = HttpMessage.Performative.REQUEST,  # type: ignore[assignment]
    sender: Optional[str] = None,
) -> MagicMock:
    """Create a mock HttpMessage."""
    msg = MagicMock(spec=HttpMessage)
    msg.method = method
    msg.url = url
    msg.body = body
    msg.headers = headers
    msg.version = version
    msg.performative = performative
    msg.sender = sender or str(HTTP_SERVER_PUBLIC_ID.without_hash())
    return msg


def _make_http_dialogue() -> MagicMock:
    """Create a mock HttpDialogue."""
    dialogue = MagicMock()
    dialogue.reply.return_value = MagicMock()
    dialogue.dialogue_label = MagicMock()
    dialogue.dialogue_label.dialogue_reference = ("nonce1", "")
    return dialogue


class TestHttpHandlerInit:
    """Tests for HttpHandler.__init__."""

    def test_init(self) -> None:
        """Test HttpHandler initialization."""
        with patch.object(BaseHttpHandler, "__init__", return_value=None):
            handler = MagicMock(spec=HttpHandler)
            handler.executor = None

            HttpHandler.__init__(handler)

            assert handler.executor is not None

    def test_executor_shutdown(self) -> None:
        """Test _executor_shutdown."""
        handler = _make_http_handler()
        HttpHandler._executor_shutdown(handler)
        handler.executor.shutdown.assert_called_once_with(
            wait=False, cancel_futures=True
        )

    def test_teardown(self) -> None:
        """Test teardown calls super and _executor_shutdown."""
        handler = _make_http_handler()
        handler._executor_shutdown = MagicMock()
        with patch.object(BaseHttpHandler, "teardown"):
            HttpHandler.teardown(handler)
        handler._executor_shutdown.assert_called_once()


class TestHttpHandlerSetup:
    """Tests for HttpHandler.setup."""

    def test_setup_without_x402(self) -> None:
        """Test setup when x402 is disabled."""
        handler = _make_http_handler()
        handler.context.params.use_x402 = False
        handler.params = handler.context.params
        handler.shared_state = handler.context.state

        with patch(
            "packages.valory.skills.memeooorr_abci.handlers.load_fsm_spec"
        ) as mock_fsm:
            mock_fsm.return_value = {
                "transition_func": {
                    "(ActionDecisionRound, done)": "ResetRound",
                }
            }
            HttpHandler.setup(handler)

        # Should NOT submit x402 check
        handler.executor.submit.assert_not_called()

    def test_setup_with_x402(self) -> None:
        """Test setup when x402 is enabled."""
        handler = _make_http_handler()
        handler.context.params.use_x402 = True
        handler.params = handler.context.params
        handler.shared_state = handler.context.state
        handler._x402_swap_future = None
        handler._submit_x402_swap_if_idle = (
            lambda: HttpHandler._submit_x402_swap_if_idle(handler)
        )

        with patch(
            "packages.valory.skills.memeooorr_abci.handlers.load_fsm_spec"
        ) as mock_fsm:
            mock_fsm.return_value = {
                "transition_func": {
                    "(ActionDecisionRound, done)": "ResetRound",
                }
            }
            HttpHandler.setup(handler)

        handler.executor.submit.assert_called_once_with(
            handler._ensure_sufficient_funds_for_x402_payments
        )
        assert handler._x402_swap_future is not None


class TestHttpHandlerProperties:
    """Tests for HttpHandler properties."""

    def test_synchronized_data(self) -> None:
        """Test synchronized_data property."""
        handler = _make_http_handler()
        handler.context.state.round_sequence.latest_synchronized_data.db = MagicMock()

        with patch(
            "packages.valory.skills.memeooorr_abci.handlers.SynchronizedData"
        ) as mock_sd:
            mock_sd.return_value = "sync_data"
            result = HttpHandler.synchronized_data.fget(handler)  # type: ignore  # pylint: disable=assignment-from-no-return
            assert result == "sync_data"

    def test_params_property(self) -> None:
        """Test params property."""
        handler = _make_http_handler()
        result = HttpHandler.params.fget(handler)  # type: ignore  # pylint: disable=assignment-from-no-return
        assert result == handler.context.params

    def test_is_memecoin_logic_enabled(self) -> None:
        """Test is_memecoin_logic_enabled property."""
        handler = _make_http_handler()
        handler.params = handler.context.params
        handler.context.params.is_memecoin_logic_enabled = True
        result = HttpHandler.is_memecoin_logic_enabled.fget(handler)  # type: ignore  # pylint: disable=assignment-from-no-return
        assert result is True

    def test_shared_state(self) -> None:
        """Test shared_state property."""
        handler = _make_http_handler()
        result = HttpHandler.shared_state.fget(handler)  # type: ignore  # pylint: disable=assignment-from-no-return
        assert result is not None

    def test_funds_status(self) -> None:
        """Test funds_status property."""
        handler = _make_http_handler()
        mock_fn = MagicMock(return_value="fund_reqs")
        handler.context.shared_state = {"get_funds_status": mock_fn}

        with patch(
            "packages.valory.skills.memeooorr_abci.handlers.GET_FUNDS_STATUS_METHOD_NAME",
            "get_funds_status",
        ):
            result = HttpHandler.funds_status.fget(handler)  # type: ignore  # pylint: disable=assignment-from-no-return
            assert result == "fund_reqs"


class TestHttpHandlerDbMethods:
    """Tests for database-related methods."""

    def test_db_connect(self) -> None:
        """Test db_connect."""
        handler = _make_http_handler()
        with patch(
            "packages.valory.skills.memeooorr_abci.handlers.peewee.SqliteDatabase"
        ) as mock_sqlite:
            mock_db = MagicMock()
            mock_sqlite.return_value = mock_db
            with patch(
                "packages.valory.skills.memeooorr_abci.handlers.db"
            ) as mock_proxy:
                HttpHandler.db_connect(handler)
                mock_proxy.initialize.assert_called_once_with(mock_db)
                mock_db.connect.assert_called_once()

    def test_db_disconnect_open(self) -> None:
        """Test db_disconnect when db is open."""
        handler = _make_http_handler()
        handler.db = MagicMock()
        handler.db.is_closed.return_value = False
        HttpHandler.db_disconnect(handler)
        handler.db.close.assert_called_once()

    def test_db_disconnect_already_closed(self) -> None:
        """Test db_disconnect when db is already closed."""
        handler = _make_http_handler()
        handler.db = MagicMock()
        handler.db.is_closed.return_value = True
        HttpHandler.db_disconnect(handler)
        handler.db.close.assert_not_called()

    def test_db_disconnect_no_db(self) -> None:
        """Test db_disconnect when no db attribute."""
        handler = _make_http_handler()
        # Simulate no db attribute
        del handler.db
        HttpHandler.db_disconnect(handler)

    def test_db_connection_context(self) -> None:
        """Test _db_connection_context context manager."""
        handler = _make_http_handler()
        handler.db_connect = MagicMock()
        handler.db_disconnect = MagicMock()

        with HttpHandler._db_connection_context(handler):
            handler.db_connect.assert_called_once()

        handler.db_disconnect.assert_called_once()

    def test_db_connection_context_exception(self) -> None:
        """Test _db_connection_context cleans up on exception."""
        handler = _make_http_handler()
        handler.db_connect = MagicMock()
        handler.db_disconnect = MagicMock()

        with pytest.raises(ValueError):
            with HttpHandler._db_connection_context(handler):
                raise ValueError("test error")

        handler.db_disconnect.assert_called_once()


class TestGetJsonFromDb:
    """Tests for _get_json_from_db."""

    def test_valid_json(self) -> None:
        """Test getting valid JSON from db."""
        handler = _make_http_handler()
        with patch.object(Store, "get_or_none") as mock_get:
            mock_record = MagicMock()
            mock_record.value = '{"key": "value"}'
            mock_get.return_value = mock_record

            result = HttpHandler._get_json_from_db(handler, "test_key")
            assert result == {"key": "value"}

    def test_no_record(self) -> None:
        """Test when no record exists."""
        handler = _make_http_handler()
        with patch.object(Store, "get_or_none") as mock_get:
            mock_get.return_value = None

            result = HttpHandler._get_json_from_db(handler, "test_key")
            assert result == {}

    def test_invalid_json(self) -> None:
        """Test when value is invalid JSON."""
        handler = _make_http_handler()
        with patch.object(Store, "get_or_none") as mock_get:
            mock_record = MagicMock()
            mock_record.value = "not-json"
            mock_get.return_value = mock_record

            result = HttpHandler._get_json_from_db(handler, "test_key")
            assert result == {}
            handler.context.logger.warning.assert_called_once()

    def test_custom_default(self) -> None:
        """Test with custom default value."""
        handler = _make_http_handler()
        with patch.object(Store, "get_or_none") as mock_get:
            mock_get.return_value = None

            result = HttpHandler._get_json_from_db(handler, "test_key", "[]")
            assert result == []

    def test_empty_value(self) -> None:
        """Test when record has empty value."""
        handler = _make_http_handler()
        with patch.object(Store, "get_or_none") as mock_get:
            mock_record = MagicMock()
            mock_record.value = ""
            mock_get.return_value = mock_record

            result = HttpHandler._get_json_from_db(handler, "test_key")
            assert result == {}


class TestGetValueFromDb:
    """Tests for _get_value_from_db."""

    def test_existing_record(self) -> None:
        """Test getting existing value."""
        with patch.object(Store, "get_or_none") as mock_get:
            mock_record = MagicMock()
            mock_record.value = "some_value"
            mock_get.return_value = mock_record

            result = HttpHandler._get_value_from_db("test_key")
            assert result == "some_value"

    def test_no_record(self) -> None:
        """Test with no record."""
        with patch.object(Store, "get_or_none") as mock_get:
            mock_get.return_value = None

            result = HttpHandler._get_value_from_db("test_key", "default")
            assert result == "default"

    def test_empty_value(self) -> None:
        """Test when value is empty string."""
        with patch.object(Store, "get_or_none") as mock_get:
            mock_record = MagicMock()
            mock_record.value = ""
            mock_get.return_value = mock_record

            result = HttpHandler._get_value_from_db("test_key", "default")
            assert result == "default"


class TestSetValueToDb:
    """Tests for _set_value_to_db."""

    def test_create_new_record(self) -> None:
        """Test creating a new record."""
        with patch.object(Store, "get_or_none") as mock_get:
            mock_get.return_value = None
            with patch.object(Store, "create") as mock_create:
                HttpHandler._set_value_to_db("key1", "value1")
                mock_create.assert_called_once_with(key="key1", value="value1")

    def test_update_existing_record(self) -> None:
        """Test updating an existing record."""
        with patch.object(Store, "get_or_none") as mock_get:
            mock_record = MagicMock()
            mock_get.return_value = mock_record
            HttpHandler._set_value_to_db("key1", "new_value")
            assert mock_record.value == "new_value"
            mock_record.save.assert_called_once()


class TestHttpHandlerGetHandler:
    """Tests for _get_handler."""

    def test_matching_url(self) -> None:
        """Test URL that matches a route."""
        handler = _make_http_handler()

        result_fn, _kwargs = HttpHandler._get_handler(
            handler, "http://localhost:8000/healthcheck", "get"
        )
        assert result_fn is not None

    def test_non_matching_url(self) -> None:
        """Test URL that doesn't match handler pattern."""
        handler = _make_http_handler()
        handler.handler_url_regex = r".*example\.com(:\d+)?\/.*"

        result_fn, _kwargs = HttpHandler._get_handler(
            handler, "http://other-host:8000/healthcheck", "get"
        )
        assert result_fn is None
        handler.context.logger.info.assert_called()

    def test_no_matching_route(self) -> None:
        """Test URL matching handler but no specific route."""
        handler = _make_http_handler()
        # Use DELETE which doesn't match any route
        result_fn, _kwargs = HttpHandler._get_handler(
            handler, "http://localhost:8000/some-path", "delete"
        )
        # Should return _handle_bad_request
        assert result_fn is not None

    def test_post_route_no_match(self) -> None:
        """Test POST with URL that doesn't match any POST route regex."""
        handler = _make_http_handler()
        # POST method matches ("post",) group, but URL won't match configure_strategies
        result_fn, _kwargs = HttpHandler._get_handler(
            handler, "http://localhost:8000/nonexistent-endpoint", "post"
        )
        assert result_fn == handler._handle_bad_request

    def test_post_route(self) -> None:
        """Test POST route matching."""
        handler = _make_http_handler()

        result_fn, _kwargs = HttpHandler._get_handler(
            handler, "http://localhost:8000/configure_strategies", "post"
        )
        assert result_fn is not None


class TestHttpHandlerHandle:
    """Tests for HttpHandler.handle."""

    def test_non_request_performative(self) -> None:
        """Test that non-REQUEST performatives are delegated to super."""
        handler = _make_http_handler()
        msg = _make_http_msg(performative=HttpMessage.Performative.RESPONSE)  # type: ignore[arg-type]

        with patch.object(BaseHttpHandler, "handle") as mock_super:
            HttpHandler.handle(handler, msg)
            mock_super.assert_called_once_with(msg)

    def test_wrong_sender(self) -> None:
        """Test message from non-http-server sender delegates to super."""
        handler = _make_http_handler()
        msg = _make_http_msg(sender="other/skill:0.1.0")

        with patch.object(BaseHttpHandler, "handle") as mock_super:
            HttpHandler.handle(handler, msg)
            mock_super.assert_called_once_with(msg)

    def test_no_matching_handler(self) -> None:
        """Test URL with no matching handler delegates to super."""
        handler = _make_http_handler()
        handler._get_handler = MagicMock(return_value=(None, {}))
        msg = _make_http_msg(url="http://other-host/path")

        with patch.object(BaseHttpHandler, "handle") as mock_super:
            HttpHandler.handle(handler, msg)
            mock_super.assert_called_once_with(msg)

    def test_invalid_dialogue(self) -> None:
        """Test handling when dialogue is None."""
        handler = _make_http_handler()
        handler._get_handler = MagicMock(return_value=(MagicMock(__name__="test"), {}))
        handler.context.http_dialogues.update.return_value = None
        msg = _make_http_msg()

        HttpHandler.handle(handler, msg)
        handler.context.logger.info.assert_called()

    def test_valid_request(self) -> None:
        """Test valid request is dispatched to handler."""
        handler = _make_http_handler()
        route_handler = MagicMock(__name__="test_handler")
        handler._get_handler = MagicMock(return_value=(route_handler, {"key": "val"}))
        dialogue = MagicMock()
        handler.context.http_dialogues.update.return_value = dialogue
        msg = _make_http_msg()

        HttpHandler.handle(handler, msg)
        route_handler.assert_called_once()

    def test_handler_exception_returns_500(self) -> None:
        """Test that an exception in a route handler returns 500 instead of crashing."""
        handler = _make_http_handler()
        route_handler = MagicMock(
            __name__="exploding_handler",
            side_effect=RuntimeError("boom"),
        )
        handler._get_handler = MagicMock(return_value=(route_handler, {}))
        dialogue = _make_http_dialogue()
        handler.context.http_dialogues.update.return_value = dialogue
        handler._handle_internal_server_error = MagicMock()
        msg = _make_http_msg()

        # Should not raise
        HttpHandler.handle(handler, msg)
        handler.context.logger.error.assert_called()
        handler._handle_internal_server_error.assert_called_once_with(msg, dialogue)


class TestHttpHandlerResponses:
    """Tests for response-sending methods."""

    def test_handle_bad_request(self) -> None:
        """Test _handle_bad_request sends 400."""
        handler = _make_http_handler()
        msg = _make_http_msg()
        dialogue = _make_http_dialogue()

        HttpHandler._handle_bad_request(handler, msg, dialogue, {"error": "bad"})

        dialogue.reply.assert_called_once()
        call_kwargs = dialogue.reply.call_args[1]
        assert call_kwargs["status_code"] == BAD_REQUEST_CODE
        handler.context.outbox.put_message.assert_called_once()

    def test_handle_bad_request_no_body(self) -> None:
        """Test _handle_bad_request with no body."""
        handler = _make_http_handler()
        msg = _make_http_msg()
        dialogue = _make_http_dialogue()

        HttpHandler._handle_bad_request(handler, msg, dialogue)

        dialogue.reply.assert_called_once()

    def test_handle_internal_server_error(self) -> None:
        """Test _handle_internal_server_error sends 500."""
        handler = _make_http_handler()
        msg = _make_http_msg()
        dialogue = _make_http_dialogue()

        HttpHandler._handle_internal_server_error(handler, msg, dialogue)

        dialogue.reply.assert_called_once()
        call_kwargs = dialogue.reply.call_args[1]
        assert call_kwargs["status_code"] == INTERNAL_SERVER_ERROR_CODE
        handler.context.outbox.put_message.assert_called_once()

    def test_send_ok_response_json(self) -> None:
        """Test _send_ok_response with dict data."""
        handler = _make_http_handler()
        msg = _make_http_msg()
        dialogue = _make_http_dialogue()

        HttpHandler._send_ok_response(handler, msg, dialogue, {"key": "val"})

        dialogue.reply.assert_called_once()
        call_kwargs = dialogue.reply.call_args[1]
        assert call_kwargs["status_code"] == OK_CODE
        assert call_kwargs["body"] == json.dumps({"key": "val"}).encode("utf-8")

    def test_send_ok_response_bytes(self) -> None:
        """Test _send_ok_response with bytes data."""
        handler = _make_http_handler()
        msg = _make_http_msg()
        dialogue = _make_http_dialogue()

        HttpHandler._send_ok_response(handler, msg, dialogue, b"binary_data")

        call_kwargs = dialogue.reply.call_args[1]
        assert call_kwargs["body"] == b"binary_data"

    def test_send_ok_response_bytes_with_content_type(self) -> None:
        """Test _send_ok_response with bytes and content_type."""
        handler = _make_http_handler()
        msg = _make_http_msg()
        dialogue = _make_http_dialogue()

        HttpHandler._send_ok_response(handler, msg, dialogue, b"data", "image/png")

        call_kwargs = dialogue.reply.call_args[1]
        assert "image/png" in call_kwargs["headers"]

    def test_send_ok_response_string(self) -> None:
        """Test _send_ok_response with string data."""
        handler = _make_http_handler()
        msg = _make_http_msg()
        dialogue = _make_http_dialogue()

        HttpHandler._send_ok_response(handler, msg, dialogue, "<html></html>")

        call_kwargs = dialogue.reply.call_args[1]
        assert call_kwargs["body"] == b"<html></html>"

    def test_send_ok_response_string_with_content_type(self) -> None:
        """Test _send_ok_response with string and explicit content_type."""
        handler = _make_http_handler()
        msg = _make_http_msg()
        dialogue = _make_http_dialogue()

        HttpHandler._send_ok_response(
            handler, msg, dialogue, "<html></html>", "text/html"
        )

        call_kwargs = dialogue.reply.call_args[1]
        assert "text/html" in call_kwargs["headers"]

    def test_send_ok_response_list(self) -> None:
        """Test _send_ok_response with list data."""
        handler = _make_http_handler()
        msg = _make_http_msg()
        dialogue = _make_http_dialogue()

        HttpHandler._send_ok_response(handler, msg, dialogue, [1, 2, 3])

        call_kwargs = dialogue.reply.call_args[1]
        assert call_kwargs["body"] == json.dumps([1, 2, 3]).encode("utf-8")

    def test_send_too_early_response(self) -> None:
        """Test _send_too_early_response."""
        handler = _make_http_handler()
        msg = _make_http_msg()
        dialogue = _make_http_dialogue()

        HttpHandler._send_too_early_response(handler, msg, dialogue, {"error": "wait"})

        call_kwargs = dialogue.reply.call_args[1]
        assert call_kwargs["status_code"] == TOO_EARLY_CODE

    def test_send_too_many_requests_response(self) -> None:
        """Test _send_too_many_requests_response."""
        handler = _make_http_handler()
        msg = _make_http_msg()
        dialogue = _make_http_dialogue()

        HttpHandler._send_too_many_requests_response(
            handler, msg, dialogue, {"error": "too many"}
        )

        call_kwargs = dialogue.reply.call_args[1]
        assert call_kwargs["status_code"] == TOO_MANY_REQUESTS_CODE

    def test_send_internal_server_error_response(self) -> None:
        """Test _send_internal_server_error_response."""
        handler = _make_http_handler()
        msg = _make_http_msg()
        dialogue = _make_http_dialogue()

        HttpHandler._send_internal_server_error_response(
            handler, msg, dialogue, {"error": "server error"}
        )

        call_kwargs = dialogue.reply.call_args[1]
        assert call_kwargs["status_code"] == INTERNAL_SERVER_ERROR_CODE

    def test_send_not_found_response(self) -> None:
        """Test _send_not_found_response."""
        handler = _make_http_handler()
        msg = _make_http_msg()
        dialogue = _make_http_dialogue()

        HttpHandler._send_not_found_response(handler, msg, dialogue)

        call_kwargs = dialogue.reply.call_args[1]
        assert call_kwargs["status_code"] == NOT_FOUND_CODE
        assert call_kwargs["body"] == b""


class TestHttpHandlerHealth:
    """Tests for _handle_get_health."""

    def _setup_handler(
        self,
        has_transition_timestamp: bool = True,
        has_abci_app: bool = True,
        is_stall_expired: bool = False,
    ) -> MagicMock:
        """Create handler with health-check setup."""
        handler = _make_http_handler()

        round_sequence = MagicMock()
        handler.context.state.round_sequence = round_sequence

        if has_transition_timestamp:
            round_sequence._last_round_transition_timestamp = datetime.now()
        else:
            round_sequence._last_round_transition_timestamp = None

        round_sequence.block_stall_deadline_expired = is_stall_expired

        if has_abci_app:
            mock_current_round = MagicMock()
            mock_current_round.round_id = "test_round"
            round_sequence._abci_app = MagicMock()
            round_sequence._abci_app.current_round = mock_current_round
            prev_round = MagicMock()
            prev_round.round_id = "prev_round"
            round_sequence._abci_app._previous_rounds = [prev_round]
        else:
            round_sequence._abci_app = None

        handler._send_ok_response = MagicMock()
        handler.synchronized_data = MagicMock()
        handler.synchronized_data.period_count = 5

        return handler

    def test_health_with_transition_and_abci(self) -> None:
        """Test health check with transition timestamp and abci app."""
        handler = self._setup_handler()
        msg = _make_http_msg()
        dialogue = _make_http_dialogue()

        HttpHandler._handle_get_health(handler, msg, dialogue)

        handler._send_ok_response.assert_called_once()
        data = handler._send_ok_response.call_args[0][2]
        assert data["rounds"] is not None
        assert data["is_tm_healthy"] is not None

    def test_health_no_transition_timestamp(self) -> None:
        """Test health check without transition timestamp."""
        handler = self._setup_handler(has_transition_timestamp=False)
        msg = _make_http_msg()
        dialogue = _make_http_dialogue()

        HttpHandler._handle_get_health(handler, msg, dialogue)

        data = handler._send_ok_response.call_args[0][2]
        assert data["seconds_since_last_transition"] is None
        assert data["is_tm_healthy"] is True  # not is_tm_unhealthy where None -> True

    def test_health_no_abci_app(self) -> None:
        """Test health check without abci app."""
        handler = self._setup_handler(has_abci_app=False)
        msg = _make_http_msg()
        dialogue = _make_http_dialogue()

        HttpHandler._handle_get_health(handler, msg, dialogue)

        data = handler._send_ok_response.call_args[0][2]
        assert data["rounds"] is None

    def test_health_stall_expired(self) -> None:
        """Test health check when TM is unhealthy."""
        handler = self._setup_handler(is_stall_expired=True)
        msg = _make_http_msg()
        dialogue = _make_http_dialogue()

        HttpHandler._handle_get_health(handler, msg, dialogue)

        data = handler._send_ok_response.call_args[0][2]
        assert data["is_tm_healthy"] is False


class TestHttpHandlerAgentDetails:
    """Tests for _handle_get_agent_details."""

    def test_with_agent_details(self) -> None:
        """Test with populated agent details."""
        handler = _make_http_handler()
        handler._db_connection_context = MagicMock()
        handler._db_connection_context.return_value.__enter__ = MagicMock()
        handler._db_connection_context.return_value.__exit__ = MagicMock()
        handler.db = MagicMock()
        handler.db.atomic.return_value.__enter__ = MagicMock()
        handler.db.atomic.return_value.__exit__ = MagicMock()
        handler._get_json_from_db = MagicMock(
            return_value={
                "safe_address": "0x123",
                "twitter_username": "test_user",
                "twitter_display_name": "Test User",
                "persona": "A fun persona",
            }
        )
        handler._send_ok_response = MagicMock()
        msg = _make_http_msg()
        dialogue = _make_http_dialogue()

        HttpHandler._handle_get_agent_details(handler, msg, dialogue)

        handler._send_ok_response.assert_called_once()
        data = handler._send_ok_response.call_args[0][2]
        assert data["address"] == "0x123"
        assert data["username"] == "test_user"

    def test_empty_agent_details(self) -> None:
        """Test with empty agent details."""
        handler = _make_http_handler()
        handler._db_connection_context = MagicMock()
        handler._db_connection_context.return_value.__enter__ = MagicMock()
        handler._db_connection_context.return_value.__exit__ = MagicMock()
        handler.db = MagicMock()
        handler.db.atomic.return_value.__enter__ = MagicMock()
        handler.db.atomic.return_value.__exit__ = MagicMock()
        handler._get_json_from_db = MagicMock(return_value={})
        handler._send_ok_response = MagicMock()
        msg = _make_http_msg()
        dialogue = _make_http_dialogue()

        HttpHandler._handle_get_agent_details(handler, msg, dialogue)

        handler._send_ok_response.assert_called_once()
        # Called with None when empty
        assert handler._send_ok_response.call_args[0][2] is None


class TestHttpHandlerXActivity:
    """Tests for _handle_get_recent_x_activity."""

    def _setup_handler(self, agent_actions: Dict) -> MagicMock:
        handler = _make_http_handler()
        handler._db_connection_context = MagicMock()
        handler._db_connection_context.return_value.__enter__ = MagicMock()
        handler._db_connection_context.return_value.__exit__ = MagicMock()
        handler.db = MagicMock()
        handler.db.atomic.return_value.__enter__ = MagicMock()
        handler.db.atomic.return_value.__exit__ = MagicMock()
        handler._get_json_from_db = MagicMock(return_value=agent_actions)
        handler._send_ok_response = MagicMock()
        return handler

    def test_no_tweet_actions(self) -> None:
        """Test when no tweet actions exist."""
        handler = self._setup_handler({})
        msg = _make_http_msg()
        dialogue = _make_http_dialogue()

        HttpHandler._handle_get_recent_x_activity(handler, msg, dialogue)

        handler._send_ok_response.assert_called_once()
        assert handler._send_ok_response.call_args[0][2] is None

    def test_tweet_action(self) -> None:
        """Test with regular tweet action."""
        handler = self._setup_handler(
            {
                "tweet_action": [
                    {
                        "action_type": "tweet",
                        "action_data": {"tweet_id": "123", "text": "Hello"},
                        "timestamp": "2024-01-01",
                    }
                ]
            }
        )
        msg = _make_http_msg()
        dialogue = _make_http_dialogue()

        HttpHandler._handle_get_recent_x_activity(handler, msg, dialogue)

        data = handler._send_ok_response.call_args[0][2]
        assert data["postId"] == "123"
        assert data["type"] == "tweet"

    def test_follow_action(self) -> None:
        """Test with follow action type."""
        handler = self._setup_handler(
            {
                "tweet_action": [
                    {
                        "action_type": "follow",
                        "action_data": {"username": "user1"},
                        "timestamp": "2024-01-01",
                    }
                ]
            }
        )
        msg = _make_http_msg()
        dialogue = _make_http_dialogue()

        HttpHandler._handle_get_recent_x_activity(handler, msg, dialogue)

        data = handler._send_ok_response.call_args[0][2]
        assert data["postId"] == "user1"
        assert data["type"] == "follow"

    def test_tweet_with_media(self) -> None:
        """Test tweet_with_media action type."""
        handler = self._setup_handler(
            {
                "tweet_action": [
                    {
                        "action_type": "tweet_with_media",
                        "action_data": {
                            "tweet_id": "456",
                            "text": "Check this out",
                            "media_ipfs_hash": "QmHash123",
                        },
                        "timestamp": "2024-01-01",
                    }
                ]
            }
        )
        msg = _make_http_msg()
        dialogue = _make_http_dialogue()

        HttpHandler._handle_get_recent_x_activity(handler, msg, dialogue)

        data = handler._send_ok_response.call_args[0][2]
        assert "media" in data
        assert "QmHash123" in data["media"][0]


class TestHttpHandlerMemeCoins:
    """Tests for _handle_get_meme_coins and _get_latest_token_activities."""

    def _setup_handler(self, agent_actions: Dict) -> MagicMock:
        handler = _make_http_handler()
        handler._db_connection_context = MagicMock()
        handler._db_connection_context.return_value.__enter__ = MagicMock()
        handler._db_connection_context.return_value.__exit__ = MagicMock()
        handler.db = MagicMock()
        handler.db.atomic.return_value.__enter__ = MagicMock()
        handler.db.atomic.return_value.__exit__ = MagicMock()
        handler._get_json_from_db = MagicMock(return_value=agent_actions)
        handler._send_ok_response = MagicMock()
        return handler

    def test_no_token_actions(self) -> None:
        """Test when no token actions exist."""
        handler = self._setup_handler({})
        msg = _make_http_msg()
        dialogue = _make_http_dialogue()

        HttpHandler._handle_get_meme_coins(handler, msg, dialogue)

        handler._send_ok_response.assert_called_once()

    def test_get_latest_token_activities_empty(self) -> None:
        """Test _get_latest_token_activities with no token actions."""
        handler = self._setup_handler({})

        result = HttpHandler._get_latest_token_activities(handler)
        assert result is None

    def test_get_latest_token_activities_with_data(self) -> None:
        """Test _get_latest_token_activities with valid data."""
        handler = self._setup_handler(
            {
                "token_action": [
                    {
                        "tweet_id": "t1",
                        "action": "buy",
                        "timestamp": "2024-01-01",
                        "token_address": "0xabc",
                        "token_nonce": 1,
                        "token_ticker": "MEME",
                    },
                ]
            }
        )

        result = HttpHandler._get_latest_token_activities(handler)
        assert result is not None
        assert len(result) == 1
        assert result[0]["type"] == "buy"

    def test_get_latest_token_activities_no_tweet_id(self) -> None:
        """Test _get_latest_token_activities skips entries without tweet_id."""
        handler = self._setup_handler(
            {
                "token_action": [
                    {
                        "action": "buy",
                        "timestamp": "2024-01-01",
                        "token_address": "0xabc",
                    },
                ]
            }
        )

        result = HttpHandler._get_latest_token_activities(handler)
        assert result is None

    def test_get_latest_token_activities_limit(self) -> None:
        """Test _get_latest_token_activities respects limit."""
        handler = self._setup_handler(
            {
                "token_action": [
                    {
                        "tweet_id": "t1",
                        "action": "buy",
                        "timestamp": "2024-01-01",
                        "token_address": "0x1",
                        "token_nonce": 1,
                        "token_ticker": "A",
                    },
                    {
                        "tweet_id": "t2",
                        "action": "sell",
                        "timestamp": "2024-01-02",
                        "token_address": "0x2",
                        "token_nonce": 2,
                        "token_ticker": "B",
                    },
                    {
                        "tweet_id": "t3",
                        "action": "buy",
                        "timestamp": "2024-01-03",
                        "token_address": "0x3",
                        "token_nonce": 3,
                        "token_ticker": "C",
                    },
                ]
            }
        )

        result = HttpHandler._get_latest_token_activities(handler, limit=2)
        assert result is not None
        assert len(result) == 2

    def test_get_latest_token_activities_no_token_address(self) -> None:
        """Test token activity where token_address is missing."""
        handler = self._setup_handler(
            {
                "token_action": [
                    {
                        "tweet_id": "t1",
                        "action": "buy",
                        "timestamp": "2024-01-01",
                        "token_nonce": 1,
                        "token_ticker": "MEME",
                    },
                ]
            }
        )

        result = HttpHandler._get_latest_token_activities(handler)
        assert result is not None
        assert result[0]["token"]["address"] is None


class TestHttpHandlerMedia:
    """Tests for _handle_get_media."""

    def _setup_handler(
        self,
        media_list: Any = None,
        agent_actions: Optional[Dict] = None,
    ) -> MagicMock:
        handler = _make_http_handler()
        handler._db_connection_context = MagicMock()
        handler._db_connection_context.return_value.__enter__ = MagicMock()
        handler._db_connection_context.return_value.__exit__ = MagicMock()
        handler.db = MagicMock()
        handler.db.atomic.return_value.__enter__ = MagicMock()
        handler.db.atomic.return_value.__exit__ = MagicMock()

        def mock_get_json(key: str, default: str = "{}") -> Any:
            if key == "media-store-list":
                return media_list if media_list is not None else []
            if key == "agent_actions":
                return agent_actions or {}
            return json.loads(default)

        handler._get_json_from_db = MagicMock(side_effect=mock_get_json)
        handler._send_ok_response = MagicMock()
        return handler

    def test_empty_media_list(self) -> None:
        """Test with empty media list."""
        handler = self._setup_handler(media_list=[])
        msg = _make_http_msg()
        dialogue = _make_http_dialogue()

        HttpHandler._handle_get_media(handler, msg, dialogue)

        handler._send_ok_response.assert_called_once()
        assert handler._send_ok_response.call_args[0][2] is None

    def test_media_with_matching_tweet(self) -> None:
        """Test media items that match tweet actions."""
        handler = self._setup_handler(
            media_list=[
                {"path": "/media/img.png", "ipfs_hash": "QmHash1"},
            ],
            agent_actions={
                "tweet_action": [
                    {
                        "action_data": {
                            "media_path": "/media/img.png",
                            "tweet_id": "tweet123",
                        }
                    }
                ]
            },
        )
        msg = _make_http_msg()
        dialogue = _make_http_dialogue()

        HttpHandler._handle_get_media(handler, msg, dialogue)

        handler._send_ok_response.assert_called_once()
        data = handler._send_ok_response.call_args[0][2]
        assert len(data) == 1
        assert data[0]["postId"] == "tweet123"

    def test_media_without_matching_tweet(self) -> None:
        """Test media items that don't match any tweet."""
        handler = self._setup_handler(
            media_list=[
                {"path": "/media/img.png", "ipfs_hash": "QmHash1"},
            ],
            agent_actions={"tweet_action": []},
        )
        msg = _make_http_msg()
        dialogue = _make_http_dialogue()

        HttpHandler._handle_get_media(handler, msg, dialogue)

        handler._send_ok_response.assert_called_once()
        data = handler._send_ok_response.call_args[0][2]
        assert data == []

    def test_tweet_action_missing_media_path(self) -> None:
        """Test tweet action without media_path is skipped in mapping."""
        handler = self._setup_handler(
            media_list=[
                {"path": "/media/img.png", "ipfs_hash": "QmHash1"},
            ],
            agent_actions={
                "tweet_action": [
                    {"action_data": {"tweet_id": "tweet123"}},
                ]
            },
        )
        msg = _make_http_msg()
        dialogue = _make_http_dialogue()

        HttpHandler._handle_get_media(handler, msg, dialogue)

        data = handler._send_ok_response.call_args[0][2]
        assert data == []

    def test_media_no_ipfs_hash(self) -> None:
        """Test media item without ipfs_hash is excluded."""
        handler = self._setup_handler(
            media_list=[
                {"path": "/media/img.png"},
            ],
            agent_actions={
                "tweet_action": [
                    {
                        "action_data": {
                            "media_path": "/media/img.png",
                            "tweet_id": "tweet123",
                        }
                    }
                ]
            },
        )
        msg = _make_http_msg()
        dialogue = _make_http_dialogue()

        HttpHandler._handle_get_media(handler, msg, dialogue)

        data = handler._send_ok_response.call_args[0][2]
        assert data == []


class TestHttpHandlerFeatures:
    """Tests for _handle_get_features."""

    def test_features_with_x402(self) -> None:
        """Test features when use_x402 is True."""
        handler = _make_http_handler()
        handler.context.params.use_x402 = True
        handler._send_ok_response = MagicMock()
        msg = _make_http_msg()
        dialogue = _make_http_dialogue()

        HttpHandler._handle_get_features(handler, msg, dialogue)

        data = handler._send_ok_response.call_args[0][2]
        assert data["isChatEnabled"] is True

    @pytest.mark.parametrize(
        "api_key,expected",
        [
            ("valid_key", True),
            (None, False),
            ("", False),
            ("   ", False),
            ("${str:}", False),
            ('""', False),
            (123, False),  # not a string
        ],
    )
    def test_features_without_x402(self, api_key: Any, expected: bool) -> None:
        """Test features without x402, checking various api_key values."""
        handler = _make_http_handler()
        handler.context.params.use_x402 = False
        handler.context.params.genai_api_key = api_key
        handler._send_ok_response = MagicMock()
        msg = _make_http_msg()
        dialogue = _make_http_dialogue()

        HttpHandler._handle_get_features(handler, msg, dialogue)

        data = handler._send_ok_response.call_args[0][2]
        assert data["isChatEnabled"] is expected


class TestHttpHandlerStaticFile:
    """Tests for _handle_get_static_file."""

    def test_serve_existing_file(self) -> None:
        """Test serving an existing static file."""
        handler = _make_http_handler()
        handler._send_ok_response = MagicMock()
        msg = _make_http_msg(url="http://localhost:8000/favicon.ico")
        dialogue = _make_http_dialogue()

        with (
            patch("pathlib.Path.is_file", return_value=True),
            patch("builtins.open", mock_open(read_data=b"icon_data")),
            patch("mimetypes.guess_type", return_value=("image/x-icon", None)),
        ):
            HttpHandler._handle_get_static_file(handler, msg, dialogue)

        handler._send_ok_response.assert_called_once()

    def test_serve_index_html_fallback(self) -> None:
        """Test SPA fallback to index.html."""
        handler = _make_http_handler()
        handler._send_ok_response = MagicMock()
        msg = _make_http_msg(url="http://localhost:8000/some/route")
        dialogue = _make_http_dialogue()

        call_count = 0

        def is_file_side_effect() -> bool:
            nonlocal call_count
            call_count += 1
            # First call: requested file doesn't exist; second: index.html exists
            return call_count > 1

        with (
            patch("pathlib.Path.is_file", side_effect=is_file_side_effect),
            patch("builtins.open", mock_open(read_data="<html>index</html>")),
        ):
            HttpHandler._handle_get_static_file(handler, msg, dialogue)

        handler._send_ok_response.assert_called_once()

    def test_no_index_html(self) -> None:
        """Test when neither file nor index.html exists."""
        handler = _make_http_handler()
        handler._send_not_found_response = MagicMock()
        msg = _make_http_msg(url="http://localhost:8000/missing")
        dialogue = _make_http_dialogue()

        with patch("pathlib.Path.is_file", return_value=False):
            HttpHandler._handle_get_static_file(handler, msg, dialogue)

        handler._send_not_found_response.assert_called_once()

    def test_unknown_content_type(self) -> None:
        """Test serving file with unknown content type."""
        handler = _make_http_handler()
        handler._send_ok_response = MagicMock()
        msg = _make_http_msg(url="http://localhost:8000/file.xyz")
        dialogue = _make_http_dialogue()

        with (
            patch("pathlib.Path.is_file", return_value=True),
            patch("builtins.open", mock_open(read_data=b"data")),
            patch("mimetypes.guess_type", return_value=(None, None)),
        ):
            HttpHandler._handle_get_static_file(handler, msg, dialogue)

        handler._send_ok_response.assert_called_once()
        call_args = handler._send_ok_response.call_args
        assert call_args[0][3] == "application/octet-stream"


class TestHttpHandlerSendMessage:
    """Tests for _send_message."""

    def test_send_message(self) -> None:
        """Test _send_message registers callback."""
        handler = _make_http_handler()
        handler.context.state.req_to_callback = {}

        message = MagicMock()
        dialogue = MagicMock()
        dialogue.dialogue_label.dialogue_reference = ("nonce42", "")
        callback = MagicMock()

        HttpHandler._send_message(handler, message, dialogue, callback, {"k": "v"})

        handler.context.outbox.put_message.assert_called_once_with(message=message)
        assert "nonce42" in handler.context.state.req_to_callback
        assert handler.context.state.req_to_callback["nonce42"] == (
            callback,
            {"k": "v"},
        )

    def test_send_message_no_kwargs(self) -> None:
        """Test _send_message without callback kwargs."""
        handler = _make_http_handler()
        handler.context.state.req_to_callback = {}

        message = MagicMock()
        dialogue = MagicMock()
        dialogue.dialogue_label.dialogue_reference = ("nonce43", "")
        callback = MagicMock()

        HttpHandler._send_message(handler, message, dialogue, callback)

        assert handler.context.state.req_to_callback["nonce43"] == (callback, {})


class TestHttpHandlerGetPromptAndSchema:
    """Tests for _get_prompt_and_schema."""

    def test_memecoin_enabled(self) -> None:
        """Test prompt when memecoin logic is enabled."""
        handler = _make_http_handler()
        handler.is_memecoin_logic_enabled = True
        handler._db_connection_context = MagicMock()
        handler._db_connection_context.return_value.__enter__ = MagicMock()
        handler._db_connection_context.return_value.__exit__ = MagicMock()
        handler.db = MagicMock()
        handler.db.atomic.return_value.__enter__ = MagicMock()
        handler.db.atomic.return_value.__exit__ = MagicMock()
        handler._get_value_from_db = MagicMock(return_value="test_value")

        with (
            patch(
                "packages.valory.skills.memeooorr_abci.handlers.CHATUI_PROMPT",
                "prompt {user_prompt} {current_persona} {current_heart_cooldown_hours} {current_summon_cooldown_seconds}",
            ),
            patch(
                "packages.valory.skills.memeooorr_abci.handlers.build_updated_agent_config_schema",
                return_value={"schema": True},
            ),
        ):
            prompt, schema = HttpHandler._get_prompt_and_schema(handler, "user input")

        assert "user input" in prompt
        assert schema == {"schema": True}

    def test_memecoin_disabled(self) -> None:
        """Test prompt when memecoin logic is disabled."""
        handler = _make_http_handler()
        handler.is_memecoin_logic_enabled = False
        handler._db_connection_context = MagicMock()
        handler._db_connection_context.return_value.__enter__ = MagicMock()
        handler._db_connection_context.return_value.__exit__ = MagicMock()
        handler.db = MagicMock()
        handler.db.atomic.return_value.__enter__ = MagicMock()
        handler.db.atomic.return_value.__exit__ = MagicMock()
        handler._get_value_from_db = MagicMock(return_value="test_persona")

        with (
            patch(
                "packages.valory.skills.memeooorr_abci.handlers.CHATUI_PROMPT_NO_MEMECOIN",
                "no_memecoin {user_prompt} {current_persona}",
            ),
            patch(
                "packages.valory.skills.memeooorr_abci.handlers.build_updated_agent_config_schema_no_memecoin",
                return_value={"schema_no_mc": True},
            ),
        ):
            prompt, schema = HttpHandler._get_prompt_and_schema(handler, "user input")

        assert "no_memecoin" in prompt
        assert schema == {"schema_no_mc": True}


class TestHttpHandlerPostProcessPrompt:
    """Tests for _handle_post_process_prompt."""

    def test_x402_not_ready(self) -> None:
        """Test POST when x402 is enabled but funds not ready."""
        handler = _make_http_handler()
        handler.context.params.use_x402 = True
        handler.shared_state = MagicMock()
        handler.shared_state.sufficient_funds_for_x402_payments = False
        handler._send_too_early_response = MagicMock()
        msg = _make_http_msg(
            method="post",
            body=json.dumps({"prompt": "test"}).encode(),
        )
        dialogue = _make_http_dialogue()

        HttpHandler._handle_post_process_prompt(handler, msg, dialogue)

        handler._send_too_early_response.assert_called_once()

    def test_empty_prompt(self) -> None:
        """Test POST with empty prompt."""
        handler = _make_http_handler()
        handler.context.params.use_x402 = False
        handler._handle_bad_request = MagicMock()
        msg = _make_http_msg(
            method="post",
            body=json.dumps({"prompt": ""}).encode(),
        )
        dialogue = _make_http_dialogue()

        HttpHandler._handle_post_process_prompt(handler, msg, dialogue)

        handler._handle_bad_request.assert_called_once()

    def test_no_prompt_field(self) -> None:
        """Test POST without prompt field."""
        handler = _make_http_handler()
        handler.context.params.use_x402 = False
        handler._handle_bad_request = MagicMock()
        msg = _make_http_msg(
            method="post",
            body=json.dumps({"other": "data"}).encode(),
        )
        dialogue = _make_http_dialogue()

        HttpHandler._handle_post_process_prompt(handler, msg, dialogue)

        handler._handle_bad_request.assert_called_once()

    def test_valid_prompt(self) -> None:
        """Test POST with valid prompt."""
        handler = _make_http_handler()
        handler.context.params.use_x402 = False
        handler._get_prompt_and_schema = MagicMock(
            return_value=("prompt", {"schema": True})
        )
        handler._send_chatui_llm_request = MagicMock()
        msg = _make_http_msg(
            method="post",
            body=json.dumps({"prompt": "Hello agent"}).encode(),
        )
        dialogue = _make_http_dialogue()

        HttpHandler._handle_post_process_prompt(handler, msg, dialogue)

        handler._send_chatui_llm_request.assert_called_once()

    def test_x402_funds_ready(self) -> None:
        """Test POST when x402 is enabled and funds are ready."""
        handler = _make_http_handler()
        handler.context.params.use_x402 = True
        handler.shared_state = MagicMock()
        handler.shared_state.sufficient_funds_for_x402_payments = True
        handler._get_prompt_and_schema = MagicMock(
            return_value=("prompt", {"schema": True})
        )
        handler._send_chatui_llm_request = MagicMock()
        msg = _make_http_msg(
            method="post",
            body=json.dumps({"prompt": "Hello agent"}).encode(),
        )
        dialogue = _make_http_dialogue()

        HttpHandler._handle_post_process_prompt(handler, msg, dialogue)

        handler._send_chatui_llm_request.assert_called_once()


class TestHttpHandlerSendChatUILlmRequest:  # pylint: disable=too-few-public-methods
    """Tests for _send_chatui_llm_request."""

    def test_send_request(self) -> None:
        """Test sending LLM request through SRR."""
        handler = _make_http_handler()
        handler._send_message = MagicMock()

        srr_dialogues = MagicMock()
        srr_dialogues.create.return_value = (MagicMock(), MagicMock())
        handler.context.srr_dialogues = srr_dialogues

        msg = _make_http_msg()
        dialogue = _make_http_dialogue()

        HttpHandler._send_chatui_llm_request(
            handler,
            prompt="test prompt",
            schema={"s": True},
            http_msg=msg,
            http_dialogue=dialogue,
        )

        handler._send_message.assert_called_once()


class TestHttpHandlerLlmResponse:
    """Tests for _handle_llm_response."""

    def _make_llm_response(self, payload: Dict) -> MagicMock:
        """Create mock LLM response."""
        msg = MagicMock(spec=SrrMessage)
        msg.payload = json.dumps(payload)
        return msg

    def test_error_response(self) -> None:
        """Test handling LLM error response."""
        handler = _make_http_handler()
        handler._handle_chatui_llm_error = MagicMock()
        msg = _make_http_msg()
        dialogue = _make_http_dialogue()

        llm_msg = self._make_llm_response({"error": "something broke"})

        HttpHandler._handle_llm_response(handler, llm_msg, MagicMock(), msg, dialogue)

        handler._handle_chatui_llm_error.assert_called_once_with(
            "something broke", msg, dialogue
        )

    def test_response_with_updated_persona(self) -> None:
        """Test response that updates persona."""
        handler = _make_http_handler()
        handler._send_ok_response = MagicMock()
        handler._db_connection_context = MagicMock()
        handler._db_connection_context.return_value.__enter__ = MagicMock()
        handler._db_connection_context.return_value.__exit__ = MagicMock()
        handler.db = MagicMock()
        handler.db.atomic.return_value.__enter__ = MagicMock()
        handler.db.atomic.return_value.__exit__ = MagicMock()
        handler._set_value_to_db = MagicMock()
        handler._get_json_from_db = MagicMock(return_value={"persona": "old"})
        handler.shared_state = MagicMock()
        msg = _make_http_msg()
        dialogue = _make_http_dialogue()

        llm_response = json.dumps(
            {
                "agent_persona": "new persona",
                "message": "Updated!",
            }
        )
        llm_msg = self._make_llm_response({"response": llm_response})

        HttpHandler._handle_llm_response(handler, llm_msg, MagicMock(), msg, dialogue)

        handler._send_ok_response.assert_called_once()
        call_data = handler._send_ok_response.call_args[0][2]
        assert "persona" in call_data["updated_params"]

    def test_response_no_updates(self) -> None:
        """Test response with no parameter updates."""
        handler = _make_http_handler()
        handler._send_ok_response = MagicMock()
        msg = _make_http_msg()
        dialogue = _make_http_dialogue()

        llm_response = json.dumps(
            {
                "message": "No changes needed",
            }
        )
        llm_msg = self._make_llm_response({"response": llm_response})

        HttpHandler._handle_llm_response(handler, llm_msg, MagicMock(), msg, dialogue)

        call_data = handler._send_ok_response.call_args[0][2]
        assert call_data["updated_params"] == {}
        assert call_data["message"] == "No Params updated"

    def test_response_with_heart_cooldown_above_min(self) -> None:
        """Test updating heart_cooldown_hours above minimum."""
        handler = _make_http_handler()
        handler._send_ok_response = MagicMock()
        handler._db_connection_context = MagicMock()
        handler._db_connection_context.return_value.__enter__ = MagicMock()
        handler._db_connection_context.return_value.__exit__ = MagicMock()
        handler.db = MagicMock()
        handler.db.atomic.return_value.__enter__ = MagicMock()
        handler.db.atomic.return_value.__exit__ = MagicMock()
        handler._set_value_to_db = MagicMock()
        msg = _make_http_msg()
        dialogue = _make_http_dialogue()

        llm_response = json.dumps(
            {
                "heart_cooldown_hours": 48,
                "message": "Updated cooldown",
            }
        )
        llm_msg = self._make_llm_response({"response": llm_response})

        HttpHandler._handle_llm_response(handler, llm_msg, MagicMock(), msg, dialogue)

        call_data = handler._send_ok_response.call_args[0][2]
        assert call_data["updated_params"]["heart_cooldown_hours"] == 48

    def test_response_with_heart_cooldown_below_min(self) -> None:
        """Test heart_cooldown_hours below 24 gets clamped to 24."""
        handler = _make_http_handler()
        handler._send_ok_response = MagicMock()
        handler._db_connection_context = MagicMock()
        handler._db_connection_context.return_value.__enter__ = MagicMock()
        handler._db_connection_context.return_value.__exit__ = MagicMock()
        handler.db = MagicMock()
        handler.db.atomic.return_value.__enter__ = MagicMock()
        handler.db.atomic.return_value.__exit__ = MagicMock()
        handler._set_value_to_db = MagicMock()
        msg = _make_http_msg()
        dialogue = _make_http_dialogue()

        llm_response = json.dumps(
            {
                "heart_cooldown_hours": 12,
                "message": "Too low",
            }
        )
        llm_msg = self._make_llm_response({"response": llm_response})

        HttpHandler._handle_llm_response(handler, llm_msg, MagicMock(), msg, dialogue)

        call_data = handler._send_ok_response.call_args[0][2]
        assert call_data["updated_params"]["heart_cooldown_hours"] == 24

    def test_response_with_summon_cooldown_above_min(self) -> None:
        """Test updating summon_cooldown_seconds above minimum."""
        handler = _make_http_handler()
        handler._send_ok_response = MagicMock()
        handler._db_connection_context = MagicMock()
        handler._db_connection_context.return_value.__enter__ = MagicMock()
        handler._db_connection_context.return_value.__exit__ = MagicMock()
        handler.db = MagicMock()
        handler.db.atomic.return_value.__enter__ = MagicMock()
        handler.db.atomic.return_value.__exit__ = MagicMock()
        handler._set_value_to_db = MagicMock()
        msg = _make_http_msg()
        dialogue = _make_http_dialogue()

        llm_response = json.dumps(
            {
                "summon_cooldown_seconds": 3000000,
                "message": "Updated",
            }
        )
        llm_msg = self._make_llm_response({"response": llm_response})

        HttpHandler._handle_llm_response(handler, llm_msg, MagicMock(), msg, dialogue)

        call_data = handler._send_ok_response.call_args[0][2]
        assert call_data["updated_params"]["summon_cooldown_seconds"] == 3000000

    def test_response_with_summon_cooldown_below_min(self) -> None:
        """Test summon_cooldown_seconds below 2592000 gets clamped."""
        handler = _make_http_handler()
        handler._send_ok_response = MagicMock()
        handler._db_connection_context = MagicMock()
        handler._db_connection_context.return_value.__enter__ = MagicMock()
        handler._db_connection_context.return_value.__exit__ = MagicMock()
        handler.db = MagicMock()
        handler.db.atomic.return_value.__enter__ = MagicMock()
        handler.db.atomic.return_value.__exit__ = MagicMock()
        handler._set_value_to_db = MagicMock()
        msg = _make_http_msg()
        dialogue = _make_http_dialogue()

        llm_response = json.dumps(
            {
                "summon_cooldown_seconds": 100,
                "message": "Too low",
            }
        )
        llm_msg = self._make_llm_response({"response": llm_response})

        HttpHandler._handle_llm_response(handler, llm_msg, MagicMock(), msg, dialogue)

        call_data = handler._send_ok_response.call_args[0][2]
        assert call_data["updated_params"]["summon_cooldown_seconds"] == 2592000


class TestHttpHandlerChatUILlmError:
    """Tests for _handle_chatui_llm_error."""

    def test_api_key_not_set_error(self) -> None:
        """Test handling GENAI API key not set error."""
        handler = _make_http_handler()
        handler._send_internal_server_error_response = MagicMock()
        msg = _make_http_msg()
        dialogue = _make_http_dialogue()

        HttpHandler._handle_chatui_llm_error(
            handler, GENAI_API_KEY_NOT_SET_ERROR, msg, dialogue
        )

        handler._send_internal_server_error_response.assert_called_once()

    def test_rate_limit_error(self) -> None:
        """Test handling rate limit error."""
        handler = _make_http_handler()
        handler._send_too_many_requests_response = MagicMock()
        msg = _make_http_msg()
        dialogue = _make_http_dialogue()

        HttpHandler._handle_chatui_llm_error(
            handler, f"Error: {GENAI_RATE_LIMIT_ERROR} rate limited", msg, dialogue
        )

        handler._send_too_many_requests_response.assert_called_once()

    def test_generic_error(self) -> None:
        """Test handling generic error."""
        handler = _make_http_handler()
        handler._send_internal_server_error_response = MagicMock()
        msg = _make_http_msg()
        dialogue = _make_http_dialogue()

        HttpHandler._handle_chatui_llm_error(
            handler, "Some unknown error", msg, dialogue
        )

        handler._send_internal_server_error_response.assert_called_once()


class TestHttpHandlerFundsStatus:
    """Tests for _handle_get_funds_status."""

    def test_funds_status_without_x402(self) -> None:
        """Test funds status when x402 is disabled."""
        handler = _make_http_handler()
        handler.params = MagicMock()
        handler.params.use_x402 = False
        handler.funds_status = MagicMock()
        handler.funds_status.get_response_body.return_value = {"status": "ok"}
        handler._send_ok_response = MagicMock()
        msg = _make_http_msg()
        dialogue = _make_http_dialogue()

        HttpHandler._handle_get_funds_status(handler, msg, dialogue)

        handler.executor.submit.assert_not_called()
        handler._send_ok_response.assert_called_once()

    def test_funds_status_with_x402(self) -> None:
        """Test funds status when x402 is enabled."""
        handler = _make_http_handler()
        handler.params = MagicMock()
        handler.params.use_x402 = True
        handler._x402_swap_future = None
        handler._submit_x402_swap_if_idle = (
            lambda: HttpHandler._submit_x402_swap_if_idle(handler)
        )
        handler.funds_status = MagicMock()
        handler.funds_status.get_response_body.return_value = {"status": "ok"}
        handler._send_ok_response = MagicMock()
        msg = _make_http_msg()
        dialogue = _make_http_dialogue()

        HttpHandler._handle_get_funds_status(handler, msg, dialogue)

        handler.executor.submit.assert_called_once_with(
            handler._ensure_sufficient_funds_for_x402_payments
        )
        assert handler._x402_swap_future is not None
        handler._send_ok_response.assert_called_once()

    def test_funds_status_x402_dedup(self) -> None:
        """Test that duplicate x402 swap submissions are skipped."""
        handler = _make_http_handler()
        handler.params = MagicMock()
        handler.params.use_x402 = True
        handler.funds_status = MagicMock()
        handler.funds_status.get_response_body.return_value = {"status": "ok"}
        handler._send_ok_response = MagicMock()
        handler.context.logger = MagicMock()
        # Simulate a future already in progress
        running_future = MagicMock()
        running_future.done.return_value = False
        handler._x402_swap_future = running_future
        handler._submit_x402_swap_if_idle = (
            lambda: HttpHandler._submit_x402_swap_if_idle(handler)
        )
        msg = _make_http_msg()
        dialogue = _make_http_dialogue()

        HttpHandler._handle_get_funds_status(handler, msg, dialogue)

        handler.executor.submit.assert_not_called()
        handler._send_ok_response.assert_called_once()


class TestGetEoaAccount:
    """Tests for _get_eoa_account."""

    def test_with_password(self) -> None:
        """Test getting EOA account with password."""
        handler = _make_http_handler()

        mock_crypto = MagicMock()
        mock_crypto.private_key = "0x" + "a" * 64

        with (
            patch(
                "packages.valory.skills.memeooorr_abci.handlers.get_password_from_args",
                return_value="pass123",
            ),
            patch(
                "packages.valory.skills.memeooorr_abci.handlers.EthereumCrypto",
                return_value=mock_crypto,
            ),
            patch(
                "packages.valory.skills.memeooorr_abci.handlers.Account.from_key",
            ) as mock_from_key,
        ):
            mock_from_key.return_value = MagicMock()
            result = HttpHandler._get_eoa_account(handler)
            assert result is not None

    def test_without_password(self) -> None:
        """Test getting EOA account without password (plaintext key)."""
        handler = _make_http_handler()

        with (
            patch(
                "packages.valory.skills.memeooorr_abci.handlers.get_password_from_args",
                return_value=None,
            ),
            patch(
                "pathlib.Path.open",
                mock_open(read_data="0x" + "b" * 64),
            ),
            patch(
                "packages.valory.skills.memeooorr_abci.handlers.Account.from_key",
            ) as mock_from_key,
        ):
            mock_from_key.return_value = MagicMock()
            result = HttpHandler._get_eoa_account(handler)
            assert result is not None

    def test_invalid_key(self) -> None:
        """Test when key is invalid."""
        handler = _make_http_handler()

        with (
            patch(
                "packages.valory.skills.memeooorr_abci.handlers.get_password_from_args",
                return_value=None,
            ),
            patch(
                "pathlib.Path.open",
                mock_open(read_data="invalid_key"),
            ),
            patch(
                "packages.valory.skills.memeooorr_abci.handlers.Account.from_key",
                side_effect=Exception("Invalid key"),
            ),
        ):
            result = HttpHandler._get_eoa_account(handler)
            assert result is None
            handler.context.logger.error.assert_called()


class TestGetWeb3Instance:
    """Tests for _get_web3_instance."""

    def test_success(self) -> None:
        """Test successful Web3 instance creation."""
        handler = _make_http_handler()
        handler.params = MagicMock()
        handler.params.base_ledger_rpc = "http://localhost:8545"

        with patch("packages.valory.skills.memeooorr_abci.handlers.Web3"):
            result = HttpHandler._get_web3_instance(handler, "base")
            assert result is not None

    def test_no_rpc_url(self) -> None:
        """Test when RPC URL is empty."""
        handler = _make_http_handler()
        handler.params = MagicMock()
        handler.params.base_ledger_rpc = ""

        result = HttpHandler._get_web3_instance(handler, "base")
        assert result is None

    def test_exception(self) -> None:
        """Test when Web3 creation raises exception."""
        handler = _make_http_handler()
        handler.params = MagicMock()
        handler.params.base_ledger_rpc = "http://localhost:8545"

        with patch(
            "packages.valory.skills.memeooorr_abci.handlers.Web3",
            side_effect=Exception("connection error"),
        ):
            result = HttpHandler._get_web3_instance(handler, "base")
            assert result is None


class TestCheckUsdcBalance:
    """Tests for _check_usdc_balance."""

    VALID_ADDRESS = "0x" + "a" * 40

    def test_success(self) -> None:
        """Test successful balance check."""
        handler = _make_http_handler()
        mock_w3 = MagicMock()
        mock_contract = MagicMock()
        mock_contract.functions.balanceOf.return_value.call.return_value = 1000000
        mock_w3.eth.contract.return_value = mock_contract
        handler._get_web3_instance = MagicMock(return_value=mock_w3)

        with patch(
            "packages.valory.skills.memeooorr_abci.handlers.Web3.to_checksum_address",
            return_value=self.VALID_ADDRESS,
        ):
            result = HttpHandler._check_usdc_balance(
                handler, self.VALID_ADDRESS, "base", USDC_ADDRESS
            )
        assert result == 1000000

    def test_no_web3(self) -> None:
        """Test when Web3 instance unavailable."""
        handler = _make_http_handler()
        handler._get_web3_instance = MagicMock(return_value=None)

        result = HttpHandler._check_usdc_balance(
            handler, self.VALID_ADDRESS, "base", USDC_ADDRESS
        )
        assert result is None

    def test_exception(self) -> None:
        """Test when balance check raises exception."""
        handler = _make_http_handler()
        handler._get_web3_instance = MagicMock(side_effect=Exception("err"))

        result = HttpHandler._check_usdc_balance(
            handler, self.VALID_ADDRESS, "base", USDC_ADDRESS
        )
        assert result is None


class TestGetLifiQuoteSync:
    """Tests for _get_lifi_quote_sync."""

    def test_success(self) -> None:
        """Test successful LiFi quote."""
        handler = _make_http_handler()
        handler.params = MagicMock()
        handler.params.lifi_quote_to_amount_url = "https://li.quest/v1/quote"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"transactionRequest": {"to": "0x1"}}

        with patch(
            "packages.valory.skills.memeooorr_abci.handlers.requests.get",
            return_value=mock_response,
        ):
            result = HttpHandler._get_lifi_quote_sync(
                handler, "0xeoa", USDC_ADDRESS, "5000000"
            )
            assert result is not None

    def test_non_200_status(self) -> None:
        """Test non-200 response from LiFi."""
        handler = _make_http_handler()
        handler.params = MagicMock()
        handler.params.lifi_quote_to_amount_url = "https://li.quest/v1/quote"

        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch(
            "packages.valory.skills.memeooorr_abci.handlers.requests.get",
            return_value=mock_response,
        ):
            result = HttpHandler._get_lifi_quote_sync(
                handler, "0xeoa", USDC_ADDRESS, "5000000"
            )
            assert result is None

    def test_exception(self) -> None:
        """Test exception during LiFi request."""
        handler = _make_http_handler()
        handler.params = MagicMock()
        handler.params.lifi_quote_to_amount_url = "https://li.quest/v1/quote"

        with patch(
            "packages.valory.skills.memeooorr_abci.handlers.requests.get",
            side_effect=Exception("network error"),
        ):
            result = HttpHandler._get_lifi_quote_sync(
                handler, "0xeoa", USDC_ADDRESS, "5000000"
            )
            assert result is None


class TestSignAndSubmitTxWeb3:
    """Tests for _sign_and_submit_tx_web3."""

    def test_success(self) -> None:
        """Test successful tx submission."""
        handler = _make_http_handler()
        mock_w3 = MagicMock()
        mock_hash = MagicMock()
        mock_hash.to_0x_hex.return_value = "0xdeadbeef"
        mock_w3.eth.send_raw_transaction.return_value = mock_hash
        handler._get_web3_instance = MagicMock(return_value=mock_w3)

        eoa_account = MagicMock()

        result = HttpHandler._sign_and_submit_tx_web3(
            handler, {"to": "0x1", "data": "0x", "value": 0}, "base", eoa_account
        )
        assert result == "0xdeadbeef"

    def test_no_web3(self) -> None:
        """Test when Web3 is unavailable."""
        handler = _make_http_handler()
        handler._get_web3_instance = MagicMock(return_value=None)

        result = HttpHandler._sign_and_submit_tx_web3(handler, {}, "base", MagicMock())
        assert result is None

    def test_exception(self) -> None:
        """Test when submission raises exception."""
        handler = _make_http_handler()
        mock_w3 = MagicMock()
        handler._get_web3_instance = MagicMock(return_value=mock_w3)
        eoa_account = MagicMock()
        eoa_account.sign_transaction.side_effect = Exception("sign error")

        result = HttpHandler._sign_and_submit_tx_web3(
            handler, {"to": "0x1"}, "base", eoa_account
        )
        assert result is None


class TestCheckTransactionStatus:
    """Tests for _check_transaction_status."""

    def test_successful_transaction(self) -> None:
        """Test successful transaction receipt."""
        handler = _make_http_handler()
        mock_w3 = MagicMock()
        receipt = MagicMock()
        receipt.status = 1
        mock_w3.eth.wait_for_transaction_receipt.return_value = receipt
        handler._get_web3_instance = MagicMock(return_value=mock_w3)

        result = HttpHandler._check_transaction_status(handler, "0xhash", "base")
        assert result is True

    def test_failed_transaction(self) -> None:
        """Test failed transaction receipt."""
        handler = _make_http_handler()
        mock_w3 = MagicMock()
        receipt = MagicMock()
        receipt.status = 0
        mock_w3.eth.wait_for_transaction_receipt.return_value = receipt
        handler._get_web3_instance = MagicMock(return_value=mock_w3)

        result = HttpHandler._check_transaction_status(handler, "0xhash", "base")
        assert result is False

    def test_no_web3(self) -> None:
        """Test when Web3 is unavailable."""
        handler = _make_http_handler()
        handler._get_web3_instance = MagicMock(return_value=None)

        result = HttpHandler._check_transaction_status(handler, "0xhash", "base")
        assert result is False

    def test_exception(self) -> None:
        """Test exception during status check."""
        handler = _make_http_handler()
        handler._get_web3_instance = MagicMock(side_effect=Exception("timeout"))

        result = HttpHandler._check_transaction_status(handler, "0xhash", "base")
        assert result is False


class TestGetNonceAndGasWeb3:
    """Tests for _get_nonce_and_gas_web3."""

    VALID_ADDRESS = "0x" + "a" * 40

    def test_success(self) -> None:
        """Test successful nonce and gas retrieval."""
        handler = _make_http_handler()
        mock_w3 = MagicMock()
        mock_w3.eth.get_transaction_count.return_value = 42
        mock_w3.eth.gas_price = 20000000000
        handler._get_web3_instance = MagicMock(return_value=mock_w3)

        with patch(
            "packages.valory.skills.memeooorr_abci.handlers.Web3.to_checksum_address",
            return_value=self.VALID_ADDRESS,
        ):
            nonce, gas = HttpHandler._get_nonce_and_gas_web3(
                handler, self.VALID_ADDRESS, "base"
            )
        assert nonce == 42
        assert gas == 20000000000

    def test_no_web3(self) -> None:
        """Test when Web3 is unavailable."""
        handler = _make_http_handler()
        handler._get_web3_instance = MagicMock(return_value=None)

        nonce, gas = HttpHandler._get_nonce_and_gas_web3(
            handler, self.VALID_ADDRESS, "base"
        )
        assert nonce is None
        assert gas is None

    def test_exception(self) -> None:
        """Test exception during nonce/gas retrieval."""
        handler = _make_http_handler()
        handler._get_web3_instance = MagicMock(side_effect=Exception("rpc error"))

        nonce, gas = HttpHandler._get_nonce_and_gas_web3(
            handler, self.VALID_ADDRESS, "base"
        )
        assert nonce is None
        assert gas is None


class TestEstimateGas:
    """Tests for _estimate_gas."""

    VALID_ADDRESS = "0x" + "a" * 40

    def test_success_hex_value(self) -> None:
        """Test successful gas estimation with hex value."""
        handler = _make_http_handler()
        mock_w3 = MagicMock()
        mock_w3.eth.estimate_gas.return_value = 100000
        handler._get_web3_instance = MagicMock(return_value=mock_w3)

        tx_request = {"to": self.VALID_ADDRESS, "data": "0x", "value": "0x1000"}

        with patch(
            "packages.valory.skills.memeooorr_abci.handlers.Web3.to_checksum_address",
            side_effect=lambda x: x,
        ):
            result = HttpHandler._estimate_gas(
                handler, tx_request, self.VALID_ADDRESS, "base"
            )
        assert result == int(100000 * 1.2)

    def test_success_int_value(self) -> None:
        """Test successful gas estimation with int value."""
        handler = _make_http_handler()
        mock_w3 = MagicMock()
        mock_w3.eth.estimate_gas.return_value = 50000
        handler._get_web3_instance = MagicMock(return_value=mock_w3)

        tx_request = {"to": self.VALID_ADDRESS, "data": "0x", "value": 4096}

        with patch(
            "packages.valory.skills.memeooorr_abci.handlers.Web3.to_checksum_address",
            side_effect=lambda x: x,
        ):
            result = HttpHandler._estimate_gas(
                handler, tx_request, self.VALID_ADDRESS, "base"
            )
        assert result == int(50000 * 1.2)

    def test_no_web3(self) -> None:
        """Test when Web3 is unavailable."""
        handler = _make_http_handler()
        handler._get_web3_instance = MagicMock(return_value=None)

        result = HttpHandler._estimate_gas(
            handler, {"to": "0x1", "data": "0x", "value": 0}, "0xeoa", "base"
        )
        assert result is None

    def test_exception(self) -> None:
        """Test exception during gas estimation."""
        handler = _make_http_handler()
        mock_w3 = MagicMock()
        mock_w3.eth.estimate_gas.side_effect = Exception("estimation failed")
        handler._get_web3_instance = MagicMock(return_value=mock_w3)

        tx_request = {"to": "0x1", "data": "0x", "value": 0}

        with patch(
            "packages.valory.skills.memeooorr_abci.handlers.Web3.to_checksum_address",
            side_effect=lambda x: x,
        ):
            result = HttpHandler._estimate_gas(handler, tx_request, "0xeoa", "base")
        assert result is None


class TestEnsureSufficientFundsForX402:
    """Tests for _ensure_sufficient_funds_for_x402_payments."""

    def test_no_eoa_account(self) -> None:
        """Test when EOA account cannot be obtained."""
        handler = _make_http_handler()
        handler._get_eoa_account = MagicMock(return_value=None)
        handler.shared_state = MagicMock()

        result = HttpHandler._ensure_sufficient_funds_for_x402_payments(handler)
        assert result is False

    def test_balance_check_returns_none(self) -> None:
        """Test when USDC balance check returns None (assumes sufficient)."""
        handler = _make_http_handler()
        handler._get_eoa_account = MagicMock()
        handler._get_eoa_account.return_value.address = "0xeoa"
        handler._check_usdc_balance = MagicMock(return_value=None)
        handler.shared_state = MagicMock()
        handler.params = MagicMock()
        handler.params.x402_payment_requirements = {
            "threshold": 1000000,
            "top_up": 5000000,
        }

        result = HttpHandler._ensure_sufficient_funds_for_x402_payments(handler)
        assert result is True
        assert handler.shared_state.sufficient_funds_for_x402_payments is True

    def test_sufficient_balance(self) -> None:
        """Test when balance is above threshold."""
        handler = _make_http_handler()
        handler._get_eoa_account = MagicMock()
        handler._get_eoa_account.return_value.address = "0xeoa"
        handler._check_usdc_balance = MagicMock(return_value=2000000)
        handler.shared_state = MagicMock()
        handler.params = MagicMock()
        handler.params.x402_payment_requirements = {
            "threshold": 1000000,
            "top_up": 5000000,
        }

        result = HttpHandler._ensure_sufficient_funds_for_x402_payments(handler)
        assert result is True

    def test_insufficient_balance_no_quote(self) -> None:
        """Test when balance is low and LiFi quote fails."""
        handler = _make_http_handler()
        handler._get_eoa_account = MagicMock()
        handler._get_eoa_account.return_value.address = "0xeoa"
        handler._check_usdc_balance = MagicMock(return_value=100)
        handler._get_lifi_quote_sync = MagicMock(return_value=None)
        handler.shared_state = MagicMock()
        handler.params = MagicMock()
        handler.params.x402_payment_requirements = {
            "threshold": 1000000,
            "top_up": 5000000,
        }

        result = HttpHandler._ensure_sufficient_funds_for_x402_payments(handler)
        assert result is False

    def test_insufficient_balance_no_tx_request(self) -> None:
        """Test when quote has no transactionRequest."""
        handler = _make_http_handler()
        handler._get_eoa_account = MagicMock()
        handler._get_eoa_account.return_value.address = "0xeoa"
        handler._check_usdc_balance = MagicMock(return_value=100)
        handler._get_lifi_quote_sync = MagicMock(return_value={"other": "data"})
        handler.shared_state = MagicMock()
        handler.params = MagicMock()
        handler.params.x402_payment_requirements = {
            "threshold": 1000000,
            "top_up": 5000000,
        }

        result = HttpHandler._ensure_sufficient_funds_for_x402_payments(handler)
        assert result is False

    def test_insufficient_balance_nonce_fails(self) -> None:
        """Test when nonce/gas retrieval fails."""
        handler = _make_http_handler()
        handler._get_eoa_account = MagicMock()
        handler._get_eoa_account.return_value.address = "0xeoa"
        handler._check_usdc_balance = MagicMock(return_value=100)
        handler._get_lifi_quote_sync = MagicMock(
            return_value={
                "transactionRequest": {"to": "0x1", "data": "0x", "value": "0x100"}
            }
        )
        handler._get_nonce_and_gas_web3 = MagicMock(return_value=(None, None))
        handler.shared_state = MagicMock()
        handler.params = MagicMock()
        handler.params.x402_payment_requirements = {
            "threshold": 1000000,
            "top_up": 5000000,
        }

        result = HttpHandler._ensure_sufficient_funds_for_x402_payments(handler)
        assert result is False

    def test_insufficient_balance_gas_estimation_fails(self) -> None:
        """Test when gas estimation fails."""
        handler = _make_http_handler()
        handler._get_eoa_account = MagicMock()
        handler._get_eoa_account.return_value.address = "0xeoa"
        handler._check_usdc_balance = MagicMock(return_value=100)
        handler._get_lifi_quote_sync = MagicMock(
            return_value={
                "transactionRequest": {"to": "0x1", "data": "0x", "value": "0x100"}
            }
        )
        handler._get_nonce_and_gas_web3 = MagicMock(return_value=(1, 20000000000))
        handler._estimate_gas = MagicMock(return_value=None)
        handler.shared_state = MagicMock()
        handler.params = MagicMock()
        handler.params.x402_payment_requirements = {
            "threshold": 1000000,
            "top_up": 5000000,
        }

        result = HttpHandler._ensure_sufficient_funds_for_x402_payments(handler)
        assert result is False

    def test_insufficient_balance_tx_submit_fails(self) -> None:
        """Test when tx submission fails."""
        handler = _make_http_handler()
        handler._get_eoa_account = MagicMock()
        handler._get_eoa_account.return_value.address = "0xeoa"
        handler._check_usdc_balance = MagicMock(return_value=100)
        handler._get_lifi_quote_sync = MagicMock(
            return_value={
                "transactionRequest": {"to": "0x1", "data": "0x", "value": "0x100"}
            }
        )
        handler._get_nonce_and_gas_web3 = MagicMock(return_value=(1, 20000000000))
        handler._estimate_gas = MagicMock(return_value=200000)
        handler._sign_and_submit_tx_web3 = MagicMock(return_value=None)
        handler.shared_state = MagicMock()
        handler.params = MagicMock()
        handler.params.x402_payment_requirements = {
            "threshold": 1000000,
            "top_up": 5000000,
        }

        with patch(
            "packages.valory.skills.memeooorr_abci.handlers.Web3.to_checksum_address",
            side_effect=lambda x: x,
        ):
            result = HttpHandler._ensure_sufficient_funds_for_x402_payments(handler)
        assert result is False

    def test_insufficient_balance_tx_fails(self) -> None:
        """Test when tx submission succeeds but transaction fails."""
        handler = _make_http_handler()
        handler._get_eoa_account = MagicMock()
        handler._get_eoa_account.return_value.address = "0xeoa"
        handler._check_usdc_balance = MagicMock(return_value=100)
        handler._get_lifi_quote_sync = MagicMock(
            return_value={
                "transactionRequest": {"to": "0x1", "data": "0x", "value": "0x100"}
            }
        )
        handler._get_nonce_and_gas_web3 = MagicMock(return_value=(1, 20000000000))
        handler._estimate_gas = MagicMock(return_value=200000)
        handler._sign_and_submit_tx_web3 = MagicMock(return_value="0xhash")
        handler._check_transaction_status = MagicMock(return_value=False)
        handler.shared_state = MagicMock()
        handler.params = MagicMock()
        handler.params.x402_payment_requirements = {
            "threshold": 1000000,
            "top_up": 5000000,
        }

        with patch(
            "packages.valory.skills.memeooorr_abci.handlers.Web3.to_checksum_address",
            side_effect=lambda x: x,
        ):
            result = HttpHandler._ensure_sufficient_funds_for_x402_payments(handler)
        assert result is False

    def test_insufficient_balance_successful_swap(self) -> None:
        """Test successful ETH to USDC swap."""
        handler = _make_http_handler()
        handler._get_eoa_account = MagicMock()
        handler._get_eoa_account.return_value.address = "0xeoa"
        handler._check_usdc_balance = MagicMock(return_value=100)
        handler._get_lifi_quote_sync = MagicMock(
            return_value={
                "transactionRequest": {"to": "0x1", "data": "0x", "value": 256}
            }
        )
        handler._get_nonce_and_gas_web3 = MagicMock(return_value=(1, 20000000000))
        handler._estimate_gas = MagicMock(return_value=200000)
        handler._sign_and_submit_tx_web3 = MagicMock(return_value="0xhash")
        handler._check_transaction_status = MagicMock(return_value=True)
        handler.shared_state = MagicMock()
        handler.params = MagicMock()
        handler.params.x402_payment_requirements = {
            "threshold": 1000000,
            "top_up": 5000000,
        }

        with patch(
            "packages.valory.skills.memeooorr_abci.handlers.Web3.to_checksum_address",
            side_effect=lambda x: x,
        ):
            result = HttpHandler._ensure_sufficient_funds_for_x402_payments(handler)
        assert result is True
        assert handler.shared_state.sufficient_funds_for_x402_payments is True

    def test_insufficient_balance_hex_value(self) -> None:
        """Test swap with hex transaction value."""
        handler = _make_http_handler()
        handler._get_eoa_account = MagicMock()
        handler._get_eoa_account.return_value.address = "0xeoa"
        handler._check_usdc_balance = MagicMock(return_value=100)
        handler._get_lifi_quote_sync = MagicMock(
            return_value={
                "transactionRequest": {"to": "0x1", "data": "0x", "value": "0x100"}
            }
        )
        handler._get_nonce_and_gas_web3 = MagicMock(return_value=(1, 20000000000))
        handler._estimate_gas = MagicMock(return_value=200000)
        handler._sign_and_submit_tx_web3 = MagicMock(return_value="0xhash")
        handler._check_transaction_status = MagicMock(return_value=True)
        handler.shared_state = MagicMock()
        handler.params = MagicMock()
        handler.params.x402_payment_requirements = {
            "threshold": 1000000,
            "top_up": 5000000,
        }

        with patch(
            "packages.valory.skills.memeooorr_abci.handlers.Web3.to_checksum_address",
            side_effect=lambda x: x,
        ):
            result = HttpHandler._ensure_sufficient_funds_for_x402_payments(handler)
        assert result is True

    def test_general_exception(self) -> None:
        """Test general exception handling."""
        handler = _make_http_handler()
        handler._get_eoa_account = MagicMock(side_effect=Exception("unexpected"))
        handler.shared_state = MagicMock()

        result = HttpHandler._ensure_sufficient_funds_for_x402_payments(handler)
        assert result is False
        assert handler.shared_state.sufficient_funds_for_x402_payments is False
