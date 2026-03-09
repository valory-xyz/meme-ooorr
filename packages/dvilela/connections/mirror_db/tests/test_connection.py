# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023-2024 Valory AG
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

"""Tests for the MirrorDB connection."""

# pylint: disable=protected-access,unused-argument,too-few-public-methods,no-member,import-outside-toplevel,missing-class-docstring

import asyncio
import json
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from aea.connections.base import ConnectionStates
from aea.protocols.base import Message

from packages.dvilela.connections.mirror_db.connection import (
    MirrorDBConnection,
    SrrDialogues,
    _handle_retryable_exception,
    retry_with_exponential_backoff,
)
from packages.valory.protocols.srr.message import SrrMessage

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_client_response_error(status: int) -> aiohttp.ClientResponseError:
    """Create a ClientResponseError with the given status."""
    mock_request_info = MagicMock()
    return aiohttp.ClientResponseError(
        request_info=mock_request_info,
        history=(),
        status=status,
        message="error",
    )


def _make_connection(base_url: str = "http://localhost:8000") -> MirrorDBConnection:
    """Create a MirrorDBConnection with mocked internals."""
    mock_config = MagicMock()
    mock_config.config = {"mirror_db_base_url": base_url}
    mock_config.public_id = MagicMock()
    mock_config.public_id.__str__ = MagicMock(return_value="dvilela/mirror_db:0.1.0")

    # _state mock: state property reads _state.get(), is_disconnected compares state
    mock_state = MagicMock()
    mock_state.get.return_value = MagicMock()

    with patch(
        "packages.dvilela.connections.mirror_db.connection.ssl.create_default_context"
    ), patch(
        "packages.dvilela.connections.mirror_db.connection.certifi.where",
        return_value="/fake/cert",
    ):
        conn = MirrorDBConnection.__new__(MirrorDBConnection)
        # Set internal attributes that back read-only properties
        object.__setattr__(conn, "_configuration", mock_config)
        object.__setattr__(conn, "_logger", MagicMock())
        object.__setattr__(conn, "_state", mock_state)
        conn.base_url = base_url
        conn.session = None
        conn.dialogues = MagicMock()
        conn._response_envelopes = None  # noqa: SLF001
        conn.task_to_request = {}
        conn.ssl_context = MagicMock()
    return conn


def _make_srr_message(
    performative: Any = None, payload: Optional[str] = None
) -> MagicMock:
    """Create a mocked SrrMessage."""
    msg = MagicMock(spec=SrrMessage)
    if performative is None:
        performative = SrrMessage.Performative.REQUEST
    msg.performative = performative
    if payload is not None:
        msg.payload = payload
    return msg


def _make_envelope(
    sender: str = "agent",
    to: str = "connection",
    message: Optional[Any] = None,
) -> MagicMock:
    """Create a mocked Envelope."""
    envelope = MagicMock()
    envelope.sender = sender
    envelope.to = to
    envelope.message = message or _make_srr_message()
    envelope.context = MagicMock()
    return envelope


# ---------------------------------------------------------------------------
# Tests for _handle_retryable_exception
# ---------------------------------------------------------------------------


class TestHandleRetryableException:
    """Tests for _handle_retryable_exception."""

    @pytest.mark.asyncio
    async def test_rate_limit_retryable(self) -> None:
        """Rate-limit (429) with remaining attempts returns True."""
        exc = _make_client_response_error(429)
        logger = MagicMock()
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await _handle_retryable_exception(exc, 0, 5, 1, logger)
        assert result is True
        mock_sleep.assert_awaited_once_with(1)
        logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_error_retryable(self) -> None:
        """Verify ClientConnectionError with remaining attempts returns True."""
        exc = aiohttp.ClientConnectionError("connection lost")
        logger = MagicMock()
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await _handle_retryable_exception(exc, 1, 5, 2, logger)
        assert result is True
        mock_sleep.assert_awaited_once_with(2)

    @pytest.mark.asyncio
    async def test_non_retryable_error(self) -> None:
        """Non-429 ClientResponseError returns False immediately."""
        exc = _make_client_response_error(500)
        logger = MagicMock()
        result = await _handle_retryable_exception(exc, 0, 5, 1, logger)
        assert result is False

    @pytest.mark.asyncio
    async def test_rate_limit_max_retries_reached(self) -> None:
        """Rate-limit on final attempt returns False and logs error."""
        exc = _make_client_response_error(429)
        logger = MagicMock()
        result = await _handle_retryable_exception(exc, 4, 5, 1, logger)
        assert result is False
        logger.error.assert_called_once()
        assert "rate limiting" in logger.error.call_args[0][0]

    @pytest.mark.asyncio
    async def test_connection_error_max_retries_reached(self) -> None:
        """Connection error on final attempt returns False and logs error."""
        exc = aiohttp.ClientConnectionError("conn error")
        logger = MagicMock()
        result = await _handle_retryable_exception(exc, 4, 5, 1, logger)
        assert result is False
        logger.error.assert_called_once()
        assert "connection error" in logger.error.call_args[0][0]


# ---------------------------------------------------------------------------
# Tests for retry_with_exponential_backoff decorator
# ---------------------------------------------------------------------------


class TestRetryWithExponentialBackoff:
    """Tests for the retry_with_exponential_backoff decorator."""

    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self) -> None:
        """Function succeeds immediately without retries."""

        class FakeConn:
            logger = MagicMock()

        @retry_with_exponential_backoff(max_retries=3, initial_delay=0.01)
        async def succeed(self: Any) -> str:
            return "ok"

        result = await succeed(FakeConn())
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_retry_then_success(self) -> None:
        """Retries on ClientConnectionError then succeeds."""
        call_count = 0

        class FakeConn:
            logger = MagicMock()

        @retry_with_exponential_backoff(
            max_retries=3, initial_delay=0.01, backoff_factor=2
        )
        async def flaky(self: Any) -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise aiohttp.ClientConnectionError("fail")
            return "recovered"

        with patch(
            "packages.dvilela.connections.mirror_db.connection.asyncio.sleep",
            new_callable=AsyncMock,
        ):
            result = await flaky(FakeConn())
        assert result == "recovered"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self) -> None:
        """All retries fail, final exception is raised."""

        class FakeConn:
            logger = MagicMock()

        @retry_with_exponential_backoff(
            max_retries=2, initial_delay=0.01, backoff_factor=2
        )
        async def always_fail(self: Any) -> None:
            raise aiohttp.ClientConnectionError("always fails")

        with patch(
            "packages.dvilela.connections.mirror_db.connection.asyncio.sleep",
            new_callable=AsyncMock,
        ):
            with pytest.raises(aiohttp.ClientConnectionError):
                await always_fail(FakeConn())

    @pytest.mark.asyncio
    async def test_unexpected_error_raises_immediately(self) -> None:
        """Non-aiohttp exception is re-raised immediately."""

        class FakeConn:
            logger = MagicMock()

        @retry_with_exponential_backoff(max_retries=3, initial_delay=0.01)
        async def bad(self: Any) -> None:
            raise RuntimeError("unexpected")

        with pytest.raises(RuntimeError, match="unexpected"):
            await bad(FakeConn())


# ---------------------------------------------------------------------------
# Tests for SrrDialogues
# ---------------------------------------------------------------------------


class TestSrrDialogues:
    """Tests for SrrDialogues helper class."""

    def test_initialization(self) -> None:
        """Verify SrrDialogues initializes with connection_id."""
        dialogues = SrrDialogues(
            connection_id=MagicMock(__str__=lambda self: "dvilela/mirror_db:0.1.0")
        )
        assert dialogues is not None


# ---------------------------------------------------------------------------
# Tests for MirrorDBConnection lifecycle
# ---------------------------------------------------------------------------


class TestMirrorDBConnectionLifecycle:
    """Tests for connect / disconnect / response_envelopes."""

    @pytest.mark.asyncio
    async def test_connect(self) -> None:
        """Connect sets up session and queue."""
        conn = _make_connection()
        with patch("aiohttp.ClientSession"), patch("aiohttp.TCPConnector"):
            await conn.connect()
        assert conn._response_envelopes is not None  # noqa: SLF001
        assert conn.session is not None

    @pytest.mark.asyncio
    async def test_disconnect(self) -> None:
        """Disconnect closes session and clears queue."""
        conn = _make_connection()
        conn._response_envelopes = asyncio.Queue()  # noqa: SLF001
        mock_session = AsyncMock()
        conn.session = mock_session
        # is_disconnected returns False by default (mock state != ConnectionStates.disconnected)

        await conn.disconnect()

        mock_session.close.assert_awaited_once()
        assert conn.session is None
        assert conn._response_envelopes is None  # noqa: SLF001

    @pytest.mark.asyncio
    async def test_disconnect_when_already_disconnected(self) -> None:
        """Disconnect returns immediately when already disconnected."""
        conn = _make_connection()
        # Make is_disconnected return True by setting state to disconnected
        conn._state.get.return_value = ConnectionStates.disconnected  # type: ignore[attr-defined]  # noqa: SLF001
        # Should not raise or do anything
        await conn.disconnect()

    @pytest.mark.asyncio
    async def test_disconnect_cancels_pending_tasks(self) -> None:
        """Disconnect cancels pending tasks."""
        conn = _make_connection()
        conn._response_envelopes = asyncio.Queue()  # noqa: SLF001
        conn.session = AsyncMock()
        # is_disconnected returns False by default (mock state != ConnectionStates.disconnected)

        mock_task = MagicMock()
        mock_task.cancelled.return_value = False
        conn.task_to_request[mock_task] = MagicMock()

        await conn.disconnect()
        mock_task.cancel.assert_called_once()

    def test_response_envelopes_when_none(self) -> None:
        """Property response_envelopes raises ValueError when not initialized."""
        conn = _make_connection()
        with pytest.raises(ValueError, match="not yet initialized"):
            _ = conn.response_envelopes

    def test_response_envelopes_returns_queue(self) -> None:
        """Property response_envelopes returns the queue when initialized."""
        conn = _make_connection()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        conn._response_envelopes = queue  # noqa: SLF001
        assert conn.response_envelopes is queue


# ---------------------------------------------------------------------------
# Tests for receive / send / _handle_envelope
# ---------------------------------------------------------------------------


class TestMirrorDBConnectionMessaging:
    """Tests for send, receive, _handle_envelope."""

    @pytest.mark.asyncio
    async def test_receive(self) -> None:
        """Receive returns item from the queue."""
        conn = _make_connection()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        conn._response_envelopes = queue  # noqa: SLF001
        sentinel = object()
        queue.put_nowait(sentinel)
        result = await conn.receive()
        assert result is sentinel

    @pytest.mark.asyncio
    async def test_send(self) -> None:
        """Send creates a task and stores it in task_to_request."""
        conn = _make_connection()
        conn._response_envelopes = asyncio.Queue()  # noqa: SLF001
        envelope = _make_envelope()
        conn.dialogues.update.return_value = MagicMock()  # type: ignore[attr-defined]

        mock_task = MagicMock()
        with patch.object(conn.loop, "create_task", return_value=mock_task):
            await conn.send(envelope)

        mock_task.add_done_callback.assert_called_once()
        assert mock_task in conn.task_to_request


# ---------------------------------------------------------------------------
# Tests for _get_response
# ---------------------------------------------------------------------------


class TestGetResponse:
    """Tests for _get_response."""

    @pytest.mark.asyncio
    async def test_non_request_performative(self) -> None:
        """Non-REQUEST performative returns error message."""
        conn = _make_connection()
        msg = _make_srr_message(performative=SrrMessage.Performative.RESPONSE)
        dialogue = MagicMock()
        conn.prepare_error_message = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]

        await conn._get_response(msg, dialogue)

        conn.prepare_error_message.assert_called_once()
        assert "not supported" in conn.prepare_error_message.call_args[0][2]

    @pytest.mark.asyncio
    async def test_invalid_json_payload(self) -> None:
        """Invalid JSON payload returns error message."""
        conn = _make_connection()
        msg = _make_srr_message(payload="not valid json{{{")
        dialogue = MagicMock()
        conn.prepare_error_message = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]

        await conn._get_response(msg, dialogue)

        conn.prepare_error_message.assert_called_once()
        assert "Invalid JSON" in conn.prepare_error_message.call_args[0][2]

    @pytest.mark.asyncio
    async def test_method_not_allowed(self) -> None:
        """Method not in _ALLOWED_METHODS returns error."""
        conn = _make_connection()
        payload = json.dumps({"method": "forbidden_method", "kwargs": {}})
        msg = _make_srr_message(payload=payload)
        dialogue = MagicMock()
        conn.prepare_error_message = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]

        await conn._get_response(msg, dialogue)

        conn.prepare_error_message.assert_called_once()
        assert "not allowed" in conn.prepare_error_message.call_args[0][2]

    @pytest.mark.asyncio
    async def test_method_not_found(self) -> None:
        """Method in allowed list but not on instance returns error."""
        conn = _make_connection()
        payload = json.dumps({"method": "create_", "kwargs": {"endpoint": "/test"}})
        msg = _make_srr_message(payload=payload)
        dialogue = MagicMock()
        conn.prepare_error_message = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]

        # Remove the method from the connection instance
        with patch.object(type(conn), "create_", new=None):
            # getattr returns None => not callable
            await conn._get_response(msg, dialogue)

        conn.prepare_error_message.assert_called_once()
        assert "not found or not callable" in conn.prepare_error_message.call_args[0][2]

    @pytest.mark.asyncio
    async def test_missing_endpoint(self) -> None:
        """Missing endpoint for CRUD methods returns error."""
        conn = _make_connection()
        payload = json.dumps({"method": "read_", "kwargs": {}})
        msg = _make_srr_message(payload=payload)
        dialogue = MagicMock()
        conn.prepare_error_message = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]

        await conn._get_response(msg, dialogue)

        conn.prepare_error_message.assert_called_once()
        assert "Missing endpoint" in conn.prepare_error_message.call_args[0][2]

    @pytest.mark.asyncio
    async def test_successful_request(self) -> None:
        """Successful method call returns proper response."""
        conn = _make_connection()
        payload = json.dumps({"method": "read_", "kwargs": {"endpoint": "/items"}})
        msg = _make_srr_message(payload=payload)
        dialogue = MagicMock()
        mock_response_msg = MagicMock()
        dialogue.reply.return_value = mock_response_msg

        conn.read_ = AsyncMock(return_value={"items": []})  # type: ignore[method-assign]

        result = await conn._get_response(msg, dialogue)

        conn.read_.assert_awaited_once_with(endpoint="/items")
        dialogue.reply.assert_called_once()
        assert result is mock_response_msg

    @pytest.mark.asyncio
    async def test_generic_exception_during_method_call(self) -> None:
        """Exception during method execution returns error message."""
        conn = _make_connection()
        payload = json.dumps({"method": "read_", "kwargs": {"endpoint": "/items"}})
        msg = _make_srr_message(payload=payload)
        dialogue = MagicMock()
        conn.prepare_error_message = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]

        conn.read_ = AsyncMock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]

        await conn._get_response(msg, dialogue)

        conn.prepare_error_message.assert_called_once()
        assert (
            "Exception processing request" in conn.prepare_error_message.call_args[0][2]
        )


# ---------------------------------------------------------------------------
# Tests for prepare_error_message
# ---------------------------------------------------------------------------


class TestPrepareErrorMessage:
    """Tests for prepare_error_message."""

    def test_dialogue_is_base_dialogue(self) -> None:
        """When dialogue is BaseDialogue instance, uses it directly."""
        from aea.protocols.dialogue.base import Dialogue as BaseDialogue

        conn = _make_connection()
        msg = _make_srr_message()
        dialogue = MagicMock(spec=BaseDialogue)
        reply_msg = MagicMock()
        dialogue.reply.return_value = reply_msg

        result = conn.prepare_error_message(msg, dialogue, "test error")

        dialogue.reply.assert_called_once()
        assert result is not None

    def test_dialogue_not_base_dialogue_found(self) -> None:
        """When dialogue is not BaseDialogue, looks it up via get_dialogue."""
        conn = _make_connection()
        msg = _make_srr_message()
        # Pass something that is NOT a BaseDialogue
        non_dialogue = "not_a_dialogue"
        found_dialogue = MagicMock()
        reply_msg = MagicMock()
        found_dialogue.reply.return_value = reply_msg
        conn.dialogues.get_dialogue.return_value = found_dialogue  # type: ignore[attr-defined]

        conn.prepare_error_message(msg, non_dialogue, "test error")  # type: ignore[arg-type]

        conn.dialogues.get_dialogue.assert_called_once_with(msg)  # type: ignore[attr-defined]
        found_dialogue.reply.assert_called_once()

    def test_dialogue_not_found_raises_value_error(self) -> None:
        """When dialogue lookup returns None, raises ValueError."""
        conn = _make_connection()
        msg = _make_srr_message()
        conn.dialogues.get_dialogue.return_value = None  # type: ignore[attr-defined]

        with pytest.raises(ValueError, match="Dialogue not found"):
            conn.prepare_error_message(msg, None, "test error")


# ---------------------------------------------------------------------------
# Tests for _handle_done_task
# ---------------------------------------------------------------------------


class TestHandleDoneTask:
    """Tests for _handle_done_task."""

    def test_successful_result(self) -> None:
        """Successful task result puts envelope in queue."""
        conn = _make_connection()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        conn._response_envelopes = queue  # noqa: SLF001

        request_envelope = _make_envelope()
        response_msg = MagicMock(spec=Message)
        # Envelope constructor checks message.to/sender match envelope to/sender
        response_msg.to = request_envelope.sender
        response_msg.sender = request_envelope.to
        task = MagicMock()
        task.result.return_value = response_msg
        conn.task_to_request[task] = request_envelope

        conn._handle_done_task(task)

        assert queue.qsize() == 1
        envelope = queue.get_nowait()
        assert envelope.to == request_envelope.sender
        assert envelope.sender == request_envelope.to

    def test_task_exception(self) -> None:
        """Task that raises exception logs error, returns None result."""
        conn = _make_connection()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        conn._response_envelopes = queue  # noqa: SLF001

        request_envelope = _make_envelope()
        task = MagicMock()
        task.result.side_effect = RuntimeError("task failed")
        conn.task_to_request[task] = request_envelope

        conn._handle_done_task(task)

        conn.logger.error.assert_called()  # type: ignore[attr-defined]
        # response_message is None after exception, so warning is logged
        conn.logger.warning.assert_called_once()  # type: ignore[attr-defined]
        assert queue.qsize() == 0

    def test_none_result(self) -> None:
        """Task returning None logs warning and does not enqueue."""
        conn = _make_connection()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        conn._response_envelopes = queue  # noqa: SLF001

        request_envelope = _make_envelope()
        task = MagicMock()
        task.result.return_value = None
        conn.task_to_request[task] = request_envelope

        conn._handle_done_task(task)

        conn.logger.warning.assert_called_once()  # type: ignore[attr-defined]
        assert queue.qsize() == 0


# ---------------------------------------------------------------------------
# Tests for _raise_for_response
# ---------------------------------------------------------------------------


class TestRaiseForResponse:
    """Tests for _raise_for_response."""

    @pytest.mark.asyncio
    async def test_status_200(self) -> None:
        """200 status returns without raising."""
        conn = _make_connection()
        response = MagicMock()
        response.status = 200
        # Should not raise
        await conn._raise_for_response(response, "test")

    @pytest.mark.asyncio
    async def test_non_200_status(self) -> None:
        """Non-200 status raises Exception with detail."""
        conn = _make_connection()
        response = AsyncMock()
        response.status = 404
        response.json.return_value = {"detail": "Not found"}
        with pytest.raises(Exception, match="Not found"):
            await conn._raise_for_response(response, "read")

    @pytest.mark.asyncio
    async def test_non_200_status_no_detail(self) -> None:
        """Non-200 status without detail key uses full response."""
        conn = _make_connection()
        response = AsyncMock()
        response.status = 500
        error_body: Dict[str, str] = {"error": "internal"}
        response.json.return_value = error_body
        with pytest.raises(Exception, match="internal"):
            await conn._raise_for_response(response, "create")


# ---------------------------------------------------------------------------
# Tests for CRUD methods with session=None
# ---------------------------------------------------------------------------


class TestCRUDSessionNone:
    """Tests for CRUD methods raising ValueError when session is None."""

    @pytest.mark.asyncio
    async def test_create_no_session(self) -> None:
        """Method create_ raises ValueError when session is None."""
        conn = _make_connection()
        conn.session = None
        with pytest.raises(ValueError, match="Session not initialized"):
            await conn.create_.__wrapped__(conn, endpoint="/test", data={"key": "val"})

    @pytest.mark.asyncio
    async def test_read_no_session(self) -> None:
        """Method read_ raises ValueError when session is None."""
        conn = _make_connection()
        conn.session = None
        with pytest.raises(ValueError, match="Session not initialized"):
            await conn.read_.__wrapped__(conn, endpoint="/test")

    @pytest.mark.asyncio
    async def test_update_no_session(self) -> None:
        """Method update_ raises ValueError when session is None."""
        conn = _make_connection()
        conn.session = None
        with pytest.raises(ValueError, match="Session not initialized"):
            await conn.update_.__wrapped__(conn, endpoint="/test", data={"key": "val"})

    @pytest.mark.asyncio
    async def test_delete_no_session(self) -> None:
        """Method delete_ raises ValueError when session is None."""
        conn = _make_connection()
        conn.session = None
        with pytest.raises(ValueError, match="Session not initialized"):
            await conn.delete_.__wrapped__(conn, endpoint="/test")


# ---------------------------------------------------------------------------
# Tests for CRUD methods with a mocked session (happy path)
# ---------------------------------------------------------------------------


class TestCRUDWithSession:
    """Tests for CRUD methods making actual HTTP calls via mocked session."""

    def _mock_response(
        self, status: int = 200, json_data: Optional[Dict[str, Any]] = None
    ) -> MagicMock:
        """Create a mock aiohttp response usable as async context manager."""
        resp = MagicMock()
        resp.status = status
        resp.json = AsyncMock(return_value=json_data or {})
        # Support `async with session.get(...) as response:`
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=resp)
        ctx.__aexit__ = AsyncMock(return_value=False)
        return ctx

    @pytest.mark.asyncio
    async def test_create_success(self) -> None:
        """Method create_ POSTs and returns JSON."""
        conn = _make_connection()
        mock_ctx = self._mock_response(200, {"id": 1})
        conn.session = MagicMock()
        conn.session.post.return_value = mock_ctx

        result = await conn.create_.__wrapped__(
            conn, endpoint="/items", data={"name": "test"}
        )
        assert result == {"id": 1}
        conn.session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_read_success(self) -> None:
        """Method read_ GETs and returns JSON."""
        conn = _make_connection()
        mock_ctx = self._mock_response(200, {"items": []})
        conn.session = MagicMock()
        conn.session.get.return_value = mock_ctx

        result = await conn.read_.__wrapped__(conn, endpoint="/items")
        assert result == {"items": []}

    @pytest.mark.asyncio
    async def test_update_success(self) -> None:
        """Method update_ PUTs and returns JSON."""
        conn = _make_connection()
        mock_ctx = self._mock_response(200, {"updated": True})
        conn.session = MagicMock()
        conn.session.put.return_value = mock_ctx

        result = await conn.update_.__wrapped__(
            conn, endpoint="/items/1", data={"name": "updated"}
        )
        assert result == {"updated": True}

    @pytest.mark.asyncio
    async def test_delete_success(self) -> None:
        """Method delete_ DELETEs and returns JSON."""
        conn = _make_connection()
        mock_ctx = self._mock_response(200, {"deleted": True})
        conn.session = MagicMock()
        conn.session.delete.return_value = mock_ctx

        result = await conn.delete_.__wrapped__(conn, endpoint="/items/1")
        assert result == {"deleted": True}
