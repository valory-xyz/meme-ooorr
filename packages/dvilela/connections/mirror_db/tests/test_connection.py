#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2021-2024 David Vilela Freire
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

"""Tests for MirrorDB connection."""

# pylint: disable=protected-access,unused-argument


import asyncio
import json
from typing import Any, Tuple, cast
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from aea.configurations.base import ConnectionConfig, PublicId
from aea.identity.base import Identity
from aea.mail.base import Envelope
from aea.protocols.dialogue.base import Dialogue as BaseDialogue

from packages.dvilela.connections.mirror_db.connection import (
    MirrorDBConnection,
    SrrDialogues,
    _handle_retryable_exception,
    retry_with_exponential_backoff,
)
from packages.valory.protocols.srr.message import SrrMessage

CONNECTION_PUBLIC_ID = PublicId.from_str("dvilela/mirror_db:0.1.0")
BASE_URL = "http://test-mirror-db.example.com"


def make_connection() -> MirrorDBConnection:
    """Create a MirrorDBConnection instance for testing."""
    identity = Identity(
        "test_agent", address="test_agent_address", public_key="test_public_key"
    )
    configuration = ConnectionConfig(
        mirror_db_base_url=BASE_URL,
        connection_id=CONNECTION_PUBLIC_ID,
    )
    return MirrorDBConnection(
        identity=identity,
        configuration=configuration,
        data_dir=".",
    )


def create_request(
    dialogues: SrrDialogues, payload: str
) -> Tuple[SrrMessage, BaseDialogue]:
    """Create a request message with a properly initialized dialogue."""
    msg, dialogue = dialogues.create(
        counterparty="test_sender",
        performative=SrrMessage.Performative.REQUEST,
        payload=payload,
    )
    return cast(SrrMessage, msg), cast(BaseDialogue, dialogue)


class TestSrrDialogues:  # pylint: disable=too-few-public-methods
    """Tests for SrrDialogues."""

    def test_role_from_first_message(self) -> None:
        """Test that SrrDialogues assigns the CONNECTION role."""
        dialogues = SrrDialogues(connection_id=CONNECTION_PUBLIC_ID)
        _msg, dialogue = dialogues.create(
            counterparty="test_sender",
            performative=SrrMessage.Performative.REQUEST,
            payload=json.dumps({"method": "read_", "kwargs": {"endpoint": "/test"}}),
        )
        assert dialogue is not None


class TestMirrorDBConnectionInit:  # pylint: disable=too-few-public-methods
    """Tests for MirrorDBConnection initialization."""

    def test_init(self) -> None:
        """Test MirrorDBConnection initializes correctly."""
        connection = make_connection()
        assert connection.base_url == BASE_URL
        assert connection.session is None
        assert connection._response_envelopes is None
        assert not connection.task_to_request
        assert connection.ssl_context is not None


class TestMirrorDBConnectionLifecycle:
    """Tests for connect/disconnect lifecycle."""

    @pytest.mark.asyncio(loop_scope="function")
    async def test_connect(self) -> None:
        """Test connect sets up session and queue."""
        connection = make_connection()
        await connection.connect()
        assert connection.session is not None
        assert connection._response_envelopes is not None
        await connection.disconnect()

    @pytest.mark.asyncio(loop_scope="function")
    async def test_disconnect_empty_tasks_and_no_session(self) -> None:
        """Test disconnect with no tasks and session already None.

        Covers branches:
        - empty task_to_request (for loop body not entered)
        - session is None (skip close)
        """
        connection = make_connection()
        await connection.connect()
        # Close and clear session to test branch
        assert connection.session is not None
        await connection.session.close()
        connection.session = None
        assert len(connection.task_to_request) == 0
        await connection.disconnect()

    @pytest.mark.asyncio(loop_scope="function")
    async def test_disconnect_when_already_disconnected(self) -> None:
        """Test disconnect is no-op when already disconnected."""
        connection = make_connection()
        await connection.disconnect()

    @pytest.mark.asyncio(loop_scope="function")
    async def test_disconnect_cancels_tasks(self) -> None:
        """Test disconnect cancels pending tasks."""
        connection = make_connection()
        await connection.connect()
        mock_future = MagicMock(spec=asyncio.Future)
        mock_future.cancelled.return_value = False
        connection.task_to_request[mock_future] = MagicMock()
        await connection.disconnect()
        mock_future.cancel.assert_called_once()

    @pytest.mark.asyncio(loop_scope="function")
    async def test_disconnect_skips_already_cancelled_tasks(self) -> None:
        """Test disconnect does not cancel already-cancelled tasks.

        Covers branch 194->193: task.cancelled() returns True.
        """
        connection = make_connection()
        await connection.connect()

        mock_future = MagicMock(spec=asyncio.Future)
        mock_future.cancelled.return_value = True
        connection.task_to_request[mock_future] = MagicMock()

        await connection.disconnect()
        mock_future.cancel.assert_not_called()

    @pytest.mark.asyncio(loop_scope="function")
    async def test_response_envelopes_not_initialized(self) -> None:
        """Test response_envelopes raises when not initialized."""
        connection = make_connection()
        with pytest.raises(ValueError, match="not yet initialized"):
            _ = connection.response_envelopes


class TestHandleRetryableException:
    """Tests for _handle_retryable_exception."""

    @pytest.mark.asyncio(loop_scope="function")
    async def test_non_retryable_client_response_error(self) -> None:
        """Test non-429 ClientResponseError returns False."""
        exc = aiohttp.ClientResponseError(
            request_info=MagicMock(), history=(), status=500
        )
        logger = MagicMock()
        result = await _handle_retryable_exception(exc, 0, 3, 1, logger)
        assert result is False

    @pytest.mark.asyncio(loop_scope="function")
    async def test_rate_limit_retry(self) -> None:
        """Test 429 error triggers retry."""
        exc = aiohttp.ClientResponseError(
            request_info=MagicMock(), history=(), status=429
        )
        logger = MagicMock()
        with patch(
            "packages.dvilela.connections.mirror_db.connection.asyncio.sleep",
            new_callable=AsyncMock,
        ):
            result = await _handle_retryable_exception(exc, 0, 3, 0.01, logger)
        assert result is True

    @pytest.mark.asyncio(loop_scope="function")
    async def test_rate_limit_max_retries(self) -> None:
        """Test 429 on last attempt returns False."""
        exc = aiohttp.ClientResponseError(
            request_info=MagicMock(), history=(), status=429
        )
        logger = MagicMock()
        result = await _handle_retryable_exception(exc, 2, 3, 1, logger)
        assert result is False
        logger.error.assert_called_once()  # pylint: disable=no-member

    @pytest.mark.asyncio(loop_scope="function")
    async def test_connection_error_retry(self) -> None:
        """Test connection error triggers retry."""
        exc = aiohttp.ClientConnectionError("connection failed")
        logger = MagicMock()
        with patch(
            "packages.dvilela.connections.mirror_db.connection.asyncio.sleep",
            new_callable=AsyncMock,
        ):
            result = await _handle_retryable_exception(exc, 0, 3, 0.01, logger)
        assert result is True

    @pytest.mark.asyncio(loop_scope="function")
    async def test_connection_error_max_retries(self) -> None:
        """Test connection error on last attempt returns False."""
        exc = aiohttp.ClientConnectionError("connection failed")
        logger = MagicMock()
        result = await _handle_retryable_exception(exc, 2, 3, 1, logger)
        assert result is False
        logger.error.assert_called_once()  # pylint: disable=no-member


class TestRetryDecorator:  # pylint: disable=too-few-public-methods
    """Tests for retry_with_exponential_backoff decorator."""

    @pytest.mark.asyncio(loop_scope="function")
    async def test_unexpected_error_raises_immediately(self) -> None:
        """Test unexpected exception re-raises immediately."""
        connection = make_connection()

        @retry_with_exponential_backoff(max_retries=3, initial_delay=0.01)
        async def failing_func(self: Any) -> None:
            raise RuntimeError("unexpected")

        with pytest.raises(RuntimeError, match="unexpected"):
            await failing_func(connection)


class TestGetResponse:
    """Tests for _get_response method."""

    @pytest.mark.asyncio(loop_scope="function")
    async def test_invalid_performative(self) -> None:
        """Test error when performative is not REQUEST."""
        connection = make_connection()
        await connection.connect()

        # Create a RESPONSE message directly (wrong performative)
        msg = SrrMessage(
            performative=SrrMessage.Performative.RESPONSE,
            payload="{}",
            error=False,
        )
        msg.sender = "test_sender"
        msg.to = str(CONNECTION_PUBLIC_ID)

        # Use a mock dialogue that allows reply
        mock_dialogue = MagicMock(spec=BaseDialogue)
        error_response = MagicMock(spec=SrrMessage)
        error_response.error = True
        mock_dialogue.reply.return_value = error_response

        result = await connection._get_response(msg, mock_dialogue)
        assert result.error is True
        await connection.disconnect()

    @pytest.mark.asyncio(loop_scope="function")
    async def test_disallowed_method(self) -> None:
        """Test error when method is not allowed."""
        connection = make_connection()
        await connection.connect()

        msg, dialogue = create_request(
            connection.dialogues,
            json.dumps({"method": "evil_method", "kwargs": {}}),
        )

        result = await connection._get_response(msg, dialogue)
        payload = json.loads(result.payload)
        assert "not allowed" in payload["error"]
        await connection.disconnect()

    @pytest.mark.asyncio(loop_scope="function")
    async def test_missing_endpoint(self) -> None:
        """Test error when endpoint is missing."""
        connection = make_connection()
        await connection.connect()

        msg, dialogue = create_request(
            connection.dialogues,
            json.dumps({"method": "read_", "kwargs": {}}),
        )

        result = await connection._get_response(msg, dialogue)
        payload = json.loads(result.payload)
        assert "Missing endpoint" in payload["error"]
        await connection.disconnect()

    @pytest.mark.asyncio(loop_scope="function")
    async def test_invalid_json_payload(self) -> None:
        """Test error for invalid JSON payload."""
        connection = make_connection()
        await connection.connect()

        msg, dialogue = create_request(
            connection.dialogues,
            "not-valid-json{{{",
        )

        result = await connection._get_response(msg, dialogue)
        payload = json.loads(result.payload)
        assert "Invalid JSON" in payload["error"]
        await connection.disconnect()

    @pytest.mark.asyncio(loop_scope="function")
    async def test_method_not_found(self) -> None:
        """Test error when method exists in allowed set but not on object."""
        connection = make_connection()
        await connection.connect()

        msg, dialogue = create_request(
            connection.dialogues,
            json.dumps({"method": "read_", "kwargs": {"endpoint": "/test"}}),
        )

        # Patch getattr to return None for read_
        with patch.object(
            type(connection), "read_", new_callable=lambda: property(lambda self: None)
        ):
            result = await connection._get_response(msg, dialogue)

        payload = json.loads(result.payload)
        assert "not found or not callable" in payload["error"]
        await connection.disconnect()

    @pytest.mark.asyncio(loop_scope="function")
    async def test_successful_read(self) -> None:
        """Test successful method dispatch."""
        connection = make_connection()
        await connection.connect()

        msg, dialogue = create_request(
            connection.dialogues,
            json.dumps({"method": "read_", "kwargs": {"endpoint": "/items/1"}}),
        )

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"id": 1, "name": "test"})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        with patch.object(connection.session, "get", return_value=mock_response):
            result = await connection._get_response(msg, dialogue)

        payload = json.loads(result.payload)
        assert payload["response"]["id"] == 1
        assert result.error is False
        await connection.disconnect()

    @pytest.mark.asyncio(loop_scope="function")
    async def test_exception_during_method_call(self) -> None:
        """Test general exception during method execution."""
        connection = make_connection()
        await connection.connect()

        msg, dialogue = create_request(
            connection.dialogues,
            json.dumps({"method": "read_", "kwargs": {"endpoint": "/fail"}}),
        )

        with patch.object(connection, "read_", side_effect=RuntimeError("boom")):
            result = await connection._get_response(msg, dialogue)

        payload = json.loads(result.payload)
        assert "Exception processing request" in payload["error"]
        await connection.disconnect()


class TestPrepareErrorMessage:
    """Tests for prepare_error_message."""

    @pytest.mark.asyncio(loop_scope="function")
    async def test_prepare_error_with_non_dialogue(self) -> None:
        """Test error preparation when dialogue param is not BaseDialogue."""
        connection = make_connection()
        await connection.connect()

        msg, _dialogue = create_request(connection.dialogues, "{}")
        # Pass None as dialogue to trigger the fallback lookup path
        result = connection.prepare_error_message(msg, None, "test error")
        assert result.error is True
        await connection.disconnect()

    @pytest.mark.asyncio(loop_scope="function")
    async def test_prepare_error_dialogue_not_found(self) -> None:
        """Test ValueError when dialogue cannot be found."""
        connection = make_connection()
        await connection.connect()

        # Create a message not registered with the connection's dialogues
        other_dialogues = SrrDialogues(connection_id=CONNECTION_PUBLIC_ID)
        msg, _ = create_request(other_dialogues, "{}")

        with pytest.raises(ValueError, match="Dialogue not found"):
            connection.prepare_error_message(msg, None, "test error")
        await connection.disconnect()


class TestHandleDoneTask:
    """Tests for _handle_done_task."""

    @pytest.mark.asyncio(loop_scope="function")
    async def test_task_returns_none(self) -> None:
        """Test handling when task result is None."""
        connection = make_connection()
        await connection.connect()

        mock_task = MagicMock(spec=asyncio.Future)
        mock_task.result.return_value = None
        mock_envelope = MagicMock(spec=Envelope)
        connection.task_to_request[mock_task] = mock_envelope

        connection._handle_done_task(mock_task)
        assert connection.response_envelopes.empty()
        await connection.disconnect()

    @pytest.mark.asyncio(loop_scope="function")
    async def test_task_raises_exception(self) -> None:
        """Test handling when task raises an exception."""
        connection = make_connection()
        await connection.connect()

        mock_task = MagicMock(spec=asyncio.Future)
        mock_task.result.side_effect = RuntimeError("task failed")
        mock_envelope = MagicMock(spec=Envelope)
        connection.task_to_request[mock_task] = mock_envelope

        connection._handle_done_task(mock_task)
        assert connection.response_envelopes.empty()
        await connection.disconnect()

    @pytest.mark.asyncio(loop_scope="function")
    async def test_task_success(self) -> None:
        """Test handling when task completes successfully."""
        connection = make_connection()
        await connection.connect()

        # Create a real SrrMessage so Envelope consistency checks pass
        mock_message = SrrMessage(
            performative=SrrMessage.Performative.RESPONSE,
            payload="{}",
            error=False,
        )
        mock_message.sender = "to_addr"
        mock_message.to = "sender_addr"

        mock_task = MagicMock(spec=asyncio.Future)
        mock_task.result.return_value = mock_message
        mock_envelope = MagicMock(spec=Envelope)
        mock_envelope.sender = "sender_addr"
        mock_envelope.to = "to_addr"
        mock_envelope.context = None
        connection.task_to_request[mock_task] = mock_envelope

        connection._handle_done_task(mock_task)
        assert not connection.response_envelopes.empty()
        await connection.disconnect()


class TestRaiseForResponse:
    """Tests for _raise_for_response."""

    @pytest.mark.asyncio(loop_scope="function")
    async def test_status_200(self) -> None:
        """Test no exception for 200 status."""
        connection = make_connection()
        mock_response = MagicMock()
        mock_response.status = 200
        await connection._raise_for_response(mock_response, "test")

    @pytest.mark.asyncio(loop_scope="function")
    async def test_error_status(self) -> None:
        """Test exception raised for non-200 status."""
        connection = make_connection()
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.json = AsyncMock(return_value={"detail": "Not found"})
        with pytest.raises(Exception, match="Not found"):
            await connection._raise_for_response(mock_response, "test")


class TestCRUDMethods:
    """Tests for create_, read_, update_, delete_ methods."""

    @pytest.mark.asyncio(loop_scope="function")
    async def test_create_no_session(self) -> None:
        """Test create_ raises when session is None."""
        connection = make_connection()
        with pytest.raises(ValueError, match="Session not initialized"):
            await connection.create_(endpoint="/test", data={"key": "val"})

    @pytest.mark.asyncio(loop_scope="function")
    async def test_read_no_session(self) -> None:
        """Test read_ raises when session is None."""
        connection = make_connection()
        with pytest.raises(ValueError, match="Session not initialized"):
            await connection.read_(endpoint="/test")

    @pytest.mark.asyncio(loop_scope="function")
    async def test_update_no_session(self) -> None:
        """Test update_ raises when session is None."""
        connection = make_connection()
        with pytest.raises(ValueError, match="Session not initialized"):
            await connection.update_(endpoint="/test", data={"key": "val"})

    @pytest.mark.asyncio(loop_scope="function")
    async def test_delete_no_session(self) -> None:
        """Test delete_ raises when session is None."""
        connection = make_connection()
        with pytest.raises(ValueError, match="Session not initialized"):
            await connection.delete_(endpoint="/test")

    @pytest.mark.asyncio(loop_scope="function")
    async def test_create_success(self) -> None:
        """Test successful create_ call."""
        connection = make_connection()
        await connection.connect()

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"id": 1})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        with patch.object(connection.session, "post", return_value=mock_response):
            result = await connection.create_(endpoint="/items", data={"name": "test"})

        assert result == {"id": 1}
        await connection.disconnect()

    @pytest.mark.asyncio(loop_scope="function")
    async def test_update_success(self) -> None:
        """Test successful update_ call."""
        connection = make_connection()
        await connection.connect()

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"id": 1, "name": "updated"})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        with patch.object(connection.session, "put", return_value=mock_response):
            result = await connection.update_(
                endpoint="/items/1", data={"name": "updated"}
            )

        assert result["name"] == "updated"
        await connection.disconnect()

    @pytest.mark.asyncio(loop_scope="function")
    async def test_delete_success(self) -> None:
        """Test successful delete_ call."""
        connection = make_connection()
        await connection.connect()

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"deleted": True})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        with patch.object(connection.session, "delete", return_value=mock_response):
            result = await connection.delete_(endpoint="/items/1")

        assert result["deleted"] is True
        await connection.disconnect()

    @pytest.mark.asyncio(loop_scope="function")
    async def test_read_retries_on_429_then_succeeds(self) -> None:
        """Test retry on 429 rate limit error then success.

        Covers lines 96-105: the ClientResponseError handler in the retry decorator.
        """
        connection = make_connection()
        await connection.connect()

        rate_limit_exc = aiohttp.ClientResponseError(
            request_info=MagicMock(), history=(), status=429
        )

        mock_success_response = AsyncMock()
        mock_success_response.status = 200
        mock_success_response.json = AsyncMock(return_value={"id": 1})
        mock_success_response.__aenter__ = AsyncMock(return_value=mock_success_response)
        mock_success_response.__aexit__ = AsyncMock(return_value=False)

        call_count = 0

        def mock_get(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise rate_limit_exc
            return mock_success_response

        with patch.object(connection.session, "get", side_effect=mock_get):
            with patch(
                "packages.dvilela.connections.mirror_db.connection.asyncio.sleep",
                new_callable=AsyncMock,
            ):
                result = await connection.read_(endpoint="/items/1")

        assert result == {"id": 1}
        assert call_count == 2
        await connection.disconnect()

    @pytest.mark.asyncio(loop_scope="function")
    async def test_read_retries_exhausted_on_429(self) -> None:
        """Test max retries exhausted on persistent 429 errors.

        Covers line 103-104: raise when should_continue is False.
        """
        connection = make_connection()
        await connection.connect()

        rate_limit_exc = aiohttp.ClientResponseError(
            request_info=MagicMock(), history=(), status=429
        )

        def mock_get(*args: Any, **kwargs: Any) -> Any:
            raise rate_limit_exc

        with patch.object(connection.session, "get", side_effect=mock_get):
            with patch(
                "packages.dvilela.connections.mirror_db.connection.asyncio.sleep",
                new_callable=AsyncMock,
            ):
                with pytest.raises(aiohttp.ClientResponseError):
                    await connection.read_(endpoint="/items/1")

        await connection.disconnect()


class TestReceiveAndSend:
    """Tests for receive and send methods."""

    @pytest.mark.asyncio(loop_scope="function")
    async def test_receive(self) -> None:
        """Test receive returns envelope from queue."""
        connection = make_connection()
        await connection.connect()

        mock_envelope = MagicMock(spec=Envelope)
        connection.response_envelopes.put_nowait(mock_envelope)
        result = await connection.receive()
        assert result is mock_envelope
        await connection.disconnect()

    @pytest.mark.asyncio(loop_scope="function")
    async def test_send_and_handle_envelope(self) -> None:
        """Test send dispatches envelope handling and registers callback.

        Covers lines 212-214 (send) and 218-221 (_handle_envelope).
        """
        connection = make_connection()
        await connection.connect()

        # Simulate an incoming message from an external skill/agent
        skill_dialogues = SrrDialogues(
            connection_id=PublicId.from_str("valory/test_skill:0.1.0")
        )
        incoming_msg, _ = skill_dialogues.create(
            counterparty=str(CONNECTION_PUBLIC_ID),
            performative=SrrMessage.Performative.REQUEST,
            payload=json.dumps({"method": "read_", "kwargs": {"endpoint": "/test"}}),
        )

        envelope = Envelope(
            to=str(CONNECTION_PUBLIC_ID),
            sender="valory/test_skill:0.1.0",
            message=incoming_msg,
        )

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"id": 1})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        with patch.object(connection.session, "get", return_value=mock_response):
            await connection.send(envelope)
            # Give the background task time to complete
            await asyncio.sleep(0.2)

        await connection.disconnect()
