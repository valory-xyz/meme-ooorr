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

"""Tests for agent_db_client module."""

# pylint: disable=no-member

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from packages.valory.skills.agent_db_abci.agent_db_client import AgentDBClient
from packages.valory.skills.agent_db_abci.agent_db_models import (
    AgentInstance,
    AgentType,
    AttributeDefinition,
    AttributeInstance,
)


def _make_client(base_url: str = "https://example.com/api/") -> AgentDBClient:
    """Create an AgentDBClient with Model.__init__ mocked out.

    The patch only affects super().__init__(**kwargs) on line 46.
    Lines 47-55 (attribute assignments) execute normally and are covered.

    :param base_url: the base URL for the client.
    :return: an AgentDBClient instance.
    """
    with patch("packages.valory.skills.agent_db_abci.agent_db_client.Model.__init__"):
        client = AgentDBClient(base_url=base_url)
    return client


def _exhaust_gen(gen: Any) -> Any:  # type: ignore[no-untyped-def]
    """Drive a generator to completion, returning its return value.

    Sends None at each yield point. Suitable for generators whose
    yield-from targets are also driven by send(None).

    :param gen: the generator to exhaust.
    :return: the generator's return value.
    """
    result = None
    try:
        gen.send(None)
        while True:
            gen.send(None)
    except StopIteration as e:
        result = e.value
    return result


class TestAgentDBClientInit:
    """Test AgentDBClient.__init__ (lines 44-55)."""

    def test_init_strips_trailing_slash(self) -> None:
        """Test that base_url trailing slash is stripped."""
        client = _make_client("https://example.com/api/")
        assert client.base_url == "https://example.com/api"

    def test_init_sets_defaults(self) -> None:
        """Test default attribute values set in __init__."""
        client = _make_client()
        assert not client._attribute_definition_cache
        assert client.agent is None
        assert client.agent_type is None
        assert client.address is None
        assert client.signing_func is None
        assert client.http_request_func is None
        assert client.logger is None
        assert client.agent_type_name is None
        assert client.agent_name_template is None


class TestAgentDBClientInitialize:  # pylint: disable=too-few-public-methods
    """Test AgentDBClient.initialize."""

    def test_initialize_sets_all_fields(self) -> None:
        """Test that initialize stores all injected dependencies."""
        client = _make_client()
        mock_http = MagicMock()
        mock_sign = MagicMock()
        mock_logger = MagicMock()

        client.initialize(
            address="0xABC",
            http_request_func=mock_http,
            signing_func=mock_sign,
            logger=mock_logger,
            agent_type_name="test_type",
            agent_name_template="agent-{address}",
        )

        assert client.address == "0xABC"
        assert client.http_request_func is mock_http
        assert client.signing_func is mock_sign
        assert client.logger is mock_logger
        assert client.agent_type_name == "test_type"
        assert client.agent_name_template == "agent-{address}"


class TestAgentDBClientCastAttributeValue:  # pylint: disable=too-few-public-methods
    """Test AgentDBClient.cast_attribute_value for all data_type branches."""

    @staticmethod
    def _attr_def(data_type: Any) -> Any:
        return AttributeDefinition(
            attr_def_id=1,
            type_id=1,
            attr_name="test",
            data_type=data_type,
            is_required=False,
            default_value="",
        )

    @pytest.mark.parametrize(
        "data_type, input_val, expected",
        [
            ("string", 123, "123"),
            ("integer", "42", 42),
            ("float", "3.14", 3.14),
            ("boolean", 1, True),
            ("json", {"key": "value"}, {"key": "value"}),
            ("date", "2025-01-01T00:00:00+00:00", None),  # checked separately
            ("unknown_type", "raw_value", "raw_value"),  # falls through all elifs
        ],
    )
    def test_cast_types(self, data_type: Any, input_val: Any, expected: Any) -> None:
        """Test cast for each data type."""
        from datetime import (  # pylint: disable=import-outside-toplevel
            datetime,
            timezone,
        )

        client = _make_client()
        result = client.cast_attribute_value(input_val, self._attr_def(data_type))
        if data_type == "date":
            assert result == datetime(2025, 1, 1, tzinfo=timezone.utc)
        else:
            assert result == expected


class TestAgentDBClientEnsureAgentInstance:
    """Test AgentDBClient._ensure_agent_instance."""

    def test_already_exists(self) -> None:
        """When agent is not None, return immediately."""
        client = _make_client()
        client.agent = MagicMock()
        gen = client._ensure_agent_instance()
        with pytest.raises(StopIteration):
            next(gen)

    def test_found_by_address(self) -> None:
        """When agent is found by address, also fetch its type."""
        client = _make_client()
        client.address = "0xABC"
        client.logger = MagicMock()

        agent_data = {
            "agent_id": 1,
            "type_id": 10,
            "agent_name": "test",
            "eth_address": "0xABC",
            "created_at": "2025-01-01T00:00:00+00:00",
        }
        agent_type_data = {"type_id": 10, "type_name": "m", "description": "d"}

        mock_resp_agent = MagicMock(
            status_code=200, body=json.dumps(agent_data).encode()
        )
        mock_resp_type = MagicMock(
            status_code=200, body=json.dumps(agent_type_data).encode()
        )

        def fake_http(**kwargs: Any) -> Any:
            """Yield-from compatible HTTP mock."""
            yield
            return mock_resp_agent

        client.http_request_func = fake_http

        gen = client._ensure_agent_instance()
        # Drive: first yield from _request -> http_request_func yields once
        gen.send(None)
        # Send None to advance past the yield in fake_http
        # But we need different responses for two requests.
        # Easier: just send None repeatedly, but the response is baked in fake_http.
        # Problem: second call also needs a response.

        # Let's use a different approach: make http_request_func return different
        # responses based on call count.
        call_count = 0
        responses = [mock_resp_agent, mock_resp_type]

        def multi_http(**kwargs: Any) -> Any:
            nonlocal call_count
            resp = responses[call_count]
            call_count += 1
            yield
            return resp

        client.http_request_func = multi_http
        # Re-create gen since we changed the func
        client.agent = None
        gen = client._ensure_agent_instance()
        _exhaust_gen(gen)

        assert client.agent is not None
        assert client.agent.agent_id == 1
        assert client.agent_type is not None

    def test_not_found_registers(self) -> None:
        """When agent not found by address, register if template is set."""
        client = _make_client()
        client.address = "0xABC"
        client.logger = MagicMock()
        client.agent_type_name = "memeooorr"
        client.agent_name_template = "agent-{address}"

        mock_404 = MagicMock(status_code=404)
        agent_type_data = {"type_id": 10, "type_name": "memeooorr", "description": "d"}
        mock_type = MagicMock(
            status_code=200, body=json.dumps(agent_type_data).encode()
        )
        agent_data = {
            "agent_id": 2,
            "type_id": 10,
            "agent_name": "agent-0xABC",
            "eth_address": "0xABC",
            "created_at": "2025-01-01T00:00:00+00:00",
        }
        mock_agent = MagicMock(status_code=201, body=json.dumps(agent_data).encode())

        call_count = 0
        responses = [mock_404, mock_type, mock_agent]

        def multi_http(**kwargs: Any) -> Any:
            nonlocal call_count
            resp = responses[call_count]
            call_count += 1
            yield
            return resp

        client.http_request_func = multi_http

        gen = client._ensure_agent_instance()
        _exhaust_gen(gen)

        assert client.agent is not None
        assert client.agent.agent_id == 2
        assert client.agent_type is not None

    def test_not_found_no_template(self) -> None:
        """When agent not found and no template, agent stays None."""
        client = _make_client()
        client.address = "0xABC"
        client.logger = MagicMock()
        client.agent_type_name = None
        client.agent_name_template = None

        mock_404 = MagicMock(status_code=404)

        def fake_http(**kwargs: Any) -> Any:
            yield
            return mock_404

        client.http_request_func = fake_http

        gen = client._ensure_agent_instance()
        _exhaust_gen(gen)

        assert client.agent is None


class TestAgentDBClientSignRequest:
    """Test AgentDBClient._sign_request."""

    def test_raises_without_signing_func(self) -> None:
        """Test ValueError when signing_func is None."""
        client = _make_client()
        client.signing_func = None
        client.agent = MagicMock()

        gen = client._sign_request("/test")
        with pytest.raises(ValueError, match="Signing function not set"):
            next(gen)

    def test_raises_when_agent_not_resolved(self) -> None:
        """Test ValueError when agent can't be found or created."""
        client = _make_client()
        client.signing_func = MagicMock()
        client.agent = None
        client.address = "0xDEAD"
        client.logger = MagicMock()
        client.agent_type_name = None
        client.agent_name_template = None

        mock_404 = MagicMock(status_code=404)

        def fake_http(**kwargs: Any) -> Any:
            yield
            return mock_404

        client.http_request_func = fake_http

        gen = client._sign_request("/test")
        with pytest.raises(ValueError, match="failed to get agent"):
            _exhaust_gen(gen)

    def test_successful_sign(self) -> None:
        """Test successful signing returns auth data."""
        client = _make_client()
        client.agent = MagicMock()
        client.agent.agent_id = 42
        client.address = "0xABC"

        def fake_sign(message: Any) -> Any:
            yield
            return "0xSIGNATURE"

        client.signing_func = fake_sign

        gen = client._sign_request("/test")
        result = _exhaust_gen(gen)

        assert result["agent_id"] == 42
        assert result["signature"] == "0xSIGNATURE"
        assert "timestamp" in result["message"]


class TestAgentDBClientRequest:
    """Test AgentDBClient._request."""

    def _setup_client(self) -> Any:
        client = _make_client()
        client.logger = MagicMock()
        client.address = "0xABC"
        return client

    def test_raises_without_http_func(self) -> None:
        """Test ValueError when http_request_func is None."""
        client = _make_client()
        client.http_request_func = None
        gen = client._request("GET", "/test")
        with pytest.raises(ValueError, match="HTTP request function not set"):
            next(gen)

    @pytest.mark.parametrize("status_code", [200, 201])
    def test_success_responses(self, status_code: Any) -> None:
        """Test 200 and 201 return parsed JSON."""
        client = self._setup_client()
        body = {"key": "value"}
        mock_resp = MagicMock(status_code=status_code, body=json.dumps(body).encode())

        def fake_http(**kwargs: Any) -> Any:
            yield
            return mock_resp

        client.http_request_func = fake_http

        gen = client._request("GET", "/test")
        result = _exhaust_gen(gen)
        assert result == body

    def test_404_returns_none(self) -> None:
        """Test 404 returns None."""
        client = self._setup_client()
        mock_resp = MagicMock(status_code=404)

        def fake_http(**kwargs: Any) -> Any:
            yield
            return mock_resp

        client.http_request_func = fake_http

        gen = client._request("GET", "/test")
        result = _exhaust_gen(gen)
        assert result is None

    def test_error_raises(self) -> None:
        """Test non-200/201/404 raises RuntimeError."""
        client = self._setup_client()
        mock_resp = MagicMock(status_code=500, text="Server Error")

        def fake_http(**kwargs: Any) -> Any:
            yield
            return mock_resp

        client.http_request_func = fake_http

        gen = client._request("GET", "/test")
        with pytest.raises(RuntimeError, match="Request failed: 500"):
            _exhaust_gen(gen)

    def test_auth_nested(self) -> None:
        """Test auth=True, nested_auth=True puts auth inside payload."""
        client = self._setup_client()
        client.agent = MagicMock()
        client.agent.agent_id = 1

        def fake_sign(msg: Any) -> Any:
            yield
            return "0xSIG"

        client.signing_func = fake_sign

        captured_kwargs = {}
        body = {"ok": True}
        mock_resp = MagicMock(status_code=200, body=json.dumps(body).encode())

        def fake_http(**kwargs: Any) -> Any:
            captured_kwargs.update(kwargs)
            yield
            return mock_resp

        client.http_request_func = fake_http

        gen = client._request(
            "POST", "/test", payload={"data": 1}, auth=True, nested_auth=True
        )
        result = _exhaust_gen(gen)

        assert result == body
        # Verify auth was nested in payload
        sent_payload = json.loads(captured_kwargs["content"])
        assert "auth" in sent_payload
        assert sent_payload["data"] == 1

    def test_auth_not_nested(self) -> None:
        """Test auth=True, nested_auth=False merges auth into payload."""
        client = self._setup_client()
        client.agent = MagicMock()
        client.agent.agent_id = 1

        def fake_sign(msg: Any) -> Any:
            yield
            return "0xSIG"

        client.signing_func = fake_sign

        captured_kwargs = {}
        body = {"ok": True}
        mock_resp = MagicMock(status_code=200, body=json.dumps(body).encode())

        def fake_http(**kwargs: Any) -> Any:
            captured_kwargs.update(kwargs)
            yield
            return mock_resp

        client.http_request_func = fake_http

        gen = client._request(
            "POST", "/test", payload={"data": 1}, auth=True, nested_auth=False
        )
        result = _exhaust_gen(gen)

        assert result == body
        sent_payload = json.loads(captured_kwargs["content"])
        assert "auth" not in sent_payload  # merged, not nested
        assert "agent_id" in sent_payload
        assert sent_payload["data"] == 1

    def test_no_payload(self) -> None:
        """Test request with no payload sends content=None."""
        client = self._setup_client()

        captured_kwargs = {}
        mock_resp = MagicMock(status_code=200, body=json.dumps({}).encode())

        def fake_http(**kwargs: Any) -> Any:
            captured_kwargs.update(kwargs)
            yield
            return mock_resp

        client.http_request_func = fake_http

        gen = client._request("GET", "/test")
        _exhaust_gen(gen)

        assert captured_kwargs["content"] is None


class TestAgentDBClientEnsureAgentTypeDefinition:
    """Test AgentDBClient._ensure_agent_type_definition."""

    def test_type_found(self) -> None:
        """When agent type exists, just fetch it."""
        client = _make_client()
        client.logger = MagicMock()
        client.agent_type_name = "memeooorr"

        type_data = {"type_id": 10, "type_name": "memeooorr", "description": "d"}
        mock_resp = MagicMock(status_code=200, body=json.dumps(type_data).encode())

        def fake_http(**kwargs: Any) -> Any:
            yield
            return mock_resp

        client.http_request_func = fake_http

        gen = client._ensure_agent_type_definition()
        _exhaust_gen(gen)

        assert client.agent_type is not None
        assert client.agent_type.type_name == "memeooorr"

    def test_type_not_found_creates(self) -> None:
        """When agent type doesn't exist, create it."""
        client = _make_client()
        client.logger = MagicMock()
        client.agent_type_name = "memeooorr"

        mock_404 = MagicMock(status_code=404)
        type_data = {
            "type_id": 10,
            "type_name": "memeooorr",
            "description": "Placeholder agent description",
        }
        mock_created = MagicMock(status_code=201, body=json.dumps(type_data).encode())

        call_count = 0
        responses = [mock_404, mock_created]

        def multi_http(**kwargs: Any) -> Any:
            nonlocal call_count
            resp = responses[call_count]
            call_count += 1
            yield
            return resp

        client.http_request_func = multi_http

        gen = client._ensure_agent_type_definition()
        _exhaust_gen(gen)

        assert client.agent_type is not None


class TestAgentDBClientEnsureAgentTypeAttributeDefinition:
    """Test AgentDBClient._ensure_agent_type_attribute_definition."""

    def test_existing_attributes_not_duplicated(self) -> None:
        """When attribute already exists, don't create it again."""
        client = _make_client()
        client.logger = MagicMock()
        client.agent_type = AgentType(type_id=10, type_name="m", description="d")

        existing = [
            AttributeDefinition(
                attr_def_id=1,
                type_id=10,
                attr_name="twitter_username",
                data_type="string",
                is_required=True,
                default_value="",
            )
        ]
        existing_json = [ad.model_dump() for ad in existing]
        mock_resp = MagicMock(status_code=200, body=json.dumps(existing_json).encode())

        def fake_http(**kwargs: Any) -> Any:
            yield
            return mock_resp

        client.http_request_func = fake_http

        new_defs = [
            AttributeDefinition(
                attr_def_id=0,
                type_id=0,
                attr_name="twitter_username",
                data_type="string",
                is_required=True,
                default_value="",
            )
        ]

        gen = client._ensure_agent_type_attribute_definition(new_defs)
        _exhaust_gen(gen)
        # No error = success. The existing attr wasn't re-created.

    def test_missing_attribute_created(self) -> None:
        """When attribute is missing, create it."""
        client = _make_client()
        client.logger = MagicMock()
        client.agent_type = AgentType(type_id=10, type_name="m", description="d")
        client.agent = MagicMock(agent_id=1)

        def fake_sign(msg: Any) -> Any:
            yield
            return "0xfakesig"

        client.signing_func = fake_sign

        # No existing attrs
        mock_empty = MagicMock(status_code=200, body=json.dumps([]).encode())
        # Created attr response
        new_attr = {
            "attr_def_id": 5,
            "type_id": 10,
            "attr_name": "twitter_username",
            "data_type": "string",
            "is_required": True,
            "default_value": "",
        }
        mock_created = MagicMock(status_code=201, body=json.dumps(new_attr).encode())

        call_count = 0
        responses = [mock_empty, mock_created]

        def multi_http(**kwargs: Any) -> Any:
            nonlocal call_count
            resp = responses[call_count]
            call_count += 1
            yield
            return resp

        client.http_request_func = multi_http

        new_defs = [
            AttributeDefinition(
                attr_def_id=0,
                type_id=0,
                attr_name="twitter_username",
                data_type="string",
                is_required=True,
                default_value="",
            )
        ]

        gen = client._ensure_agent_type_attribute_definition(new_defs)
        _exhaust_gen(gen)


class TestAgentDBClientCRUDMethods:
    """Test CRUD generator methods."""

    def _setup(self) -> Any:
        client = _make_client()
        client.logger = MagicMock()
        return client

    def test_create_agent_type(self) -> None:
        """Test create_agent_type."""
        client = self._setup()
        data = {"type_id": 1, "type_name": "t", "description": "d"}
        mock_resp = MagicMock(status_code=201, body=json.dumps(data).encode())

        def fake_http(**kwargs: Any) -> Any:
            yield
            return mock_resp

        client.http_request_func = fake_http

        gen = client.create_agent_type("t", "d")
        result = _exhaust_gen(gen)
        assert result.type_id == 1

    def test_create_agent_type_returns_none(self) -> None:
        """Test create_agent_type returns None on 404."""
        client = self._setup()
        mock_resp = MagicMock(status_code=404)

        def fake_http(**kwargs: Any) -> Any:
            yield
            return mock_resp

        client.http_request_func = fake_http

        gen = client.create_agent_type("t", "d")
        result = _exhaust_gen(gen)
        assert result is None

    def test_get_agent_type_by_type_id(self) -> None:
        """Test get_agent_type_by_type_id."""
        client = self._setup()
        data = {"type_id": 1, "type_name": "t", "description": "d"}
        mock_resp = MagicMock(status_code=200, body=json.dumps(data).encode())

        def fake_http(**kwargs: Any) -> Any:
            yield
            return mock_resp

        client.http_request_func = fake_http

        gen = client.get_agent_type_by_type_id(1)
        result = _exhaust_gen(gen)
        assert result.type_id == 1

    def test_get_agent_type_by_type_name(self) -> None:
        """Test get_agent_type_by_type_name."""
        client = self._setup()
        data = {"type_id": 1, "type_name": "t", "description": "d"}
        mock_resp = MagicMock(status_code=200, body=json.dumps(data).encode())

        def fake_http(**kwargs: Any) -> Any:
            yield
            return mock_resp

        client.http_request_func = fake_http

        gen = client.get_agent_type_by_type_name("t")
        result = _exhaust_gen(gen)
        assert result.type_name == "t"

    def test_delete_agent_type(self) -> None:
        """Test delete_agent_type."""
        client = self._setup()
        client.agent = MagicMock()
        client.agent.agent_id = 1

        def fake_sign(msg: Any) -> Any:
            yield
            return "0xSIG"

        client.signing_func = fake_sign

        data = {"type_id": 1, "type_name": "t", "description": "d"}
        mock_resp = MagicMock(status_code=200, body=json.dumps(data).encode())

        def fake_http(**kwargs: Any) -> Any:
            yield
            return mock_resp

        client.http_request_func = fake_http

        at = AgentType(type_id=1, type_name="t", description="d")
        gen = client.delete_agent_type(at)
        result = _exhaust_gen(gen)
        assert result is not None

    def test_create_agent_instance(self) -> None:
        """Test create_agent_instance."""
        client = self._setup()
        data = {
            "agent_id": 1,
            "type_id": 10,
            "agent_name": "a",
            "eth_address": "0xABC",
            "created_at": "2025-01-01T00:00:00+00:00",
        }
        mock_resp = MagicMock(status_code=201, body=json.dumps(data).encode())

        def fake_http(**kwargs: Any) -> Any:
            yield
            return mock_resp

        client.http_request_func = fake_http

        at = AgentType(type_id=10, type_name="t", description="d")
        gen = client.create_agent_instance("a", at, "0xABC")
        result = _exhaust_gen(gen)
        assert result.agent_id == 1

    def test_get_agent_instance_by_address(self) -> None:
        """Test get_agent_instance_by_address."""
        client = self._setup()
        data = {
            "agent_id": 1,
            "type_id": 10,
            "agent_name": "a",
            "eth_address": "0xABC",
            "created_at": "2025-01-01T00:00:00+00:00",
        }
        mock_resp = MagicMock(status_code=200, body=json.dumps(data).encode())

        def fake_http(**kwargs: Any) -> Any:
            yield
            return mock_resp

        client.http_request_func = fake_http

        gen = client.get_agent_instance_by_address("0xABC")
        result = _exhaust_gen(gen)
        assert result.eth_address == "0xABC"

    def test_get_agent_instances_by_type_id(self) -> None:
        """Test get_agent_instances_by_type_id."""
        client = self._setup()
        data = [
            {
                "agent_id": 1,
                "type_id": 10,
                "agent_name": "a",
                "eth_address": "0xABC",
                "created_at": "2025-01-01T00:00:00+00:00",
            }
        ]
        mock_resp = MagicMock(status_code=200, body=json.dumps(data).encode())

        def fake_http(**kwargs: Any) -> Any:
            yield
            return mock_resp

        client.http_request_func = fake_http

        gen = client.get_agent_instances_by_type_id(10)
        result = _exhaust_gen(gen)
        assert len(result) == 1

    def test_get_agent_instances_by_type_id_empty(self) -> None:
        """Test get_agent_instances_by_type_id returns empty list on 404."""
        client = self._setup()
        mock_resp = MagicMock(status_code=404)

        def fake_http(**kwargs: Any) -> Any:
            yield
            return mock_resp

        client.http_request_func = fake_http

        gen = client.get_agent_instances_by_type_id(10)
        result = _exhaust_gen(gen)
        assert result == []

    def test_delete_agent_instance(self) -> None:
        """Test delete_agent_instance."""
        client = self._setup()
        client.agent = MagicMock()
        client.agent.agent_id = 99

        def fake_sign(msg: Any) -> Any:
            yield
            return "0xSIG"

        client.signing_func = fake_sign

        data = {
            "agent_id": 1,
            "type_id": 10,
            "agent_name": "a",
            "eth_address": "0xABC",
            "created_at": "2025-01-01T00:00:00+00:00",
        }
        mock_resp = MagicMock(status_code=200, body=json.dumps(data).encode())

        def fake_http(**kwargs: Any) -> Any:
            yield
            return mock_resp

        client.http_request_func = fake_http

        ai = AgentInstance(
            agent_id=1,
            type_id=10,
            agent_name="a",
            eth_address="0xABC",
            created_at="2025-01-01T00:00:00+00:00",
        )
        gen = client.delete_agent_instance(ai)
        result = _exhaust_gen(gen)
        assert result is not None

    def test_create_attribute_definition(self) -> None:
        """Test create_attribute_definition."""
        client = self._setup()
        client.agent = MagicMock()
        client.agent.agent_id = 1

        def fake_sign(msg: Any) -> Any:
            yield
            return "0xSIG"

        client.signing_func = fake_sign

        data = {
            "attr_def_id": 5,
            "type_id": 10,
            "attr_name": "x",
            "data_type": "string",
            "is_required": False,
            "default_value": "",
        }
        mock_resp = MagicMock(status_code=201, body=json.dumps(data).encode())

        def fake_http(**kwargs: Any) -> Any:
            yield
            return mock_resp

        client.http_request_func = fake_http

        at = AgentType(type_id=10, type_name="t", description="d")
        gen = client.create_attribute_definition(at, "x", "string", "", False)
        result = _exhaust_gen(gen)
        assert result.attr_name == "x"

    def test_get_attribute_definition_by_name(self) -> None:
        """Test get_attribute_definition_by_name."""
        client = self._setup()
        data = {
            "attr_def_id": 1,
            "type_id": 10,
            "attr_name": "x",
            "data_type": "string",
            "is_required": False,
            "default_value": "",
        }
        mock_resp = MagicMock(status_code=200, body=json.dumps(data).encode())

        def fake_http(**kwargs: Any) -> Any:
            yield
            return mock_resp

        client.http_request_func = fake_http

        gen = client.get_attribute_definition_by_name("x")
        result = _exhaust_gen(gen)
        assert result.attr_name == "x"

    def test_get_attribute_definition_by_id_cached(self) -> None:
        """Test get_attribute_definition_by_id returns cached value."""
        client = self._setup()
        cached = AttributeDefinition(
            attr_def_id=1,
            type_id=10,
            attr_name="x",
            data_type="string",
            is_required=False,
            default_value="",
        )
        client._attribute_definition_cache[1] = cached

        gen = client.get_attribute_definition_by_id(1)
        result = _exhaust_gen(gen)
        assert result is cached

    def test_get_attribute_definition_by_id_fetches(self) -> None:
        """Test get_attribute_definition_by_id fetches from API and caches."""
        client = self._setup()
        data = {
            "attr_def_id": 1,
            "type_id": 10,
            "attr_name": "x",
            "data_type": "string",
            "is_required": False,
            "default_value": "",
        }
        mock_resp = MagicMock(status_code=200, body=json.dumps(data).encode())

        def fake_http(**kwargs: Any) -> Any:
            yield
            return mock_resp

        client.http_request_func = fake_http

        gen = client.get_attribute_definition_by_id(1)
        result = _exhaust_gen(gen)
        assert result.attr_name == "x"
        assert 1 in client._attribute_definition_cache

    def test_get_attribute_definition_by_id_not_found(self) -> None:
        """Test get_attribute_definition_by_id returns None when not found."""
        client = self._setup()
        mock_resp = MagicMock(status_code=404)

        def fake_http(**kwargs: Any) -> Any:
            yield
            return mock_resp

        client.http_request_func = fake_http

        gen = client.get_attribute_definition_by_id(999)
        result = _exhaust_gen(gen)
        assert result is None

    def test_get_attribute_definitions_by_agent_type(self) -> None:
        """Test get_attribute_definitions_by_agent_type."""
        client = self._setup()
        data = [
            {
                "attr_def_id": 1,
                "type_id": 10,
                "attr_name": "x",
                "data_type": "string",
                "is_required": False,
                "default_value": "",
            }
        ]
        mock_resp = MagicMock(status_code=200, body=json.dumps(data).encode())

        def fake_http(**kwargs: Any) -> Any:
            yield
            return mock_resp

        client.http_request_func = fake_http

        at = AgentType(type_id=10, type_name="t", description="d")
        gen = client.get_attribute_definitions_by_agent_type(at)
        result = _exhaust_gen(gen)
        assert len(result) == 1

    def test_get_attribute_definitions_by_agent_type_empty(self) -> None:
        """Test returns empty list on 404."""
        client = self._setup()
        mock_resp = MagicMock(status_code=404)

        def fake_http(**kwargs: Any) -> Any:
            yield
            return mock_resp

        client.http_request_func = fake_http

        at = AgentType(type_id=10, type_name="t", description="d")
        gen = client.get_attribute_definitions_by_agent_type(at)
        result = _exhaust_gen(gen)
        assert result == []

    def test_delete_attribute_definition(self) -> None:
        """Test delete_attribute_definition."""
        client = self._setup()
        client.agent = MagicMock()
        client.agent.agent_id = 1

        def fake_sign(msg: Any) -> Any:
            yield
            return "0xSIG"

        client.signing_func = fake_sign

        data = {
            "attr_def_id": 1,
            "type_id": 10,
            "attr_name": "x",
            "data_type": "string",
            "is_required": False,
            "default_value": "",
        }
        mock_resp = MagicMock(status_code=200, body=json.dumps(data).encode())

        def fake_http(**kwargs: Any) -> Any:
            yield
            return mock_resp

        client.http_request_func = fake_http

        ad = AttributeDefinition(
            attr_def_id=1,
            type_id=10,
            attr_name="x",
            data_type="string",
            is_required=False,
            default_value="",
        )
        gen = client.delete_attribute_definition(ad)
        result = _exhaust_gen(gen)
        assert result is not None

    def test_create_attribute_instance(self) -> None:
        """Test create_attribute_instance."""
        client = self._setup()
        client.agent = MagicMock()
        client.agent.agent_id = 1

        def fake_sign(msg: Any) -> Any:
            yield
            return "0xSIG"

        client.signing_func = fake_sign

        data = {
            "attribute_id": 1,
            "attr_def_id": 5,
            "agent_id": 1,
            "last_updated": "2025-01-01T00:00:00+00:00",
            "string_value": "hello",
            "integer_value": None,
            "float_value": None,
            "boolean_value": None,
            "date_value": None,
            "json_value": None,
        }
        mock_resp = MagicMock(status_code=201, body=json.dumps(data).encode())

        def fake_http(**kwargs: Any) -> Any:
            yield
            return mock_resp

        client.http_request_func = fake_http

        ai = AgentInstance(
            agent_id=1,
            type_id=10,
            agent_name="a",
            eth_address="0xABC",
            created_at="2025-01-01T00:00:00+00:00",
        )
        ad = AttributeDefinition(
            attr_def_id=5,
            type_id=10,
            attr_name="x",
            data_type="string",
            is_required=False,
            default_value="",
        )
        gen = client.create_attribute_instance(ai, ad, "hello")
        result = _exhaust_gen(gen)
        assert result.string_value == "hello"

    def test_get_attribute_instance(self) -> None:
        """Test get_attribute_instance."""
        client = self._setup()
        data = {
            "attribute_id": 1,
            "attr_def_id": 5,
            "agent_id": 1,
            "last_updated": "2025-01-01T00:00:00+00:00",
            "string_value": "hello",
            "integer_value": None,
            "float_value": None,
            "boolean_value": None,
            "date_value": None,
            "json_value": None,
        }
        mock_resp = MagicMock(status_code=200, body=json.dumps(data).encode())

        def fake_http(**kwargs: Any) -> Any:
            yield
            return mock_resp

        client.http_request_func = fake_http

        ai = AgentInstance(
            agent_id=1,
            type_id=10,
            agent_name="a",
            eth_address="0xABC",
            created_at="2025-01-01T00:00:00+00:00",
        )
        ad = AttributeDefinition(
            attr_def_id=5,
            type_id=10,
            attr_name="x",
            data_type="string",
            is_required=False,
            default_value="",
        )
        gen = client.get_attribute_instance(ai, ad)
        result = _exhaust_gen(gen)
        assert result.attribute_id == 1

    def test_update_attribute_instance(self) -> None:
        """Test update_attribute_instance."""
        client = self._setup()
        client.agent = MagicMock()
        client.agent.agent_id = 1

        def fake_sign(msg: Any) -> Any:
            yield
            return "0xSIG"

        client.signing_func = fake_sign

        data = {
            "attribute_id": 1,
            "attr_def_id": 5,
            "agent_id": 1,
            "last_updated": "2025-01-01T00:00:00+00:00",
            "string_value": "updated",
            "integer_value": None,
            "float_value": None,
            "boolean_value": None,
            "date_value": None,
            "json_value": None,
        }
        mock_resp = MagicMock(status_code=200, body=json.dumps(data).encode())

        def fake_http(**kwargs: Any) -> Any:
            yield
            return mock_resp

        client.http_request_func = fake_http

        ai = AgentInstance(
            agent_id=1,
            type_id=10,
            agent_name="a",
            eth_address="0xABC",
            created_at="2025-01-01T00:00:00+00:00",
        )
        ad = AttributeDefinition(
            attr_def_id=5,
            type_id=10,
            attr_name="x",
            data_type="string",
            is_required=False,
            default_value="",
        )
        attr_inst = AttributeInstance(
            attribute_id=1,
            attr_def_id=5,
            agent_id=1,
            last_updated="2025-01-01T00:00:00+00:00",
            string_value="old",
            integer_value=None,
            float_value=None,
            boolean_value=None,
            date_value=None,
            json_value=None,
        )
        gen = client.update_attribute_instance(ai, ad, attr_inst, "updated")
        result = _exhaust_gen(gen)
        assert result.string_value == "updated"

    def test_delete_attribute_instance(self) -> None:
        """Test delete_attribute_instance."""
        client = self._setup()
        client.agent = MagicMock()
        client.agent.agent_id = 1

        def fake_sign(msg: Any) -> Any:
            yield
            return "0xSIG"

        client.signing_func = fake_sign

        data = {
            "attribute_id": 1,
            "attr_def_id": 5,
            "agent_id": 1,
            "last_updated": "2025-01-01T00:00:00+00:00",
            "string_value": "x",
            "integer_value": None,
            "float_value": None,
            "boolean_value": None,
            "date_value": None,
            "json_value": None,
        }
        mock_resp = MagicMock(status_code=200, body=json.dumps(data).encode())

        def fake_http(**kwargs: Any) -> Any:
            yield
            return mock_resp

        client.http_request_func = fake_http

        attr_inst = AttributeInstance(
            attribute_id=1,
            attr_def_id=5,
            agent_id=1,
            last_updated="2025-01-01T00:00:00+00:00",
            string_value="x",
            integer_value=None,
            float_value=None,
            boolean_value=None,
            date_value=None,
            json_value=None,
        )
        gen = client.delete_attribute_instance(attr_inst)
        result = _exhaust_gen(gen)
        assert result is not None

    def test_get_all_agent_instance_attributes_raw(self) -> None:
        """Test get_all_agent_instance_attributes_raw."""
        client = self._setup()
        client.agent = MagicMock()
        client.agent.agent_id = 1

        def fake_sign(msg: Any) -> Any:
            yield
            return "0xSIG"

        client.signing_func = fake_sign

        data = [{"key": "val"}]
        mock_resp = MagicMock(status_code=200, body=json.dumps(data).encode())

        def fake_http(**kwargs: Any) -> Any:
            yield
            return mock_resp

        client.http_request_func = fake_http

        ai = AgentInstance(
            agent_id=1,
            type_id=10,
            agent_name="a",
            eth_address="0xABC",
            created_at="2025-01-01T00:00:00+00:00",
        )
        gen = client.get_all_agent_instance_attributes_raw(ai)
        result = _exhaust_gen(gen)
        assert result == data

    def test_parse_attribute_instance(self) -> None:
        """Test parse_attribute_instance."""
        client = self._setup()
        # Cache the attribute definition
        attr_def = AttributeDefinition(
            attr_def_id=5,
            type_id=10,
            attr_name="test_attr",
            data_type="string",
            is_required=False,
            default_value="",
        )
        client._attribute_definition_cache[5] = attr_def

        attr_inst = AttributeInstance(
            attribute_id=1,
            attr_def_id=5,
            agent_id=1,
            last_updated="2025-01-01T00:00:00+00:00",
            string_value="hello",
            integer_value=None,
            float_value=None,
            boolean_value=None,
            date_value=None,
            json_value=None,
        )
        gen = client.parse_attribute_instance(attr_inst)
        result = _exhaust_gen(gen)
        assert result["attr_name"] == "test_attr"
        assert result["attr_value"] == "hello"

    def test_get_all_agent_instance_attributes_parsed(self) -> None:
        """Test get_all_agent_instance_attributes_parsed."""
        client = self._setup()
        client.agent = MagicMock()
        client.agent.agent_id = 1

        def fake_sign(msg: Any) -> Any:
            yield
            return "0xSIG"

        client.signing_func = fake_sign

        # Cache attr def
        attr_def = AttributeDefinition(
            attr_def_id=5,
            type_id=10,
            attr_name="test_attr",
            data_type="string",
            is_required=False,
            default_value="",
        )
        client._attribute_definition_cache[5] = attr_def

        raw_data = [
            {
                "attribute_id": 1,
                "attr_def_id": 5,
                "agent_id": 1,
                "last_updated": "2025-01-01T00:00:00+00:00",
                "string_value": "hello",
                "integer_value": None,
                "float_value": None,
                "boolean_value": None,
                "date_value": None,
                "json_value": None,
            }
        ]
        mock_resp = MagicMock(status_code=200, body=json.dumps(raw_data).encode())

        def fake_http(**kwargs: Any) -> Any:
            yield
            return mock_resp

        client.http_request_func = fake_http

        ai = AgentInstance(
            agent_id=1,
            type_id=10,
            agent_name="a",
            eth_address="0xABC",
            created_at="2025-01-01T00:00:00+00:00",
        )
        gen = client.get_all_agent_instance_attributes_parsed(ai)
        result = _exhaust_gen(gen)
        assert len(result) == 1
        assert result[0]["attr_name"] == "test_attr"

    def test_update_or_create_agent_attribute_update(self) -> None:
        """Test update_or_create_agent_attribute when attr_instance exists."""
        client = self._setup()
        client.agent = MagicMock()
        client.agent.agent_id = 1

        def fake_sign(msg: Any) -> Any:
            yield
            return "0xSIG"

        client.signing_func = fake_sign

        attr_def_data = {
            "attr_def_id": 5,
            "type_id": 10,
            "attr_name": "x",
            "data_type": "string",
            "is_required": False,
            "default_value": "",
        }
        attr_inst_data = {
            "attribute_id": 1,
            "attr_def_id": 5,
            "agent_id": 1,
            "last_updated": "2025-01-01T00:00:00+00:00",
            "string_value": "old",
            "integer_value": None,
            "float_value": None,
            "boolean_value": None,
            "date_value": None,
            "json_value": None,
        }
        updated_data = dict(attr_inst_data, string_value="new")

        call_count = 0
        responses = [
            MagicMock(
                status_code=200, body=json.dumps(attr_def_data).encode()
            ),  # get_attribute_definition_by_name
            MagicMock(
                status_code=200, body=json.dumps(attr_inst_data).encode()
            ),  # get_attribute_instance
            MagicMock(
                status_code=200, body=json.dumps(updated_data).encode()
            ),  # update_attribute_instance
        ]

        def multi_http(**kwargs: Any) -> Any:
            nonlocal call_count
            resp = responses[call_count]
            call_count += 1
            yield
            return resp

        client.http_request_func = multi_http

        gen = client.update_or_create_agent_attribute("x", "new")
        result = _exhaust_gen(gen)
        assert result is True

    def test_update_or_create_agent_attribute_create(self) -> None:
        """Test update_or_create_agent_attribute when attr_instance doesn't exist."""
        client = self._setup()
        client.agent = MagicMock()
        client.agent.agent_id = 1

        def fake_sign(msg: Any) -> Any:
            yield
            return "0xSIG"

        client.signing_func = fake_sign

        attr_def_data = {
            "attr_def_id": 5,
            "type_id": 10,
            "attr_name": "x",
            "data_type": "string",
            "is_required": False,
            "default_value": "",
        }
        created_data = {
            "attribute_id": 99,
            "attr_def_id": 5,
            "agent_id": 1,
            "last_updated": "2025-01-01T00:00:00+00:00",
            "string_value": "new",
            "integer_value": None,
            "float_value": None,
            "boolean_value": None,
            "date_value": None,
            "json_value": None,
        }

        call_count = 0
        responses = [
            MagicMock(
                status_code=200, body=json.dumps(attr_def_data).encode()
            ),  # get_attribute_definition_by_name
            MagicMock(status_code=404),  # get_attribute_instance returns None
            MagicMock(
                status_code=201, body=json.dumps(created_data).encode()
            ),  # create_attribute_instance
        ]

        def multi_http(**kwargs: Any) -> Any:
            nonlocal call_count
            resp = responses[call_count]
            call_count += 1
            yield
            return resp

        client.http_request_func = multi_http

        gen = client.update_or_create_agent_attribute("x", "new")
        result = _exhaust_gen(gen)
        assert result is True

    def test_update_or_create_agent_attribute_update_fails(self) -> None:
        """Test update_or_create returns False when update fails."""
        client = self._setup()
        client.agent = MagicMock()
        client.agent.agent_id = 1

        def fake_sign(msg: Any) -> Any:
            yield
            return "0xSIG"

        client.signing_func = fake_sign

        attr_def_data = {
            "attr_def_id": 5,
            "type_id": 10,
            "attr_name": "x",
            "data_type": "string",
            "is_required": False,
            "default_value": "",
        }
        attr_inst_data = {
            "attribute_id": 1,
            "attr_def_id": 5,
            "agent_id": 1,
            "last_updated": "2025-01-01T00:00:00+00:00",
            "string_value": "old",
            "integer_value": None,
            "float_value": None,
            "boolean_value": None,
            "date_value": None,
            "json_value": None,
        }

        call_count = 0
        responses = [
            MagicMock(status_code=200, body=json.dumps(attr_def_data).encode()),
            MagicMock(status_code=200, body=json.dumps(attr_inst_data).encode()),
            MagicMock(status_code=404),  # update fails
        ]

        def multi_http(**kwargs: Any) -> Any:
            nonlocal call_count
            resp = responses[call_count]
            call_count += 1
            yield
            return resp

        client.http_request_func = multi_http

        gen = client.update_or_create_agent_attribute("x", "new")
        result = _exhaust_gen(gen)
        assert result is False

    def test_update_or_create_agent_attribute_create_fails(self) -> None:
        """Test update_or_create returns False when create fails."""
        client = self._setup()
        client.agent = MagicMock()
        client.agent.agent_id = 1

        def fake_sign(msg: Any) -> Any:
            yield
            return "0xSIG"

        client.signing_func = fake_sign

        attr_def_data = {
            "attr_def_id": 5,
            "type_id": 10,
            "attr_name": "x",
            "data_type": "string",
            "is_required": False,
            "default_value": "",
        }

        call_count = 0
        responses = [
            MagicMock(status_code=200, body=json.dumps(attr_def_data).encode()),
            MagicMock(status_code=404),  # attr instance not found
            MagicMock(status_code=404),  # create also fails
        ]

        def multi_http(**kwargs: Any) -> Any:
            nonlocal call_count
            resp = responses[call_count]
            call_count += 1
            yield
            return resp

        client.http_request_func = multi_http

        gen = client.update_or_create_agent_attribute("x", "new")
        result = _exhaust_gen(gen)
        assert result is False
