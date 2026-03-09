# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023-2025 Valory AG
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

"""Tests for agent_db_client.py."""

import json
from datetime import datetime, timezone
from typing import Any, Generator
from unittest.mock import MagicMock

import pytest

from packages.valory.skills.agent_db_abci.agent_db_client import AgentDBClient
from packages.valory.skills.agent_db_abci.agent_db_models import (
    AgentInstance,
    AgentType,
    AttributeDefinition,
    AttributeInstance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NOW = datetime.now(timezone.utc)

AGENT_TYPE_DICT = {"type_id": 1, "type_name": "test_type", "description": "desc"}
AGENT_TYPE = AgentType(**AGENT_TYPE_DICT)

AGENT_INSTANCE_DICT = {
    "agent_id": 10,
    "type_id": 1,
    "agent_name": "agent-10",
    "eth_address": "0xABC",
    "created_at": NOW.isoformat(),
}
AGENT_INSTANCE = AgentInstance(**AGENT_INSTANCE_DICT)

ATTR_DEF_DICT = {
    "attr_def_id": 100,
    "type_id": 1,
    "attr_name": "my_attr",
    "data_type": "string",
    "is_required": False,
    "default_value": "",
}
ATTR_DEF = AttributeDefinition(**ATTR_DEF_DICT)

ATTR_INSTANCE_DICT = {
    "attribute_id": 200,
    "attr_def_id": 100,
    "agent_id": 10,
    "last_updated": NOW.isoformat(),
    "string_value": "hello",
    "integer_value": None,
    "float_value": None,
    "boolean_value": None,
    "date_value": None,
    "json_value": None,
}
ATTR_INSTANCE = AttributeInstance(**ATTR_INSTANCE_DICT)


def _exhaust(gen: Generator) -> Any:
    """Drive a generator to completion and return its return value."""
    try:
        while True:
            next(gen)
    except StopIteration as exc:
        return exc.value


def make_client(base_url: str = "https://example.com/") -> AgentDBClient:
    """Create an AgentDBClient with mocked Model.__init__."""
    client = AgentDBClient.__new__(AgentDBClient)
    client.base_url = base_url.rstrip("/")
    client._attribute_definition_cache = {}
    client.agent = None
    client.agent_type = None
    client.address = None
    client.signing_func = None
    client.http_request_func = None
    client.logger = MagicMock()
    client.agent_type_name = None
    client.agent_name_template = None
    return client


def _make_response(status_code: int, body_dict: Any = None):
    """Create a mock HTTP response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.body = json.dumps(body_dict).encode() if body_dict is not None else b""
    resp.text = ""
    return resp


def _http_gen(*responses):
    """Return a generator-based http_request_func that yields successive responses."""
    idx = [0]

    def http_func(**kwargs):
        r = responses[idx[0]]
        idx[0] += 1
        return r
        yield  # noqa: E501  make it a generator

    return http_func


def _sign_gen():
    """Return a generator-based signing_func that always yields '0xSIG'."""

    def sign_func(msg):
        return "0xSIG"
        yield  # noqa: E501

    return sign_func


# ---------------------------------------------------------------------------
# Tests: __init__ and initialize
# ---------------------------------------------------------------------------


class TestAgentDBClientInit:
    """Tests for AgentDBClient initialization and initialize method."""

    def test_base_url_trailing_slash_stripped(self) -> None:
        """Test that trailing slash is stripped from base_url."""
        client = make_client("https://example.com/")
        assert client.base_url == "https://example.com"

    def test_base_url_no_trailing_slash(self) -> None:
        """Test base_url without trailing slash."""
        client = make_client("https://example.com")
        assert client.base_url == "https://example.com"

    def test_initial_state(self) -> None:
        """Test initial state of client."""
        client = make_client()
        assert client.agent is None
        assert client.agent_type is None
        assert client.address is None
        assert client.signing_func is None
        assert client.http_request_func is None
        assert client._attribute_definition_cache == {}

    def test_initialize(self) -> None:
        """Test initialize sets all external dependencies."""
        client = make_client()
        mock_http = MagicMock()
        mock_sign = MagicMock()
        mock_logger = MagicMock()

        client.initialize(
            address="0xabc",
            http_request_func=mock_http,
            signing_func=mock_sign,
            logger=mock_logger,
            agent_type_name="memeooorr",
            agent_name_template="agent-{address}",
        )

        assert client.address == "0xabc"
        assert client.http_request_func is mock_http
        assert client.signing_func is mock_sign
        assert client.logger is mock_logger
        assert client.agent_type_name == "memeooorr"
        assert client.agent_name_template == "agent-{address}"

    def test_initialize_defaults(self) -> None:
        """Test initialize with default optional parameters."""
        client = make_client()
        client.initialize(
            address="0x1",
            http_request_func=MagicMock(),
            signing_func=MagicMock(),
            logger=MagicMock(),
        )
        assert client.agent_type_name is None
        assert client.agent_name_template is None


# ---------------------------------------------------------------------------
# Tests: cast_attribute_value (non-generator)
# ---------------------------------------------------------------------------


class TestCastAttributeValue:
    """Tests for cast_attribute_value (non-generator method)."""

    @pytest.fixture()
    def client(self) -> AgentDBClient:
        """Create a client for testing."""
        return make_client()

    def _make_attr_def(self, data_type: str) -> AttributeDefinition:
        """Helper to create AttributeDefinition with given data_type."""
        return AttributeDefinition(
            attr_def_id=1, type_id=1, attr_name="test",
            data_type=data_type, is_required=False, default_value="",
        )

    def test_cast_string(self, client: AgentDBClient) -> None:
        """Test casting to string."""
        result = client.cast_attribute_value(123, self._make_attr_def("string"))
        assert result == "123"
        assert isinstance(result, str)

    def test_cast_integer(self, client: AgentDBClient) -> None:
        """Test casting to integer."""
        result = client.cast_attribute_value("42", self._make_attr_def("integer"))
        assert result == 42

    def test_cast_float(self, client: AgentDBClient) -> None:
        """Test casting to float."""
        result = client.cast_attribute_value("3.14", self._make_attr_def("float"))
        assert result == pytest.approx(3.14)

    def test_cast_boolean_true(self, client: AgentDBClient) -> None:
        """Test casting truthy value to boolean."""
        assert client.cast_attribute_value(1, self._make_attr_def("boolean")) is True

    def test_cast_boolean_false(self, client: AgentDBClient) -> None:
        """Test casting falsy value to boolean."""
        assert client.cast_attribute_value(0, self._make_attr_def("boolean")) is False

    def test_cast_date(self, client: AgentDBClient) -> None:
        """Test casting to date."""
        result = client.cast_attribute_value(
            "2025-01-15T12:00:00+00:00", self._make_attr_def("date"),
        )
        assert result == datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

    def test_cast_date_z_suffix(self, client: AgentDBClient) -> None:
        """Test casting date with Z suffix."""
        result = client.cast_attribute_value(
            "2025-01-15T12:00:00Z", self._make_attr_def("date"),
        )
        assert result == datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

    def test_cast_json_passthrough(self, client: AgentDBClient) -> None:
        """Test that json type passes through unchanged."""
        data = {"key": "value"}
        result = client.cast_attribute_value(data, self._make_attr_def("json"))
        assert result is data

    def test_cast_unknown_type_passthrough(self, client: AgentDBClient) -> None:
        """Test that unknown data_type passes through unchanged."""
        result = client.cast_attribute_value("anything", self._make_attr_def("unknown"))
        assert result == "anything"


# ---------------------------------------------------------------------------
# Tests: attribute definition cache
# ---------------------------------------------------------------------------


class TestAttributeDefinitionCache:
    """Tests for the attribute definition cache."""

    def test_cache_starts_empty(self) -> None:
        """Test that cache starts empty."""
        assert make_client()._attribute_definition_cache == {}

    def test_cache_can_be_populated(self) -> None:
        """Test that cache can be populated manually."""
        client = make_client()
        client._attribute_definition_cache[5] = ATTR_DEF
        assert client._attribute_definition_cache[5].attr_name == "my_attr"


# ---------------------------------------------------------------------------
# Tests: _sign_request
# ---------------------------------------------------------------------------


class TestSignRequest:
    """Tests for _sign_request validation and happy path."""

    def test_raises_without_signing_func(self) -> None:
        """Test that _sign_request raises if signing_func is not set."""
        client = make_client()
        client.signing_func = None
        gen = client._sign_request("/api/test")
        with pytest.raises(ValueError, match="Signing function not set"):
            gen.send(None)

    def test_raises_when_agent_is_none(self) -> None:
        """Test that _sign_request raises if agent cannot be resolved."""
        client = make_client()
        client.signing_func = MagicMock()
        client.address = "0xdead"

        def mock_ensure():
            return
            yield  # noqa: E501

        client._ensure_agent_instance = mock_ensure
        gen = client._sign_request("/api/test")
        with pytest.raises(ValueError, match="failed to get agent"):
            gen.send(None)

    def test_success_returns_auth_data(self) -> None:
        """Test successful signing returns auth_data dict."""
        client = make_client()
        client.signing_func = _sign_gen()
        client.agent = AGENT_INSTANCE
        client.address = "0xABC"
        gen = client._sign_request("/api/test")
        result = _exhaust(gen)
        assert result["agent_id"] == AGENT_INSTANCE.agent_id
        assert result["signature"] == "0xSIG"
        assert "timestamp:" in result["message"]
        assert "endpoint:/api/test" in result["message"]


# ---------------------------------------------------------------------------
# Tests: _request
# ---------------------------------------------------------------------------


class TestRequest:
    """Tests for the _request generator method."""

    def test_raises_without_http_func(self) -> None:
        """Test raises when http_request_func is None."""
        client = make_client()
        gen = client._request("GET", "/api/test")
        with pytest.raises(ValueError, match="HTTP request function not set"):
            gen.send(None)

    def test_get_200(self) -> None:
        """Test a successful GET returning 200."""
        client = make_client()
        client.http_request_func = _http_gen(_make_response(200, {"key": "val"}))
        result = _exhaust(client._request("GET", "/api/test"))
        assert result == {"key": "val"}

    def test_post_201(self) -> None:
        """Test a POST returning 201."""
        client = make_client()
        client.http_request_func = _http_gen(_make_response(201, {"id": 1}))
        result = _exhaust(client._request("POST", "/api/test", payload={"a": "b"}))
        assert result == {"id": 1}

    def test_404_returns_none(self) -> None:
        """Test a 404 response returns None."""
        client = make_client()
        client.http_request_func = _http_gen(_make_response(404))
        result = _exhaust(client._request("GET", "/api/missing"))
        assert result is None

    def test_500_raises(self) -> None:
        """Test a 500 raises an exception."""
        resp = _make_response(500)
        resp.text = "Internal Server Error"
        client = make_client()
        client.http_request_func = _http_gen(resp)
        with pytest.raises(Exception, match="Request failed: 500"):
            _exhaust(client._request("GET", "/api/err"))

    def test_auth_nested(self) -> None:
        """Test request with auth=True and nested_auth=True puts auth inside payload."""
        captured = {}

        def http_func(**kwargs):
            captured.update(kwargs)
            return _make_response(200, {"ok": True})
            yield  # noqa: E501

        client = make_client()
        client.http_request_func = http_func
        client.signing_func = _sign_gen()
        client.agent = AGENT_INSTANCE
        client.address = "0xABC"
        result = _exhaust(
            client._request("POST", "/ep", payload={"d": 1}, auth=True, nested_auth=True)
        )
        assert result == {"ok": True}
        body = json.loads(captured["content"])
        assert "auth" in body
        assert body["d"] == 1

    def test_auth_not_nested(self) -> None:
        """Test request with auth=True and nested_auth=False merges auth at top level."""
        captured = {}

        def http_func(**kwargs):
            captured.update(kwargs)
            return _make_response(200, {"ok": True})
            yield  # noqa: E501

        client = make_client()
        client.http_request_func = http_func
        client.signing_func = _sign_gen()
        client.agent = AGENT_INSTANCE
        client.address = "0xABC"
        result = _exhaust(
            client._request("POST", "/ep", payload={"d": 1}, auth=True, nested_auth=False)
        )
        assert result == {"ok": True}
        body = json.loads(captured["content"])
        assert "agent_id" in body
        assert "signature" in body
        assert "auth" not in body

    def test_auth_with_no_initial_payload(self) -> None:
        """Test auth=True with payload=None creates an empty dict for payload."""
        captured = {}

        def http_func(**kwargs):
            captured.update(kwargs)
            return _make_response(200, {"ok": True})
            yield  # noqa: E501

        client = make_client()
        client.http_request_func = http_func
        client.signing_func = _sign_gen()
        client.agent = AGENT_INSTANCE
        client.address = "0xABC"
        _exhaust(client._request("POST", "/ep", payload=None, auth=True, nested_auth=True))
        body = json.loads(captured["content"])
        assert "auth" in body

    def test_request_no_payload_sends_none_content(self) -> None:
        """Test that no payload results in content=None."""
        captured = {}

        def http_func(**kwargs):
            captured.update(kwargs)
            return _make_response(200, [])
            yield  # noqa: E501

        client = make_client()
        client.http_request_func = http_func
        _exhaust(client._request("GET", "/ep"))
        assert captured["content"] is None

    def test_params_passed_through(self) -> None:
        """Test that params are forwarded."""
        captured = {}

        def http_func(**kwargs):
            captured.update(kwargs)
            return _make_response(200, [])
            yield  # noqa: E501

        client = make_client()
        client.http_request_func = http_func
        _exhaust(client._request("GET", "/ep", params={"skip": 0}))
        assert captured["parameters"] == {"skip": 0}


# ---------------------------------------------------------------------------
# Tests: _ensure_agent_instance
# ---------------------------------------------------------------------------


class TestEnsureAgentInstance:
    """Tests for _ensure_agent_instance."""

    def test_returns_immediately_when_agent_set(self) -> None:
        """Test early return when self.agent is already set."""
        client = make_client()
        client.agent = AGENT_INSTANCE
        _exhaust(client._ensure_agent_instance())
        assert client.agent is AGENT_INSTANCE

    def test_fetches_agent_and_type(self) -> None:
        """Test fetches agent by address, then fetches its type."""
        client = make_client()
        client.address = "0xABC"
        client.http_request_func = _http_gen(
            _make_response(200, AGENT_INSTANCE_DICT),  # get_agent_instance_by_address
            _make_response(200, AGENT_TYPE_DICT),       # get_agent_type_by_type_id
        )
        _exhaust(client._ensure_agent_instance())
        assert client.agent is not None
        assert client.agent.agent_id == 10
        assert client.agent_type is not None
        assert client.agent_type.type_id == 1

    def test_registers_when_not_found(self) -> None:
        """Test registers agent when address lookup returns 404."""
        client = make_client()
        client.address = "0xABC"
        client.agent_type_name = "memeooorr"
        client.agent_name_template = "agent-{address}"
        client.http_request_func = _http_gen(
            _make_response(404),                        # get_agent_instance_by_address
            _make_response(200, AGENT_TYPE_DICT),       # get_agent_type_by_type_name
            _make_response(201, AGENT_INSTANCE_DICT),   # create_agent_instance
        )
        _exhaust(client._ensure_agent_instance())
        assert client.agent is not None
        assert client.agent_type is not None

    def test_no_register_when_type_name_missing(self) -> None:
        """Test does not register when agent_type_name is None."""
        client = make_client()
        client.address = "0xABC"
        client.http_request_func = _http_gen(_make_response(404))
        _exhaust(client._ensure_agent_instance())
        assert client.agent is None

    def test_no_register_when_template_missing(self) -> None:
        """Test does not register when agent_name_template is None but type_name set."""
        client = make_client()
        client.address = "0xABC"
        client.agent_type_name = "memeooorr"
        client.agent_name_template = None  # template missing
        client.http_request_func = _http_gen(_make_response(404))
        _exhaust(client._ensure_agent_instance())
        assert client.agent is None


# ---------------------------------------------------------------------------
# Tests: _ensure_agent_type_definition
# ---------------------------------------------------------------------------


class TestEnsureAgentTypeDefinition:
    """Tests for _ensure_agent_type_definition."""

    def test_fetches_existing(self) -> None:
        """Test fetches existing agent type."""
        client = make_client()
        client.agent_type_name = "memeooorr"
        client.http_request_func = _http_gen(_make_response(200, AGENT_TYPE_DICT))
        _exhaust(client._ensure_agent_type_definition())
        assert client.agent_type is not None

    def test_creates_when_not_found(self) -> None:
        """Test creates agent type when lookup returns 404."""
        client = make_client()
        client.agent_type_name = "memeooorr"
        client.http_request_func = _http_gen(
            _make_response(404),                   # get_agent_type_by_type_name
            _make_response(201, AGENT_TYPE_DICT),  # create_agent_type
        )
        _exhaust(client._ensure_agent_type_definition("desc"))
        assert client.agent_type is not None


# ---------------------------------------------------------------------------
# Tests: _ensure_agent_type_attribute_definition
# ---------------------------------------------------------------------------


class TestEnsureAgentTypeAttributeDefinition:
    """Tests for _ensure_agent_type_attribute_definition."""

    def test_creates_missing_definitions(self) -> None:
        """Test creates attribute definitions that don't already exist."""
        new_def = AttributeDefinition(
            attr_def_id=0, type_id=1, attr_name="new_attr",
            data_type="string", is_required=False, default_value="",
        )
        new_def_dict = {**ATTR_DEF_DICT, "attr_name": "new_attr", "attr_def_id": 101}

        client = make_client()
        client.agent = AGENT_INSTANCE
        client.agent_type = AGENT_TYPE
        client.signing_func = _sign_gen()
        client.http_request_func = _http_gen(
            _make_response(200, [ATTR_DEF_DICT]),  # get existing
            _make_response(201, new_def_dict),      # create new_attr
        )
        _exhaust(client._ensure_agent_type_attribute_definition([ATTR_DEF, new_def]))

    def test_skips_when_all_exist(self) -> None:
        """Test does not create any when all definitions already exist."""
        call_count = [0]

        def http_func(**kwargs):
            call_count[0] += 1
            return _make_response(200, [ATTR_DEF_DICT])
            yield  # noqa: E501

        client = make_client()
        client.agent_type = AGENT_TYPE
        client.http_request_func = http_func
        _exhaust(client._ensure_agent_type_attribute_definition([ATTR_DEF]))
        assert call_count[0] == 1  # only the GET


# ---------------------------------------------------------------------------
# Tests: CRUD Agent Type
# ---------------------------------------------------------------------------


class TestAgentTypeCRUD:
    """Tests for agent type CRUD methods."""

    def test_create_agent_type(self) -> None:
        """Test create_agent_type returns AgentType."""
        client = make_client()
        client.http_request_func = _http_gen(_make_response(201, AGENT_TYPE_DICT))
        result = _exhaust(client.create_agent_type("test_type", "desc"))
        assert isinstance(result, AgentType)
        assert result.type_name == "test_type"

    def test_create_agent_type_none_on_404(self) -> None:
        """Test create_agent_type returns None on 404."""
        client = make_client()
        client.http_request_func = _http_gen(_make_response(404))
        result = _exhaust(client.create_agent_type("x", "y"))
        assert result is None

    def test_get_agent_type_by_type_id(self) -> None:
        """Test get_agent_type_by_type_id."""
        client = make_client()
        client.http_request_func = _http_gen(_make_response(200, AGENT_TYPE_DICT))
        result = _exhaust(client.get_agent_type_by_type_id(1))
        assert isinstance(result, AgentType)

    def test_get_agent_type_by_type_id_none(self) -> None:
        """Test get_agent_type_by_type_id returns None on 404."""
        client = make_client()
        client.http_request_func = _http_gen(_make_response(404))
        result = _exhaust(client.get_agent_type_by_type_id(999))
        assert result is None

    def test_get_agent_type_by_type_name(self) -> None:
        """Test get_agent_type_by_type_name."""
        client = make_client()
        client.http_request_func = _http_gen(_make_response(200, AGENT_TYPE_DICT))
        result = _exhaust(client.get_agent_type_by_type_name("test_type"))
        assert isinstance(result, AgentType)

    def test_get_agent_type_by_type_name_none(self) -> None:
        """Test get_agent_type_by_type_name returns None on 404."""
        client = make_client()
        client.http_request_func = _http_gen(_make_response(404))
        result = _exhaust(client.get_agent_type_by_type_name("missing"))
        assert result is None

    def test_delete_agent_type(self) -> None:
        """Test delete_agent_type."""
        client = make_client()
        client.signing_func = _sign_gen()
        client.agent = AGENT_INSTANCE
        client.address = "0xABC"
        client.http_request_func = _http_gen(_make_response(200, AGENT_TYPE_DICT))
        result = _exhaust(client.delete_agent_type(AGENT_TYPE))
        assert isinstance(result, AgentType)

    def test_delete_agent_type_none_on_404(self) -> None:
        """Test delete_agent_type returns None on 404."""
        client = make_client()
        client.signing_func = _sign_gen()
        client.agent = AGENT_INSTANCE
        client.address = "0xABC"
        client.http_request_func = _http_gen(_make_response(404))
        result = _exhaust(client.delete_agent_type(AGENT_TYPE))
        assert result is None


# ---------------------------------------------------------------------------
# Tests: CRUD Agent Instance
# ---------------------------------------------------------------------------


class TestAgentInstanceCRUD:
    """Tests for agent instance CRUD methods."""

    def test_create_agent_instance(self) -> None:
        """Test create_agent_instance."""
        client = make_client()
        client.http_request_func = _http_gen(_make_response(201, AGENT_INSTANCE_DICT))
        result = _exhaust(client.create_agent_instance("agent-10", AGENT_TYPE, "0xABC"))
        assert isinstance(result, AgentInstance)
        assert result.agent_id == 10

    def test_create_agent_instance_none(self) -> None:
        """Test create_agent_instance returns None on 404."""
        client = make_client()
        client.http_request_func = _http_gen(_make_response(404))
        result = _exhaust(client.create_agent_instance("x", AGENT_TYPE, "0x"))
        assert result is None

    def test_get_agent_instance_by_address(self) -> None:
        """Test get_agent_instance_by_address."""
        client = make_client()
        client.http_request_func = _http_gen(_make_response(200, AGENT_INSTANCE_DICT))
        result = _exhaust(client.get_agent_instance_by_address("0xABC"))
        assert isinstance(result, AgentInstance)

    def test_get_agent_instance_by_address_none(self) -> None:
        """Test get_agent_instance_by_address returns None on 404."""
        client = make_client()
        client.http_request_func = _http_gen(_make_response(404))
        result = _exhaust(client.get_agent_instance_by_address("0xMISSING"))
        assert result is None

    def test_get_agent_instances_by_type_id(self) -> None:
        """Test get_agent_instances_by_type_id returns list."""
        client = make_client()
        client.http_request_func = _http_gen(_make_response(200, [AGENT_INSTANCE_DICT]))
        result = _exhaust(client.get_agent_instances_by_type_id(1))
        assert len(result) == 1
        assert isinstance(result[0], AgentInstance)

    def test_get_agent_instances_by_type_id_empty(self) -> None:
        """Test get_agent_instances_by_type_id returns empty list on 404."""
        client = make_client()
        client.http_request_func = _http_gen(_make_response(404))
        result = _exhaust(client.get_agent_instances_by_type_id(999))
        assert result == []

    def test_delete_agent_instance(self) -> None:
        """Test delete_agent_instance."""
        client = make_client()
        client.signing_func = _sign_gen()
        client.agent = AGENT_INSTANCE
        client.address = "0xABC"
        client.http_request_func = _http_gen(_make_response(200, AGENT_INSTANCE_DICT))
        result = _exhaust(client.delete_agent_instance(AGENT_INSTANCE))
        assert isinstance(result, AgentInstance)

    def test_delete_agent_instance_none(self) -> None:
        """Test delete_agent_instance returns None on 404."""
        client = make_client()
        client.signing_func = _sign_gen()
        client.agent = AGENT_INSTANCE
        client.address = "0xABC"
        client.http_request_func = _http_gen(_make_response(404))
        result = _exhaust(client.delete_agent_instance(AGENT_INSTANCE))
        assert result is None


# ---------------------------------------------------------------------------
# Tests: CRUD Attribute Definition
# ---------------------------------------------------------------------------


class TestAttributeDefinitionCRUD:
    """Tests for attribute definition CRUD methods."""

    def test_create_attribute_definition(self) -> None:
        """Test create_attribute_definition."""
        client = make_client()
        client.signing_func = _sign_gen()
        client.agent = AGENT_INSTANCE
        client.address = "0xABC"
        client.http_request_func = _http_gen(_make_response(201, ATTR_DEF_DICT))
        result = _exhaust(
            client.create_attribute_definition(AGENT_TYPE, "my_attr", "string", "", False)
        )
        assert isinstance(result, AttributeDefinition)

    def test_create_attribute_definition_none(self) -> None:
        """Test create_attribute_definition returns None on 404."""
        client = make_client()
        client.signing_func = _sign_gen()
        client.agent = AGENT_INSTANCE
        client.address = "0xABC"
        client.http_request_func = _http_gen(_make_response(404))
        result = _exhaust(
            client.create_attribute_definition(AGENT_TYPE, "x", "string", "", False)
        )
        assert result is None

    def test_get_attribute_definition_by_name(self) -> None:
        """Test get_attribute_definition_by_name."""
        client = make_client()
        client.http_request_func = _http_gen(_make_response(200, ATTR_DEF_DICT))
        result = _exhaust(client.get_attribute_definition_by_name("my_attr"))
        assert isinstance(result, AttributeDefinition)

    def test_get_attribute_definition_by_name_none(self) -> None:
        """Test get_attribute_definition_by_name returns None on 404."""
        client = make_client()
        client.http_request_func = _http_gen(_make_response(404))
        result = _exhaust(client.get_attribute_definition_by_name("missing"))
        assert result is None

    def test_get_attribute_definition_by_id_cache_miss(self) -> None:
        """Test get_attribute_definition_by_id with cache miss fetches and caches."""
        client = make_client()
        client.http_request_func = _http_gen(_make_response(200, ATTR_DEF_DICT))
        result = _exhaust(client.get_attribute_definition_by_id(100))
        assert isinstance(result, AttributeDefinition)
        assert 100 in client._attribute_definition_cache

    def test_get_attribute_definition_by_id_cache_hit(self) -> None:
        """Test get_attribute_definition_by_id with cache hit returns directly."""
        client = make_client()
        client._attribute_definition_cache[100] = ATTR_DEF
        result = _exhaust(client.get_attribute_definition_by_id(100))
        assert result is ATTR_DEF

    def test_get_attribute_definition_by_id_not_found(self) -> None:
        """Test get_attribute_definition_by_id returns None on 404."""
        client = make_client()
        client.http_request_func = _http_gen(_make_response(404))
        result = _exhaust(client.get_attribute_definition_by_id(999))
        assert result is None

    def test_get_attribute_definitions_by_agent_type(self) -> None:
        """Test get_attribute_definitions_by_agent_type."""
        client = make_client()
        client.http_request_func = _http_gen(_make_response(200, [ATTR_DEF_DICT]))
        result = _exhaust(client.get_attribute_definitions_by_agent_type(AGENT_TYPE))
        assert len(result) == 1

    def test_get_attribute_definitions_by_agent_type_empty(self) -> None:
        """Test returns empty list on 404."""
        client = make_client()
        client.http_request_func = _http_gen(_make_response(404))
        result = _exhaust(client.get_attribute_definitions_by_agent_type(AGENT_TYPE))
        assert result == []

    def test_delete_attribute_definition(self) -> None:
        """Test delete_attribute_definition."""
        client = make_client()
        client.signing_func = _sign_gen()
        client.agent = AGENT_INSTANCE
        client.address = "0xABC"
        client.http_request_func = _http_gen(_make_response(200, ATTR_DEF_DICT))
        result = _exhaust(client.delete_attribute_definition(ATTR_DEF))
        assert isinstance(result, AttributeDefinition)

    def test_delete_attribute_definition_none(self) -> None:
        """Test delete_attribute_definition returns None on 404."""
        client = make_client()
        client.signing_func = _sign_gen()
        client.agent = AGENT_INSTANCE
        client.address = "0xABC"
        client.http_request_func = _http_gen(_make_response(404))
        result = _exhaust(client.delete_attribute_definition(ATTR_DEF))
        assert result is None


# ---------------------------------------------------------------------------
# Tests: CRUD Attribute Instance
# ---------------------------------------------------------------------------


class TestAttributeInstanceCRUD:
    """Tests for attribute instance CRUD methods."""

    def _make_auth_client(self, *responses):
        client = make_client()
        client.signing_func = _sign_gen()
        client.agent = AGENT_INSTANCE
        client.address = "0xABC"
        client.http_request_func = _http_gen(*responses)
        return client

    def test_create_attribute_instance(self) -> None:
        """Test create_attribute_instance."""
        client = self._make_auth_client(_make_response(201, ATTR_INSTANCE_DICT))
        result = _exhaust(client.create_attribute_instance(AGENT_INSTANCE, ATTR_DEF, "hello"))
        assert isinstance(result, AttributeInstance)

    def test_create_attribute_instance_custom_type(self) -> None:
        """Test create_attribute_instance with non-default value_type."""
        captured = {}

        def http_func(**kwargs):
            captured.update(kwargs)
            return _make_response(201, ATTR_INSTANCE_DICT)
            yield  # noqa: E501

        client = make_client()
        client.signing_func = _sign_gen()
        client.agent = AGENT_INSTANCE
        client.address = "0xABC"
        client.http_request_func = http_func
        _exhaust(
            client.create_attribute_instance(AGENT_INSTANCE, ATTR_DEF, 42, value_type="integer")
        )
        body = json.loads(captured["content"])
        assert "integer_value" in body["agent_attr"]

    def test_create_attribute_instance_none(self) -> None:
        """Test create_attribute_instance returns None on 404."""
        client = self._make_auth_client(_make_response(404))
        result = _exhaust(client.create_attribute_instance(AGENT_INSTANCE, ATTR_DEF, "x"))
        assert result is None

    def test_get_attribute_instance(self) -> None:
        """Test get_attribute_instance."""
        client = make_client()
        client.http_request_func = _http_gen(_make_response(200, ATTR_INSTANCE_DICT))
        result = _exhaust(client.get_attribute_instance(AGENT_INSTANCE, ATTR_DEF))
        assert isinstance(result, AttributeInstance)

    def test_get_attribute_instance_none(self) -> None:
        """Test get_attribute_instance returns None on 404."""
        client = make_client()
        client.http_request_func = _http_gen(_make_response(404))
        result = _exhaust(client.get_attribute_instance(AGENT_INSTANCE, ATTR_DEF))
        assert result is None

    def test_update_attribute_instance(self) -> None:
        """Test update_attribute_instance."""
        client = self._make_auth_client(_make_response(200, ATTR_INSTANCE_DICT))
        result = _exhaust(
            client.update_attribute_instance(
                AGENT_INSTANCE, ATTR_DEF, ATTR_INSTANCE, "new_val",
            )
        )
        assert isinstance(result, AttributeInstance)

    def test_update_attribute_instance_none(self) -> None:
        """Test update_attribute_instance returns None on 404."""
        client = self._make_auth_client(_make_response(404))
        result = _exhaust(
            client.update_attribute_instance(
                AGENT_INSTANCE, ATTR_DEF, ATTR_INSTANCE, "x",
            )
        )
        assert result is None

    def test_delete_attribute_instance(self) -> None:
        """Test delete_attribute_instance."""
        client = self._make_auth_client(_make_response(200, ATTR_INSTANCE_DICT))
        result = _exhaust(client.delete_attribute_instance(ATTR_INSTANCE))
        assert isinstance(result, AttributeInstance)

    def test_delete_attribute_instance_none(self) -> None:
        """Test delete_attribute_instance returns None on 404."""
        client = self._make_auth_client(_make_response(404))
        result = _exhaust(client.delete_attribute_instance(ATTR_INSTANCE))
        assert result is None


# ---------------------------------------------------------------------------
# Tests: parse / get_all methods
# ---------------------------------------------------------------------------


class TestAttributeParsing:
    """Tests for attribute parsing and getting all attributes."""

    def test_parse_attribute_instance(self) -> None:
        """Test parse_attribute_instance generator."""
        client = make_client()
        client._attribute_definition_cache[100] = ATTR_DEF
        result = _exhaust(client.parse_attribute_instance(ATTR_INSTANCE))
        assert result["attr_name"] == "my_attr"
        assert result["attr_value"] == "hello"

    def test_parse_attribute_instance_fetches_definition(self) -> None:
        """Test parse_attribute_instance when definition is not cached."""
        client = make_client()
        client.http_request_func = _http_gen(_make_response(200, ATTR_DEF_DICT))
        result = _exhaust(client.parse_attribute_instance(ATTR_INSTANCE))
        assert result["attr_name"] == "my_attr"

    def test_get_all_agent_instance_attributes_raw(self) -> None:
        """Test get_all_agent_instance_attributes_raw."""
        client = make_client()
        client.signing_func = _sign_gen()
        client.agent = AGENT_INSTANCE
        client.address = "0xABC"
        client.http_request_func = _http_gen(_make_response(200, [ATTR_INSTANCE_DICT]))
        result = _exhaust(client.get_all_agent_instance_attributes_raw(AGENT_INSTANCE))
        assert len(result) == 1

    def test_get_all_agent_instance_attributes_parsed(self) -> None:
        """Test get_all_agent_instance_attributes_parsed."""
        client = make_client()
        client.signing_func = _sign_gen()
        client.agent = AGENT_INSTANCE
        client.address = "0xABC"
        client.http_request_func = _http_gen(_make_response(200, [ATTR_INSTANCE_DICT]))
        client._attribute_definition_cache[100] = ATTR_DEF
        result = _exhaust(client.get_all_agent_instance_attributes_parsed(AGENT_INSTANCE))
        assert len(result) == 1
        assert result[0]["attr_name"] == "my_attr"


# ---------------------------------------------------------------------------
# Tests: update_or_create_agent_attribute
# ---------------------------------------------------------------------------


class TestUpdateOrCreateAgentAttribute:
    """Tests for update_or_create_agent_attribute."""

    def test_update_existing(self) -> None:
        """Test updating an existing attribute returns True."""
        client = make_client()
        client.signing_func = _sign_gen()
        client.agent = AGENT_INSTANCE
        client.address = "0xABC"
        client.http_request_func = _http_gen(
            _make_response(200, ATTR_DEF_DICT),          # get_attribute_definition_by_name
            _make_response(200, ATTR_INSTANCE_DICT),     # get_attribute_instance
            _make_response(200, ATTR_INSTANCE_DICT),     # update_attribute_instance
        )
        result = _exhaust(client.update_or_create_agent_attribute("my_attr", "new_val"))
        assert result is True

    def test_update_fails_returns_false(self) -> None:
        """Test returns False when update fails (returns None)."""
        client = make_client()
        client.signing_func = _sign_gen()
        client.agent = AGENT_INSTANCE
        client.address = "0xABC"
        client.http_request_func = _http_gen(
            _make_response(200, ATTR_DEF_DICT),
            _make_response(200, ATTR_INSTANCE_DICT),
            _make_response(404),  # update returns None
        )
        result = _exhaust(client.update_or_create_agent_attribute("my_attr", "val"))
        assert result is False

    def test_create_new(self) -> None:
        """Test creating a new attribute returns True."""
        client = make_client()
        client.signing_func = _sign_gen()
        client.agent = AGENT_INSTANCE
        client.address = "0xABC"
        client.http_request_func = _http_gen(
            _make_response(200, ATTR_DEF_DICT),
            _make_response(404),                          # get_attribute_instance -> None
            _make_response(201, ATTR_INSTANCE_DICT),      # create_attribute_instance
        )
        result = _exhaust(client.update_or_create_agent_attribute("my_attr", "val"))
        assert result is True

    def test_create_fails_returns_false(self) -> None:
        """Test returns False when create fails."""
        client = make_client()
        client.signing_func = _sign_gen()
        client.agent = AGENT_INSTANCE
        client.address = "0xABC"
        client.http_request_func = _http_gen(
            _make_response(200, ATTR_DEF_DICT),
            _make_response(404),  # get_attribute_instance -> None
            _make_response(404),  # create fails
        )
        result = _exhaust(client.update_or_create_agent_attribute("my_attr", "val"))
        assert result is False
