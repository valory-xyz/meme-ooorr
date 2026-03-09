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

"""Tests for agent_db_models.py."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from packages.valory.skills.agent_db_abci.agent_db_models import (
    AgentInstance,
    AgentsFunAgentType,
    AgentType,
    AttributeDefinition,
    AttributeInstance,
)


TIMESTAMP = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


class TestAgentType:
    """Tests for AgentType."""

    def test_construction(self) -> None:
        """Test basic construction."""
        at = AgentType(type_id=1, type_name="memeooorr", description="Test agent")
        assert at.type_id == 1
        assert at.type_name == "memeooorr"
        assert at.description == "Test agent"

    def test_missing_field_raises(self) -> None:
        """Test that missing required fields raise ValidationError."""
        with pytest.raises(ValidationError):
            AgentType(type_id=1, type_name="test")  # type: ignore[call-arg]

    def test_serialization(self) -> None:
        """Test model_dump roundtrip."""
        at = AgentType(type_id=1, type_name="test", description="desc")
        data = at.model_dump()
        restored = AgentType.model_validate(data)
        assert restored == at


class TestAgentInstance:
    """Tests for AgentInstance."""

    def test_construction(self) -> None:
        """Test basic construction."""
        ai = AgentInstance(
            agent_id=1,
            type_id=2,
            agent_name="agent-1",
            eth_address="0xabc",
            created_at=TIMESTAMP,
        )
        assert ai.agent_id == 1
        assert ai.type_id == 2
        assert ai.agent_name == "agent-1"
        assert ai.eth_address == "0xabc"
        assert ai.created_at == TIMESTAMP

    def test_missing_field_raises(self) -> None:
        """Test that missing required fields raise ValidationError."""
        with pytest.raises(ValidationError):
            AgentInstance(agent_id=1, type_id=2)  # type: ignore[call-arg]

    def test_serialization(self) -> None:
        """Test model_dump roundtrip."""
        ai = AgentInstance(
            agent_id=1,
            type_id=2,
            agent_name="agent",
            eth_address="0x0",
            created_at=TIMESTAMP,
        )
        data = ai.model_dump()
        restored = AgentInstance.model_validate(data)
        assert restored == ai


class TestAttributeDefinition:
    """Tests for AttributeDefinition."""

    def test_construction(self) -> None:
        """Test basic construction."""
        ad = AttributeDefinition(
            attr_def_id=1,
            type_id=2,
            attr_name="twitter_username",
            data_type="string",
            is_required=True,
            default_value="",
        )
        assert ad.attr_def_id == 1
        assert ad.type_id == 2
        assert ad.attr_name == "twitter_username"
        assert ad.data_type == "string"
        assert ad.is_required is True
        assert ad.default_value == ""

    def test_construction_with_json_type(self) -> None:
        """Test construction with json data type."""
        ad = AttributeDefinition(
            attr_def_id=3,
            type_id=1,
            attr_name="twitter_interactions",
            data_type="json",
            is_required=False,
            default_value="{}",
        )
        assert ad.data_type == "json"
        assert ad.default_value == "{}"

    def test_serialization(self) -> None:
        """Test model_dump roundtrip."""
        ad = AttributeDefinition(
            attr_def_id=1,
            type_id=1,
            attr_name="test",
            data_type="string",
            is_required=False,
            default_value="default",
        )
        data = ad.model_dump()
        restored = AttributeDefinition.model_validate(data)
        assert restored == ad


class TestAttributeInstance:
    """Tests for AttributeInstance."""

    def test_construction_string_value(self) -> None:
        """Test construction with string value."""
        ai = AttributeInstance(
            attribute_id=1,
            attr_def_id=2,
            agent_id=3,
            last_updated=TIMESTAMP,
            string_value="hello",
            integer_value=None,
            float_value=None,
            boolean_value=None,
            date_value=None,
            json_value=None,
        )
        assert ai.attribute_id == 1
        assert ai.string_value == "hello"
        assert ai.integer_value is None

    def test_construction_integer_value(self) -> None:
        """Test construction with integer value."""
        ai = AttributeInstance(
            attribute_id=1,
            attr_def_id=2,
            agent_id=3,
            last_updated=TIMESTAMP,
            string_value=None,
            integer_value=42,
            float_value=None,
            boolean_value=None,
            date_value=None,
            json_value=None,
        )
        assert ai.integer_value == 42

    def test_construction_json_value(self) -> None:
        """Test construction with json value."""
        json_data = {"key": "value", "nested": [1, 2, 3]}
        ai = AttributeInstance(
            attribute_id=1,
            attr_def_id=2,
            agent_id=3,
            last_updated=TIMESTAMP,
            string_value=None,
            integer_value=None,
            float_value=None,
            boolean_value=None,
            date_value=None,
            json_value=json_data,
        )
        assert ai.json_value == json_data

    def test_construction_boolean_value(self) -> None:
        """Test construction with boolean value."""
        ai = AttributeInstance(
            attribute_id=1,
            attr_def_id=2,
            agent_id=3,
            last_updated=TIMESTAMP,
            string_value=None,
            integer_value=None,
            float_value=None,
            boolean_value=True,
            date_value=None,
            json_value=None,
        )
        assert ai.boolean_value is True

    def test_construction_float_value(self) -> None:
        """Test construction with float value."""
        ai = AttributeInstance(
            attribute_id=1,
            attr_def_id=2,
            agent_id=3,
            last_updated=TIMESTAMP,
            string_value=None,
            integer_value=None,
            float_value=3.14,
            boolean_value=None,
            date_value=None,
            json_value=None,
        )
        assert ai.float_value == pytest.approx(3.14)

    def test_construction_date_value(self) -> None:
        """Test construction with date value."""
        ai = AttributeInstance(
            attribute_id=1,
            attr_def_id=2,
            agent_id=3,
            last_updated=TIMESTAMP,
            string_value=None,
            integer_value=None,
            float_value=None,
            boolean_value=None,
            date_value=TIMESTAMP,
            json_value=None,
        )
        assert ai.date_value == TIMESTAMP

    def test_serialization(self) -> None:
        """Test model_dump roundtrip."""
        ai = AttributeInstance(
            attribute_id=1,
            attr_def_id=2,
            agent_id=3,
            last_updated=TIMESTAMP,
            string_value="test",
            integer_value=None,
            float_value=None,
            boolean_value=None,
            date_value=None,
            json_value=None,
        )
        data = ai.model_dump()
        restored = AttributeInstance.model_validate(data)
        assert restored == ai


class TestAgentsFunAgentType:
    """Tests for AgentsFunAgentType."""

    def test_construction(self) -> None:
        """Test basic construction."""
        at = AgentType(type_id=1, type_name="memeooorr", description="Test")
        ad = AttributeDefinition(
            attr_def_id=1,
            type_id=1,
            attr_name="twitter_username",
            data_type="string",
            is_required=True,
            default_value="",
        )
        afat = AgentsFunAgentType(agent_type=at, attribute_definitions=[ad])
        assert afat.agent_type == at
        assert len(afat.attribute_definitions) == 1
        assert afat.attribute_definitions[0].attr_name == "twitter_username"

    def test_empty_attribute_definitions(self) -> None:
        """Test with empty attribute definitions list."""
        at = AgentType(type_id=1, type_name="test", description="Test")
        afat = AgentsFunAgentType(agent_type=at, attribute_definitions=[])
        assert afat.attribute_definitions == []

    def test_serialization(self) -> None:
        """Test model_dump roundtrip."""
        at = AgentType(type_id=1, type_name="test", description="Test")
        afat = AgentsFunAgentType(agent_type=at, attribute_definitions=[])
        data = afat.model_dump()
        restored = AgentsFunAgentType.model_validate(data)
        assert restored == afat
