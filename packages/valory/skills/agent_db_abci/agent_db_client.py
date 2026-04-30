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

"""This module contains classes to interact with AgentDB."""

import json
from datetime import datetime, timezone
from logging import Logger
from typing import Any, Callable, Dict, Generator, List, Optional

from aea.skills.base import Model

from packages.valory.skills.agent_db_abci.agent_db_models import (
    AgentInstance,
    AgentType,
    AttributeDefinition,
    AttributeInstance,
)

# Docs at:
# https://axatbhardwaj.notion.site/MirrorDB-Agent-and-Attribute-Data-Flow-1eac8d38bc0b80edae04ff1017d80f58
# https://afmdb.autonolas.tech/docs#/default/read_attribute_definitions_by_type_api_agent_types__type_id__attributes__get


class AgentDBClient(Model):
    """AgentDBClient"""

    def __init__(self, base_url: str, **kwargs: Any) -> None:
        """Constructor"""
        super().__init__(**kwargs)
        self.base_url: str = base_url.rstrip("/")
        self._attribute_definition_cache: Dict[int, AttributeDefinition] = {}
        self.agent: Optional[AgentInstance] = None
        self.agent_type: Optional[AgentType] = None
        self.address: Optional[str] = None
        self.signing_func: Optional[Callable[..., Any]] = None
        self.http_request_func: Optional[Callable[..., Any]] = None
        self.logger: Optional[Logger] = None
        self.agent_type_name: Optional[str] = None
        self.agent_name_template: Optional[str] = None

    def initialize(
        self,
        address: str,
        http_request_func: Callable[..., Any],
        signing_func: Callable[..., Any],
        logger: Logger,
        agent_type_name: Optional[str] = None,
        agent_name_template: Optional[str] = None,
    ) -> None:
        """Inject external functions"""
        self.address = address
        self.http_request_func = http_request_func
        self.signing_func = signing_func
        self.logger = logger
        self.agent_type_name = agent_type_name
        self.agent_name_template = agent_name_template

    def _ensure_agent_instance(self) -> Generator[Any, None, None]:
        """Fetch or create the agent instance if it doesn't exist."""
        if self.agent is not None:
            return

        if self.address is None:
            raise ValueError("Address is not set. Call initialize first.")

        self.agent = yield from self.get_agent_instance_by_address(self.address)
        if self.agent:
            self.agent_type = yield from self.get_agent_type_by_type_id(
                self.agent.type_id
            )
        elif self.agent_type_name and self.agent_name_template:
            assert self.logger is not None
            self.logger.info(
                f"Agent with address {self.address} not found. Registering..."
            )
            agent_name = self.agent_name_template.format(address=self.address)
            agent_type = yield from self.get_agent_type_by_type_name(
                self.agent_type_name
            )
            if agent_type is None:
                raise ValueError(f"Agent type {self.agent_type_name} not found")
            self.agent = yield from self.create_agent_instance(
                agent_name=agent_name,
                agent_type=agent_type,
                eth_address=self.address,
            )
            self.agent_type = agent_type

    def _ensure_agent_type_definition(
        self, description: Optional[str] = "Placeholder agent description"
    ) -> Generator[Any, None, None]:
        """Fetch or create the agent type definition if it doesn't exist."""
        if self.agent_type_name is None:
            raise ValueError("agent_type_name is not set. Call initialize first.")
        self.agent_type = yield from self.get_agent_type_by_type_name(
            self.agent_type_name
        )

        if not self.agent_type:
            self.agent_type = yield from self.create_agent_type(
                self.agent_type_name, description
            )

    def _ensure_agent_type_attribute_definition(
        self, attribute_definitions: List[AttributeDefinition]
    ) -> Generator[Any, None, None]:
        """Fetch or create the agent type attribute definition if it doesn't exist."""
        if self.agent_type is None:
            raise ValueError(
                "agent_type is not set. Call _ensure_agent_type_definition first."
            )
        existing_attribute_definitions = (
            yield from self.get_attribute_definitions_by_agent_type(self.agent_type)
        )
        for type_attribute_definition in attribute_definitions:
            if not any(
                ad.attr_name == type_attribute_definition.attr_name
                for ad in existing_attribute_definitions
            ):
                yield from self.create_attribute_definition(
                    agent_type=self.agent_type,
                    attr_name=type_attribute_definition.attr_name,
                    data_type=type_attribute_definition.data_type,
                    default_value=type_attribute_definition.default_value,
                    is_required=type_attribute_definition.is_required,
                )

    def _sign_request(self, endpoint: str) -> Generator[Any, None, Dict[str, Any]]:
        """Generate authentication"""

        if self.signing_func is None:
            raise ValueError(
                "Signing function not set. Use set_external_funcs to set it."
            )

        yield from self._ensure_agent_instance()

        if self.agent is None:
            raise ValueError(
                f"failed to get agent with address {self.address} or register it"
            )

        timestamp = int(datetime.now(timezone.utc).timestamp())
        message_to_sign = f"timestamp:{timestamp},endpoint:{endpoint}"

        signature_hex = yield from self.signing_func(message_to_sign.encode("utf-8"))

        auth_data = {
            "agent_id": self.agent.agent_id,
            "signature": signature_hex,
            "message": message_to_sign,
        }
        return auth_data

    def _request(
        self,
        method: str,
        endpoint: str,
        payload: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        auth: bool = False,
        nested_auth: bool = True,
    ) -> Generator[Any, None, Any]:
        """Make the request"""

        if self.http_request_func is None:
            raise ValueError(
                "HTTP request function not set. Use set_external_funcs to set it."
            )

        url = f"{self.base_url}{endpoint}"
        headers = {"Content-Type": "application/json"}
        if auth:
            payload = payload or {}
            auth_data = yield from self._sign_request(endpoint)
            if nested_auth:
                payload["auth"] = auth_data
            else:
                payload = payload | auth_data

        assert self.logger is not None
        self.logger.info(
            f"Making {method} request to {url} with payload: {payload} and params: {params}"
        )

        content = json.dumps(payload).encode() if payload else None

        response = yield from self.http_request_func(
            method=method, url=url, content=content, headers=headers, parameters=params
        )

        self.logger.info(f"Response status: {response.status_code}")

        if response.status_code in [200, 201]:
            return json.loads(response.body)

        if response.status_code == 404:
            return None

        raise RuntimeError(f"Request failed: {response.status_code} - {response.text}")

    # Agent Type Methods

    def create_agent_type(
        self, type_name: str, description: Optional[str]
    ) -> Generator[Any, None, Optional[AgentType]]:
        """Create agent type"""
        endpoint = "/api/agent-types/"
        payload = {"type_name": type_name, "description": description}
        result = yield from self._request("POST", endpoint, payload)
        return AgentType.model_validate(result) if result else None

    def get_agent_type_by_type_id(
        self, type_id: int
    ) -> Generator[Any, None, Optional[AgentType]]:
        """Get agent by type"""
        endpoint = f"/api/agent-types/{type_id}/"
        result = yield from self._request("GET", endpoint)
        return AgentType.model_validate(result) if result else None

    def get_agent_type_by_type_name(
        self, type_name: str
    ) -> Generator[Any, None, Optional[AgentType]]:
        """Get agent by type"""
        endpoint = f"/api/agent-types/name/{type_name}/"
        result = yield from self._request("GET", endpoint)
        return AgentType.model_validate(result) if result else None

    def delete_agent_type(
        self, agent_type: AgentType
    ) -> Generator[Any, None, Optional[AgentType]]:
        """Delete agent type"""
        endpoint = f"/api/agent-types/{agent_type.type_id}/"
        result = yield from self._request(
            "DELETE", endpoint, auth=True, nested_auth=True
        )
        return AgentType.model_validate(result) if result else None

    # Agent Instance Methods

    def create_agent_instance(
        self, agent_name: str, agent_type: AgentType, eth_address: str
    ) -> Generator[Any, None, Optional[AgentInstance]]:
        """Create agent instance"""
        endpoint = "/api/agent-registry/"
        payload = {
            "agent_name": agent_name,
            "type_id": agent_type.type_id,
            "eth_address": eth_address,
        }
        result = yield from self._request("POST", endpoint, payload)
        return AgentInstance.model_validate(result) if result else None

    def get_agent_instance_by_address(
        self, eth_address: str
    ) -> Generator[Any, None, Optional[AgentInstance]]:
        """Get agent by Ethereum address"""
        endpoint = f"/api/agent-registry/address/{eth_address}"
        result = yield from self._request("GET", endpoint)
        return AgentInstance.model_validate(result) if result else None

    def get_agent_instances_by_type_id(
        self, type_id: int
    ) -> Generator[Any, None, List[AgentInstance]]:
        """Get agent instances by type"""
        endpoint = f"/api/agent-types/{type_id}/agents/"
        params = {
            "skip": 0,
            "limit": 100,
        }
        result = yield from self._request(
            method="GET", endpoint=endpoint, params=params
        )
        return (
            [AgentInstance.model_validate(agent) for agent in result] if result else []
        )

    def delete_agent_instance(
        self, agent_instance: AgentInstance
    ) -> Generator[Any, None, Optional[AgentInstance]]:
        """Delete agent instance"""
        endpoint = f"/api/agent-registry/{agent_instance.agent_id}/"
        result = yield from self._request(
            "DELETE", endpoint, auth=True, nested_auth=False
        )
        return AgentInstance.model_validate(result) if result else None

    # Attribute Definition Methods

    def create_attribute_definition(
        self,
        agent_type: AgentType,
        attr_name: str,
        data_type: str,
        default_value: str,
        is_required: bool = False,
    ) -> Generator[Any, None, Optional[AttributeDefinition]]:
        """Create attribute definition"""
        endpoint = f"/api/agent-types/{agent_type.type_id}/attributes/"
        payload = {
            "type_id": agent_type.type_id,
            "attr_name": attr_name,
            "data_type": data_type,
            "default_value": default_value,
            "is_required": is_required,
        }
        result = yield from self._request(
            "POST", endpoint, {"attr_def": payload}, auth=True
        )
        return AttributeDefinition.model_validate(result) if result else None

    def get_attribute_definition_by_name(
        self, attr_name: str
    ) -> Generator[Any, None, Optional[AttributeDefinition]]:
        """Get attribute definition by name"""
        endpoint = f"/api/attributes/name/{attr_name}"
        result = yield from self._request("GET", endpoint)
        return AttributeDefinition.model_validate(result) if result else None

    def get_attribute_definition_by_id(
        self, attr_id: int
    ) -> Generator[Any, None, Optional[AttributeDefinition]]:
        """Get attribute definition by id"""
        if attr_id in self._attribute_definition_cache:
            return self._attribute_definition_cache[attr_id]
        endpoint = f"/api/attributes/{attr_id}"
        result = yield from self._request("GET", endpoint)
        if result:
            definition = AttributeDefinition.model_validate(result)
            self._attribute_definition_cache[attr_id] = definition
            return definition
        return None

    def get_attribute_definitions_by_agent_type(
        self, agent_type: AgentType
    ) -> Generator[Any, None, List[AttributeDefinition]]:
        """Get attributes by agent type"""
        endpoint = f"/api/agent-types/{agent_type.type_id}/attributes/"
        result = yield from self._request("GET", endpoint)
        return (
            [AttributeDefinition.model_validate(attr) for attr in result]
            if result
            else []
        )

    def delete_attribute_definition(
        self, attr_def: AttributeDefinition
    ) -> Generator[Any, None, Optional[AttributeDefinition]]:
        """Delete attribute definition"""
        endpoint = f"/api/attributes/{attr_def.attr_def_id}/"
        result = yield from self._request(
            "DELETE", endpoint, auth=True, nested_auth=True
        )
        return AttributeDefinition.model_validate(result) if result else None

    # Attribute Instance Methods

    def create_attribute_instance(
        self,
        agent_instance: AgentInstance,
        attribute_def: AttributeDefinition,
        value: Any,
        value_type: str = "string",
    ) -> Generator[Any, None, Optional[AttributeInstance]]:
        """Create attribute instance"""
        endpoint = f"/api/agents/{agent_instance.agent_id}/attributes/"
        payload = {
            "agent_id": agent_instance.agent_id,
            "attr_def_id": attribute_def.attr_def_id,
            f"{value_type}_value": value,
        }
        result = yield from self._request(
            "POST", endpoint, {"agent_attr": payload}, auth=True
        )
        return AttributeInstance.model_validate(result) if result else None

    def get_attribute_instance(
        self,
        agent_instance: AgentInstance,
        attr_def: AttributeDefinition,
    ) -> Generator[Any, None, Optional[AttributeInstance]]:
        """Get attribute instance by agent ID and attribute definition ID"""
        endpoint = (
            f"/api/agents/{agent_instance.agent_id}/attributes/{attr_def.attr_def_id}/"
        )
        result = yield from self._request("GET", endpoint)
        return AttributeInstance.model_validate(result) if result else None

    def update_attribute_instance(
        self,
        agent_instance: AgentInstance,
        attribute_def: AttributeDefinition,
        attribute_instance: AttributeInstance,
        value: Any,
        value_type: str = "string",
    ) -> Generator[Any, None, Optional[AttributeInstance]]:
        """Update attribute instance"""
        endpoint = f"/api/agent-attributes/{attribute_instance.attribute_id}"
        payload = {
            "agent_id": agent_instance.agent_id,
            "attr_def_id": attribute_def.attr_def_id,
            f"{value_type}_value": value,
        }
        result = yield from self._request(
            "PUT", endpoint, {"agent_attr": payload}, auth=True
        )
        return AttributeInstance.model_validate(result) if result else None

    def delete_attribute_instance(
        self, attribute_instance: AttributeInstance
    ) -> Generator[Any, None, Optional[AttributeInstance]]:
        """Delete attribute instance"""
        endpoint = f"/api/agent-attributes/{attribute_instance.attribute_id}"
        result = yield from self._request(
            "DELETE", endpoint, auth=True, nested_auth=True
        )
        return AttributeInstance.model_validate(result) if result else None

    # Get all attributes of an agent instance

    def get_all_agent_instance_attributes_raw(
        self, agent_instance: AgentInstance
    ) -> Generator[Any, None, Any]:
        """Get all attributes of an agent by agent ID"""
        endpoint = f"/api/agents/{agent_instance.agent_id}/attributes/"
        payload = {
            "agent_id": agent_instance.agent_id,
        }
        result = yield from self._request(
            "GET", endpoint, {"agent_attr": payload}, auth=True
        )
        return result

    def parse_attribute_instance(
        self, attribute_instance: AttributeInstance
    ) -> Generator[Any, None, Dict[str, Any]]:
        """Parse attribute instance"""
        attribute_definition = yield from self.get_attribute_definition_by_id(
            attribute_instance.attr_def_id
        )
        if attribute_definition is None:
            raise ValueError(
                f"Attribute definition for id {attribute_instance.attr_def_id} not found"
            )
        data_type = attribute_definition.data_type
        attr_value = getattr(attribute_instance, f"{data_type}_value", None)

        attr_value = self.cast_attribute_value(attr_value, attribute_definition)

        parsed_attribute_instance = {
            "attr_name": attribute_definition.attr_name,
            "attr_value": attr_value,
        }
        return parsed_attribute_instance

    def cast_attribute_value(
        self, attr_value: Any, attribute_definition: AttributeDefinition
    ) -> Any:
        """Cast an attribute value to its defined data type."""

        # fetch data type from attribute definition

        data_type = attribute_definition.data_type

        if data_type == "date":
            attr_value = datetime.fromisoformat(
                attr_value.replace("Z", "+00:00")
            ).astimezone(timezone.utc)
        elif data_type == "json":
            pass
        elif data_type == "string":
            attr_value = str(attr_value)
        elif data_type == "integer":
            attr_value = int(attr_value)
        elif data_type == "float":
            attr_value = float(attr_value)
        elif data_type == "boolean":
            attr_value = bool(attr_value)

        return attr_value

    def get_all_agent_instance_attributes_parsed(
        self, agent_instance: AgentInstance
    ) -> Generator[Any, None, List[Dict[str, Any]]]:
        """Get all attributes of an agent by agent ID"""
        attribute_instances = yield from self.get_all_agent_instance_attributes_raw(
            agent_instance
        )
        parsed_attributes = []
        for attr in attribute_instances:
            result = yield from self.parse_attribute_instance(AttributeInstance(**attr))
            parsed_attributes.append(result)
        return parsed_attributes

    def update_or_create_agent_attribute(
        self, attr_name: str, attr_value: Any
    ) -> Generator[Any, None, bool]:
        """Helper to update or create a single agent attribute."""
        attr_def = yield from self.get_attribute_definition_by_name(attr_name)
        if attr_def is None:
            raise ValueError(f"Attribute definition '{attr_name}' not found")
        if self.agent is None:
            raise ValueError("Agent instance not loaded")
        attr_instance = yield from self.get_attribute_instance(self.agent, attr_def)

        # casting the attr_value to the correct type from attribute definition data type
        attr_value = self.cast_attribute_value(attr_value, attr_def)
        assert self.logger is not None
        if attr_instance:

            self.logger.info(f"Updating attribute {attr_name} with value {attr_value}")

            updated = yield from self.update_attribute_instance(
                agent_instance=self.agent,
                attribute_def=attr_def,
                attribute_instance=attr_instance,
                value=attr_value,
                value_type=attr_def.data_type,
            )
            if not updated:
                return False
        else:
            self.logger.info(f"Creating attribute {attr_name} with value {attr_value}")

            created = yield from self.create_attribute_instance(
                agent_instance=self.agent,
                attribute_def=attr_def,
                value=attr_value,
                value_type=attr_def.data_type,
            )
            if not created:
                return False

        return True
