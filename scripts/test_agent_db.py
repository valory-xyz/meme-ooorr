#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2021-2025 Valory AG
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

"""Test the AgentDBClient class."""


import os
from datetime import datetime, timezone
from typing import Any, List, Optional

import dotenv
import requests
from eth_account import Account
from eth_account.messages import encode_defunct
from pydantic import BaseModel


dotenv.load_dotenv(override=True)


# https://axatbhardwaj.notion.site/MirrorDB-Agent-and-Attribute-Data-Flow-1eac8d38bc0b80edae04ff1017d80f58
# https://afmdb.autonolas.tech/docs#/default/read_attribute_definitions_by_type_api_agent_types__type_id__attributes__get


class AgentType(BaseModel):
    """AgentType"""

    type_id: int
    type_name: str
    description: str


class AgentInstance(BaseModel):
    """AgentInstance"""

    agent_id: int
    type_id: int
    agent_name: str
    eth_address: str
    created_at: datetime


class AttributeDefinition(BaseModel):
    """AttributeDefinition"""

    attr_def_id: int
    type_id: int
    attr_name: str
    data_type: str
    is_required: bool
    default_value: Any


class AttributeInstance(BaseModel):
    """AttributeInstance"""

    attribute_id: int
    attr_def_id: int
    agent_id: int
    last_updated: datetime
    string_value: str | None
    integer_value: int | None
    float_value: float | None
    boolean_value: bool | None
    date_value: datetime | None
    json_value: Any | None


class AgentsFunAgentType(BaseModel):
    """AgentsFunAgentType"""

    agent_type: AgentType
    attribute_definitions: List[AttributeDefinition]


class AgentsFunInstance(BaseModel):
    """AgentsFunInstance"""

    twitter_username: str
    interactions: List[Any]


class AgentDBClient:
    """AgentDBClient"""

    def __init__(self, base_url, eth_address, private_key):
        """Constructor"""
        self.base_url = base_url.rstrip("/")
        self.eth_address = eth_address
        self.private_key = private_key
        self.agent = self.get_agent_by_address(self.eth_address)

    def _sign_request(self, endpoint):
        """Generate authentication"""
        timestamp = int(datetime.now(timezone.utc).timestamp())
        message_to_sign = f"timestamp:{timestamp},endpoint:{endpoint}"
        signed_message = Account.sign_message(
            encode_defunct(text=message_to_sign), private_key=self.private_key
        )

        auth_data = {
            "agent_id": self.agent.agent_id,
            "signature": signed_message.signature.hex(),
            "message": message_to_sign,
        }
        return auth_data

    def _request(self, method, endpoint, payload=None, params=None, auth=False):
        """Make the request"""
        url = f"{self.base_url}{endpoint}"
        headers = {"Content-Type": "application/json"}
        if auth:
            payload = payload or {}
            payload["auth"] = self._sign_request(endpoint)
        response = requests.request(
            method, url, headers=headers, json=payload, params=params
        )
        if response.status_code in [200, 201]:
            return response.json()
        if response.status_code == 404:
            return None
        raise Exception(f"Request failed: {response.status_code} - {response.text}")

    # Agent Type Methods
    def get_agent_type(self, type_name) -> Optional[AgentType]:
        """Get agent type by name"""
        endpoint = f"/api/agent-types/name/{type_name}"
        result = self._request("GET", endpoint)
        return AgentType.model_validate(result) if result else None

    def create_agent_type(self, type_name, description) -> Optional[AgentType]:
        """Create agent type"""
        endpoint = "/api/agent-types/"
        payload = {"type_name": type_name, "description": description}
        result = self._request("POST", endpoint, payload)
        return AgentType.model_validate(result) if result else None

    # Agent Registry Methods
    def get_agent_by_address(self, eth_address) -> Optional[AgentInstance]:
        """Get agent by Ethereum address"""
        endpoint = f"/api/agent-registry/address/{eth_address}"
        result = self._request("GET", endpoint)
        return AgentInstance.model_validate(result) if result else None

    def get_agent_by_type(self, type_id):
        """Get agent by type"""
        endpoint = f"/api/agent-types/{type_id}/agents/"
        return self._request("GET", endpoint)

    def create_agent(
        self, agent_name: str, agent_type: AgentType, eth_address: str
    ) -> Optional[AgentInstance]:
        """Create agent"""
        endpoint = "/api/agent-registry/"
        payload = {
            "agent_name": agent_name,
            "type_id": agent_type.type_id,
            "eth_address": eth_address,
        }
        result = self._request("POST", endpoint, payload)
        return AgentInstance.model_validate(result) if result else None

    # Attribute Definition Methods
    def get_attribute_definition(self, attr_name: str) -> Optional[AttributeDefinition]:
        """Get attribute definition by name"""
        endpoint = f"/api/attributes/name/{attr_name}"
        result = self._request("GET", endpoint)
        return AttributeDefinition.model_validate(result) if result else None

    def create_attribute_definition(
        self,
        agent_type: AgentType,
        attr_name: str,
        data_type: str,
        is_required: bool = False,
    ):
        """Create attribute definition"""
        endpoint = f"/api/agent-types/{agent_type.type_id}/attributes/"
        payload = {
            "type_id": agent_type.type_id,
            "attr_name": attr_name,
            "data_type": data_type,
            "is_required": is_required,
        }
        result = self._request("POST", endpoint, {"attr_def": payload}, auth=True)
        return AttributeDefinition.model_validate(result) if result else None

    def get_attributes_by_agent_type(self, agent_type: AgentType):
        """Get attributes by agent type"""
        endpoint = f"/api/agent-types/{agent_type.type_id}/attributes/"
        result = self._request("GET", endpoint)
        return (
            [AttributeDefinition.model_validate(attr) for attr in result]
            if result
            else []
        )

    # Attribute Instance Methods
    def get_attribute_instance(
        self, agent_instance: AgentInstance, attr_def: AttributeDefinition
    ) -> Optional[AttributeInstance]:
        """Get attribute instance by agent ID and attribute definition ID"""
        endpoint = (
            f"/api/agents/{agent_instance.agent_id}/attributes/{attr_def.attr_def_id}/"
        )
        result = self._request("GET", endpoint)
        return AttributeInstance.model_validate(result) if result else None

    def create_attribute_instance(
        self,
        agent_instance: AgentInstance,
        attribute_def: AttributeDefinition,
        value: Any,
        value_type="string",
    ) -> Optional[AttributeInstance]:
        """Create attribute instance"""
        endpoint = f"/api/agents/{agent_instance.agent_id}/attributes/"
        payload = {
            "agent_id": agent_instance.agent_id,
            "attr_def_id": attribute_def.attr_def_id,
            f"{value_type}_value": value,
        }
        result = self._request("POST", endpoint, {"agent_attr": payload}, auth=True)
        return AttributeInstance.model_validate(result) if result else None

    def update_attribute_instance(
        self,
        agent_instance: AgentInstance,
        attribute_def: AttributeDefinition,
        attribute_instance: AttributeInstance,
        value: Any,
        value_type="string",
    ) -> Optional[AttributeInstance]:
        """Update attribute instance"""
        endpoint = f"/api/agent-attributes/{attribute_instance.attribute_id}"
        payload = {f"{value_type}_value": value}
        payload = {
            "agent_id": agent_instance.agent_id,
            "attr_def_id": attribute_def.attr_def_id,
            f"{value_type}_value": value,
        }
        result = self._request("PUT", endpoint, {"agent_attr": payload}, auth=True)
        return AttributeInstance.model_validate(result) if result else None

    def get_all_attributes(self, agent_id):
        """Get all attributes of an agent by agent ID"""
        endpoint = f"/api/agents/{agent_id}/attributes/"
        payload = {
            "agent_id": agent_id,
        }
        return self._request("GET", endpoint, {"agent_attr": payload}, auth=True)


if __name__ == "__main__":
    # Initialize the client
    client = AgentDBClient(
        base_url=os.getenv("MIRROR_DB_BASE_URL"),
        eth_address=os.getenv("AGENT_ADDRESS"),
        private_key=os.getenv("AGENT_PRIVATE_KEY"),
    )

    # Read or create agent type
    memeooorr_type = client.get_agent_type("memeooorr")
    print(f"memeooorr_type = {memeooorr_type}")

    if not memeooorr_type:
        agent_type = client.create_agent_type(
            type_name="memeooorr", description="Description of memeooorr"
        )

    # Read or create agent instance
    memeooorr_instance = client.get_agent_by_address(client.eth_address)
    print(f"agent_instance = {memeooorr_instance}")

    if not memeooorr_instance:
        agent_instance = client.create_agent(
            agent_name="Terminator",
            agent_type=memeooorr_type,
            eth_address=client.eth_address,
        )
        print(f"memeooorr_instance = {memeooorr_instance}")

    # Read or create atttribute definition
    twitter_username_attr_def = client.get_attribute_definition("twitter_username")
    print(f"twitter_username_attr_def = {twitter_username_attr_def}")

    if not twitter_username_attr_def:
        twitter_username_attr_def = client.create_attribute_definition(
            agent_type=memeooorr_type,
            attr_name="twitter_username",
            data_type="string",
            is_required=True,
        )
        print(f"twitter_username_attr_def = {twitter_username_attr_def}")

    # Get agent type attributes
    memeooorr_attrs = client.get_attributes_by_agent_type(memeooorr_type)
    print(f"memeooorr_attrs = {memeooorr_attrs}")

    # Ensure Attribute Instance exists
    twitter_username_attr_instance = client.get_attribute_instance(
        memeooorr_instance, twitter_username_attr_def
    )
    print(f"twitter_username_attr_instance = {twitter_username_attr_instance}")

    if not twitter_username_attr_instance:
        twitter_username_instance = client.create_attribute_instance(
            agent_instance=memeooorr_instance,
            attribute_def=twitter_username_attr_def,
            value="user123",
        )
        print(f"twitter_username_instance = {twitter_username_instance}")
    else:
        client.update_attribute_instance(
            agent_instance=memeooorr_instance,
            attribute_def=twitter_username_attr_def,
            attribute_instance=twitter_username_attr_instance,
            value="new_terminator",
        )
        print(f"Updated twitter_username_instance = {twitter_username_attr_instance}")


# # Tables: agent definition, agent instance, attribute definition, attribute instance

# # Create agent
# # Create table
#     # For each value
