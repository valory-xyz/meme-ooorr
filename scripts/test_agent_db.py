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
from typing import Any, Dict, List, Optional

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


class AgentDBClient:
    """AgentDBClient"""

    def __init__(self, base_url, eth_address, private_key):
        """Constructor"""
        self.base_url = base_url.rstrip("/")
        self.eth_address = eth_address
        self.private_key = private_key
        self.agent = self.get_agent_by_address(self.eth_address)
        self.agent_type = (
            self.get_agent_type_by_type_id(self.agent.type_id) if self.agent else None
        )

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

    def get_agent_by_address(self, eth_address) -> Optional[AgentInstance]:
        """Get agent by Ethereum address"""
        endpoint = f"/api/agent-registry/address/{eth_address}"
        result = self._request("GET", endpoint)
        return AgentInstance.model_validate(result) if result else None

    def get_agent_type_by_type_id(self, type_id) -> Optional[AgentType]:
        """Get agent by type"""
        endpoint = f"/api/agent-types/{type_id}/"
        result = self._request("GET", endpoint)
        return AgentType.model_validate(result) if result else None

    def get_agent_instances_by_type_id(self, type_id) -> List[AgentInstance]:
        """Get agent instances by type"""
        endpoint = f"/api/agent-types/{type_id}/agents/"
        params = {
            "skip": 0,
            "limit": 100,
        }
        result = self._request(method="GET", endpoint=endpoint, params=params)
        return (
            [AgentInstance.model_validate(agent) for agent in result] if result else []
        )

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
    def get_attribute_definition_by_name(
        self, attr_name: str
    ) -> Optional[AttributeDefinition]:
        """Get attribute definition by name"""
        endpoint = f"/api/attributes/name/{attr_name}"
        result = self._request("GET", endpoint)
        return AttributeDefinition.model_validate(result) if result else None

    def get_attribute_definition_by_id(
        self, attr_id: int
    ) -> Optional[AttributeDefinition]:
        """Get attribute definition by id"""
        endpoint = f"/api/attributes/{attr_id}"
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

    def get_all_instance_attributes_raw(self, agent_instance: AgentInstance):
        """Get all attributes of an agent by agent ID"""
        endpoint = f"/api/agents/{agent_instance.agent_id}/attributes/"
        payload = {
            "agent_id": agent_instance.agent_id,
        }
        return self._request("GET", endpoint, {"agent_attr": payload}, auth=True)

    def parse_attribute_instance(self, attribute_instance: AttributeInstance):
        """Parse attribute instance"""
        attribute_definition = self.get_attribute_definition_by_id(
            attribute_instance.attr_def_id
        )
        data_type = attribute_definition.data_type
        attr_value = getattr(attribute_instance, f"{data_type}_value", None)

        if data_type == "date":
            attr_value = datetime.fromisoformat(attr_value).astimezone(timezone.utc)
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

        parsed_attribute_instance = {
            "attr_name": attribute_definition.attr_name,
            "attr_value": attr_value,
        }
        return parsed_attribute_instance

    def get_all_instance_attributes_parsed(self, agent_instance: AgentInstance):
        """Get all attributes of an agent by agent ID"""
        attribute_instances = self.get_all_instance_attributes_raw(agent_instance)
        parsed_attributes = [
            self.parse_attribute_instance(AttributeInstance(**attr))
            for attr in attribute_instances
        ]
        return parsed_attributes


class TwitterAction(BaseModel):
    """TwitterAction"""

    action: str
    timestamp: datetime


class TwitterPost(TwitterAction):
    """TwitterPost"""

    tweet_id: str
    text: str
    reply_to_tweet_id: Optional[str] = None

    @classmethod
    def from_nested_json(cls, data: Dict[str, Any]) -> "TwitterPost":
        """Convert nested JSON to TwitterPost"""
        return cls(
            action=data["action"],
            timestamp=datetime.fromisoformat(
                data["timestamp"].replace("Z", "+00:00")
            ).astimezone(timezone.utc),
            tweet_id=data["details"]["tweet_id"],
            text=data["details"]["text"],
            reply_to_tweet_id=data["details"].get("reply_to_tweet_id", None),
        )


class TwitterRewtweet(TwitterAction):
    """TwitterRewtweet"""

    tweet_id: str

    @classmethod
    def from_nested_json(cls, data: Dict[str, Any]) -> "TwitterPost":
        """Convert nested JSON to TwitterPost"""
        return cls(
            action=data["action"],
            timestamp=datetime.fromisoformat(
                data["timestamp"].replace("Z", "+00:00")
            ).astimezone(timezone.utc),
            tweet_id=data["details"]["tweet_id"],
        )


class TwitterFollow(TwitterAction):
    """TwitterFollow"""

    username: str

    @classmethod
    def from_nested_json(cls, data: Dict[str, Any]) -> "TwitterPost":
        """Convert nested JSON to TwitterPost"""
        return cls(
            action=data["action"],
            timestamp=datetime.fromisoformat(
                data["timestamp"].replace("Z", "+00:00")
            ).astimezone(timezone.utc),
            username=data["details"]["username"],
        )


class TwitterLike(TwitterAction):
    """TwitterLike"""

    tweet_id: str

    @classmethod
    def from_nested_json(cls, data: Dict[str, Any]) -> "TwitterPost":
        """Convert nested JSON to TwitterPost"""
        return cls(
            action=data["action"],
            timestamp=datetime.fromisoformat(
                data["timestamp"].replace("Z", "+00:00")
            ).astimezone(timezone.utc),
            tweet_id=data["details"]["tweet_id"],
        )


class AgentsFunAgent:
    """AgentsFunAgent"""

    action_to_class: Dict[str, Any] = {
        "post": TwitterPost,
        "retweet": TwitterRewtweet,
        "follow": TwitterFollow,
        "like": TwitterLike,
    }

    def __init__(self, client: AgentDBClient, agent_instance: AgentInstance):
        """Constructor"""
        self.client = client
        self.agent_instance = agent_instance
        self.twitter_username: str = None
        self.posts: List[TwitterPost] = []
        self.likes: List[TwitterLike] = []
        self.retweets: List[TwitterRewtweet] = []
        self.follows: List[TwitterFollow] = []
        self.loaded = False

    def load(self):
        """Load agent data"""
        attributes = self.client.get_all_instance_attributes_parsed(self.agent_instance)

        interactions = []
        for attr in attributes:
            if attr["attr_name"] == "twitter_username":
                self.twitter_username = attr["attr_value"]
            elif attr["attr_name"] == "twitter_interactions":
                action_class = self.action_to_class.get(
                    attr["attr_value"]["action"], None
                )
                if not action_class:
                    raise ValueError(
                        f"Unknown Twitter action: {attr['attr_value']['action']}"
                    )
                interactions.append(
                    action_class.from_nested_json(data=attr["attr_value"])
                )

        # Separate the interactions into different lists and sort by timestamp
        interactions.sort(key=lambda x: x.timestamp)
        self.posts = [
            interaction
            for interaction in interactions
            if isinstance(interaction, TwitterPost)
        ]
        self.retweets = [
            interaction
            for interaction in interactions
            if isinstance(interaction, TwitterRewtweet)
        ]
        self.likes = [
            interaction
            for interaction in interactions
            if isinstance(interaction, TwitterLike)
        ]
        self.follows = [
            interaction
            for interaction in interactions
            if isinstance(interaction, TwitterFollow)
        ]
        self.loaded = True


class AgentsFunDatabase:
    """AgentsFunDatabase"""

    def __init__(self, client: AgentDBClient):
        """Constructor"""
        self.client = client
        self.agent_type = client.get_agent_type("memeooorr")
        self.agents = []

    def load(self):
        """Load data"""
        agent_instances = self.client.get_agent_instances_by_type_id(
            self.agent_type.type_id
        )
        print(f"Found {len(agent_instances)} agent instances")
        for agent_instance in agent_instances:
            print(f"Loading agent instance {agent_instance.agent_id}")
            self.agents.append(AgentsFunAgent(self.client, agent_instance))
            self.agents[-1].load()


def example(client: AgentDBClient):
    """Example usage of the AgentDBClient class."""

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
    twitter_username_attr_def = client.get_attribute_definition_by_name(
        "twitter_username"
    )
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

    # Create or update attribute instance
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

    # Get all attributes of an agent
    all_attributes = client.get_all_instance_attributes_parsed(memeooorr_instance)
    print(f"all_attributes = {all_attributes}")


if __name__ == "__main__":
    # Initialize the client
    client = AgentDBClient(
        base_url=os.getenv("MIRROR_DB_BASE_URL"),
        eth_address=os.getenv("AGENT_ADDRESS"),
        private_key=os.getenv("AGENT_PRIVATE_KEY"),
    )

    agents_fun = AgentsFunDatabase(client=client)
    agents_fun.load()

    # example(client)
