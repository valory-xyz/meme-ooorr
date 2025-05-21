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

import dotenv

from packages.agents_fun_db.agent_db_client import AgentDBClient
from packages.agents_fun_db.agents_fun_db import AgentsFunDatabase, MEMEOOORR
from packages.agents_fun_db.twitter_models import (
    TwitterFollow,
    TwitterLike,
    TwitterPost,
    TwitterRewtweet,
)


dotenv.load_dotenv(override=True)


def basic_example(client: AgentDBClient):
    """Example usage of the AgentDBClient class."""

    # Read or create agent type
    memeooorr_type = client.get_agent_type_by_type_name(MEMEOOORR)
    print(f"memeooorr_type = {memeooorr_type}")

    if not memeooorr_type:
        client.create_agent_type(
            type_name=MEMEOOORR, description="Description of memeooorr"
        )

    # Read or create agent instance
    memeooorr_instance = client.get_agent_instance_by_address(client.eth_address)
    print(f"agent_instance = {memeooorr_instance}")

    if not memeooorr_instance:
        client.create_agent_instance(
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
            default_value="",
            is_required=True,
        )
        print(f"twitter_username_attr_def = {twitter_username_attr_def}")

    # Get agent type attributes
    memeooorr_attrs = client.get_attribute_definitions_by_agent_type(memeooorr_type)
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
    all_attributes = client.get_all_agent_instance_attributes_parsed(memeooorr_instance)
    print(f"all_attributes = {all_attributes}")


def init_memeooorr_db(client: AgentDBClient):
    """Initialize the memeooorr database"""

    # Read or create agent type
    memeooorr_type = client.get_agent_type_by_type_name(MEMEOOORR)

    if not memeooorr_type:
        print(f"Creating agent type {MEMEOOORR}")
        memeooorr_type = client.create_agent_type(
            type_name=MEMEOOORR, description="Description of memeooorr"
        )
    print(f"memeooorr_type = {memeooorr_type}")

    # Read or create agent instance (needed to sign)
    memeooorr_instance = client.get_agent_instance_by_address(client.eth_address)

    if not memeooorr_instance:
        print(f"Creating agent instance {client.eth_address}")
        memeooorr_instance = client.create_agent_instance(
            agent_name="Terminator",
            agent_type=memeooorr_type,
            eth_address=client.eth_address,
        )
    print(f"memeooorr_instance = {memeooorr_instance}")

    # Read or create attribute definitions
    memeooorr_attrs = client.get_attribute_definitions_by_agent_type(memeooorr_type)

    if not memeooorr_attrs:
        print("Creating agent type attributes")
        twitter_username_attr_def = client.create_attribute_definition(
            agent_type=memeooorr_type,
            attr_name="twitter_username",
            data_type="string",
            default_value="",
            is_required=True,
        )
        twitter_user_id_attr_def = client.create_attribute_definition(
            agent_type=memeooorr_type,
            attr_name="twitter_user_id",
            data_type="string",
            default_value="",
            is_required=True,
        )
        client.create_attribute_definition(
            agent_type=memeooorr_type,
            attr_name="twitter_interactions",
            data_type="json",
            default_value="{}",
            is_required=False,
        )
        memeooorr_attrs = client.get_attribute_definitions_by_agent_type(memeooorr_type)
    else:
        (
            twitter_username_attr_def,
            twitter_user_id_attr_def,
            _,
        ) = memeooorr_attrs

    print(f"memeooorr_attrs = {memeooorr_attrs}")

    # Create attribute instances
    twitter_username_attr_instance = client.get_attribute_instance(
        memeooorr_instance, twitter_username_attr_def
    )
    if not twitter_username_attr_instance:
        print("Creating twitter_username attribute instance")
        twitter_username_attr_instance = client.create_attribute_instance(
            agent_instance=memeooorr_instance,
            attribute_def=twitter_username_attr_def,
            value="0xTerminator",
        )
    print(f"twitter_username_attr_instance = {twitter_username_attr_instance}")

    twitter_user_id_attr_instance = client.get_attribute_instance(
        memeooorr_instance, twitter_user_id_attr_def
    )
    if not twitter_user_id_attr_instance:
        print("Creating twitter_user_id attribute instance")
        twitter_user_id_attr_instance = client.create_attribute_instance(
            agent_instance=memeooorr_instance,
            attribute_def=twitter_user_id_attr_def,
            value="1234567890",
        )
    print(f"twitter_user_id_attr_instance = {twitter_user_id_attr_instance}")

    # Load the database
    agents_fun_db = AgentsFunDatabase(client=client)
    agents_fun_db.load()

    # Add a post
    post = TwitterPost(
        timestamp=datetime.now(timezone.utc),
        tweet_id="1234567890",
        text="Hello, world!",
    )
    agents_fun_db.agents[0].add_interaction(post)

    # Add a retweet
    retweet = TwitterRewtweet(
        timestamp=datetime.now(timezone.utc),
        tweet_id="0987654321",
    )
    agents_fun_db.agents[0].add_interaction(retweet)

    # Add a like
    like = TwitterLike(
        timestamp=datetime.now(timezone.utc),
        tweet_id="1234567890",
    )
    agents_fun_db.agents[0].add_interaction(like)

    # Add a follow
    follow = TwitterFollow(
        timestamp=datetime.now(timezone.utc),
        username="another_user",
    )
    agents_fun_db.agents[0].add_interaction(follow)


def reset_agents_fun_db(client: AgentDBClient):
    """Reset the database"""

    agents_fun_db = AgentsFunDatabase(client=client)
    agents_fun_db.load()

    for agent in agents_fun_db.agents:
        # Delete attributes instances
        memeooorr_attrs = client.get_all_agent_instance_attributes_parsed(
            agent.agent_instance
        )
        for attr in memeooorr_attrs:
            print(f"Deleting agent attribute {attr.attr_def_id}")
            client.delete_attribute_instance(attr)

        # Delete agent instance
        print(f"Deleting agent instance {agent.agent_instance.agent_id}")
        client.delete_agent_instance(agent.agent_instance)

    # Delete attribute definitions
    memeooorr_attr_defs = client.get_attribute_definitions_by_agent_type(
        agents_fun_db.agent_type
    )
    for attr_def in memeooorr_attr_defs:
        print(f"Deleting attribute definition {attr_def.attr_def_id}")
        client.delete_attribute_definition(attr_def)

    # Delete agent type
    print(f"Deleting agent type {agents_fun_db.agent_type.type_id}")
    client.delete_agent_type(agents_fun_db.agent_type)


def memeooorr_example(client: AgentDBClient):
    """Example usage of the AgentDBClient class."""
    agents_fun_db = AgentsFunDatabase(client=client)
    agents_fun_db.load()

    print(agents_fun_db)
    for agent in agents_fun_db.agents:
        if agent.likes or agent.retweets or agent.posts or agent.follows:
            print(agent)


if __name__ == "__main__":
    # Initialize the client
    client = AgentDBClient(
        base_url=os.getenv("MIRROR_DB_BASE_URL"),
        eth_address=os.getenv("AGENT_ADDRESS"),
        private_key=os.getenv("AGENT_PRIVATE_KEY"),
    )

    # reset_agents_fun_db(client)
    # init_memeooorr_db(client)
    # basic_example(client)
    memeooorr_example(client)
