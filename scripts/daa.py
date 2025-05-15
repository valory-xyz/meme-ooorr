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

"""Calculate DAAs."""

import json
import os
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

import dotenv
import requests
from pydantic import BaseModel
from web3 import Web3
from web3.contract import Contract


dotenv.load_dotenv(override=True)

BASESCAN_API_BASE = "https://api.basescan.org/api"
BASE_BLOCKS_PER_SECOND = 2

CELOSCAN_API_BASE = "https://api.celoscan.io/api"
CELO_BLOCKS_PER_SECOND = 1

BLOCK_MARGIN = 2
MEMEOOORR_DESCRIPTION_PATTERN = r"^Memeooorr @(\w+)$"

DAA_DB_PATH = "daa_base.json"


CHAIN_CONFIGS = {
    "BASE": {
        "API_BASE": BASESCAN_API_BASE,
        "API_KEY": os.getenv("BASESCAN_API_KEY"),
        "BLOCKS_PER_DAY": 24 * 60 * 60 / BASE_BLOCKS_PER_SECOND,
        "WEB3": Web3(Web3.HTTPProvider(os.getenv("BASE_LEDGER_RPC"))),
        "SERVICE_REGISTRY_ADDRES": "0x3C1fF68f5aa342D296d4DEe4Bb1cACCA912D95fE",
    },
    "CELO": {
        "API_BASE": CELOSCAN_API_BASE,
        "API_KEY": os.getenv("CELOSCAN_API_KEY"),
        "BLOCKS_PER_DAY": 24 * 60 * 60 / CELO_BLOCKS_PER_SECOND,
        "WEB3": Web3(Web3.HTTPProvider(os.getenv("CELO_LEDGER_RPC"))),
        "SERVICE_REGISTRY_ADDRES": "0xE3607b00E75f6405248323A9417ff6b39B244b50",
    },
}

PACKAGE_QUERY = """
query getPackages($package_type: String!, $first: Int, $skip: Int) {
    units(where: {packageType: $package_type}, first: $first, skip: $skip) {
        id,
        packageType,
        publicId,
        packageHash,
        tokenId,
        metadataHash,
        description,
        owner,
        image
    }
}
"""

HTTP_OK = 200


class Transaction(BaseModel):
    """Transaction"""

    block_number: int
    timestamp: int
    transaction_hash: str


class AgentsFun(BaseModel):
    """AgentsFun"""

    service_id: int
    multisig: str
    transactions: Dict[str, Transaction] = {}


class DAAdatabase(BaseModel):
    """DAAdatabase"""

    agents: Dict[int, AgentsFun] = {}


if Path(DAA_DB_PATH).exists():
    with open(DAA_DB_PATH, "r", encoding="utf8") as daa_db_file:
        daa_json = json.load(daa_db_file)
        daa_db = DAAdatabase(**daa_json)
else:
    daa_db = DAAdatabase()


def save_daa_db():
    """Save the DAA database."""
    with open(DAA_DB_PATH, "w", encoding="utf8") as db_file:
        json.dump(daa_db.model_dump(), db_file, indent=4, default=str)


def load_contract(
    contract_address: str,
    abi_file_name: str,
    chain_config: Dict,
    has_abi_key: bool = True,
) -> Contract:
    """Load a smart contract"""
    with open(Path("abis", f"{abi_file_name}.json"), "r", encoding="utf-8") as abi_file:
        contract_abi = json.load(abi_file)
        if has_abi_key:
            contract_abi = contract_abi["abi"]

    contract = chain_config["WEB3"].eth.contract(
        address=contract_address, abi=contract_abi
    )
    return contract


def get_latest_block_number(chain_config):
    """Get the latest block number from BaseScan."""
    params = {
        "module": "proxy",
        "action": "eth_blockNumber",
        "apikey": chain_config["API_KEY"],
    }
    response = requests.get(chain_config["API_BASE"], params=params, timeout=60)
    data = response.json()

    return int(data["result"], 16) if "result" in data else None


def get_block_by_datetime(
    dt: datetime, chain_config, closest: str = "before"
) -> Optional[int]:
    """Get the block number closest to a given datetime."""
    now = int(datetime.now(timezone.utc).timestamp())
    timestamp = int(dt.timestamp())

    timestamp = min(now, timestamp)

    params = {
        "module": "block",
        "action": "getblocknobytime",
        "timestamp": timestamp,
        "closest": closest,
        "apikey": chain_config["API_KEY"],
    }

    response = requests.get(chain_config["API_BASE"], params=params, timeout=60)
    data = response.json()

    if data["status"] != "1":
        print(f"Scanner error: {data}")
        return None

    return int(data["result"])


def update_transactions(chain_config, day, latest_parsed_block):
    """Get transactions."""

    day_start = datetime.combine(day, time.min, tzinfo=timezone.utc)
    day_end = datetime.combine(day, time.max, tzinfo=timezone.utc)

    start_block = get_block_by_datetime(day_start, chain_config)
    end_block = get_block_by_datetime(day_end, chain_config)

    # Check if this day has been already processed
    if latest_parsed_block is not None and latest_parsed_block > end_block:
        print(f"Already processed day {day}")
        return

    if not start_block or not end_block:
        print(f"Error while pulling the transactions: {day}")
        return

    for service_id, service in daa_db.agents.items():
        print(f"    Reading transactions for service {service_id}")

        params = {
            "module": "account",
            "action": "txlist",
            "address": service.multisig,
            "startblock": start_block,
            "endblock": end_block,
            "sort": "desc",
            "apikey": chain_config["API_KEY"],
        }

        response = requests.get(chain_config["API_BASE"], params=params, timeout=60)
        data = response.json()

        if data["status"] != "1":
            if data.get("message", None) != "No transactions found":
                print(f"Error while pulling the transactions: {data}")
            continue

        for tx in data["result"]:
            daa_db.agents[service_id].transactions[tx["hash"]] = Transaction(
                block_number=int(tx["blockNumber"]),
                timestamp=int(tx["timeStamp"]),
                transaction_hash=tx["hash"],
            )

    save_daa_db()


def was_service_active(service_id: int, day: datetime.date) -> bool:
    """Check if the address has made any transactions in a specific day."""
    day_start = datetime.combine(day, time.min, tzinfo=timezone.utc)
    day_end = datetime.combine(day, time.max, tzinfo=timezone.utc)

    transactions = [
        tx
        for tx in daa_db.agents[service_id].transactions.values()
        if tx.timestamp >= day_start.timestamp() and tx.timestamp <= day_end.timestamp()
    ]

    return len(transactions) > 0


def get_packages(package_type: str):
    """Gets minted packages from the Olas subgraph"""

    url = "https://subgraph.autonolas.tech/subgraphs/name/autonolas-base/"  # TODO: Celo

    headers = {"Content-Type": "application/json"}

    query = {
        "query": PACKAGE_QUERY,
        "variables": {"package_type": package_type, "first": 1000, "skip": None},
    }

    response = requests.post(url=url, json=query, headers=headers, timeout=60)

    # Handle HTTP errors
    if response.status_code != HTTP_OK:
        print(f"Error while pulling the memes from subgraph: {response}")
        return []

    response_json = response.json()["data"]  # type: ignore
    return response_json


def update_agents(chain_config):
    """Update agent info."""

    service_registry = load_contract(
        contract_address=chain_config["SERVICE_REGISTRY_ADDRES"],
        abi_file_name="service_registry/ServiceRegistry",
        chain_config=chain_config,
        has_abi_key=False,
    )

    n_services = service_registry.functions.totalSupply().call()

    if not daa_db.agents:
        latest_parsed_service = 0
    else:
        latest_parsed_service = max(daa_db.agents.keys())

    if latest_parsed_service >= n_services:
        return

    for service_id in range(latest_parsed_service, n_services):
        print(f"Reading service {service_id} of {n_services}")
        (
            security_deposit,  # pylint: disable=unused-variable
            multisig,
            config_hash,  # pylint: disable=unused-variable
            threshold,  # pylint: disable=unused-variable
            max_num_agent_instances,  # pylint: disable=unused-variable
            num_agent_instances,  # pylint: disable=unused-variable
            state,  # pylint: disable=unused-variable
            agent_ids,
        ) = service_registry.functions.getService(service_id).call()

        if agent_ids != [43]:
            continue

        daa_db.agents[service_id] = AgentsFun(
            service_id=service_id,
            multisig=multisig,
        )

    save_daa_db()


def calculate_daas(chain_config):
    """Calculate the DAAs."""

    # Read all new agents from the chain
    update_agents(chain_config)

    if not daa_db.agents:
        print("No agents found")
        return

    # Get the latest parsed number
    latest_parsed_block = max(
        (
            max(tx.block_number for tx in service.transactions.values())
            for service in daa_db.agents.values()
            if service.transactions
        ),
        default=None,
    )

    # Update transactions and calculate daas for the past week
    today = datetime.utcnow().date()

    daas = {}

    for i in range(7):
        day = today - timedelta(days=i)
        print(f"Reading transactions for {day}")

        update_transactions(chain_config, day, latest_parsed_block)

        daas[day] = 0

        for service_id in daa_db.agents.keys():
            if was_service_active(service_id, day):
                daas[day] += 1

        print(f"DAAs for {day}: {daas[day]}")

    trailing_average = sum(daas.values()) / len(daas)
    print(f"Trailing average: {trailing_average}")


if __name__ == "__main__":
    calculate_daas(CHAIN_CONFIGS["BASE"])

# TODO: other KPIs
#  engagement rate = total no. of engagements/total impressions
#  engagements = likes + retweets + replies + profile clicks + link clicks
