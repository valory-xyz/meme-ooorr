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
import time
from pathlib import Path
from typing import Dict

import dotenv
import requests
from web3 import Web3
from web3.contract import Contract


dotenv.load_dotenv(override=True)

BASESCAN_API_BASE = "https://api.basescan.org/api"
BASE_BLOCKS_PER_SECOND = 2

CELOSCAN_API_BASE = "https://api.celoscan.io/api"
CELO_BLOCKS_PER_SECOND = 1

BLOCK_MARGIN = 2
MEMEOOORR_DESCRIPTION_PATTERN = r"^Memeooorr @(\w+)$"


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


def get_transactions(address, chain_config):
    """Get transactions for a given address."""

    end_block = get_latest_block_number(chain_config)
    if not end_block:
        return None

    # Parse the last 24 hours only
    start_block = end_block - int(BLOCK_MARGIN * chain_config["BLOCKS_PER_DAY"])

    params = {
        "module": "account",
        "action": "txlist",
        "address": address,
        "startblock": start_block,
        "endblock": end_block,
        "sort": "desc",
        "apikey": chain_config["API_KEY"],
    }
    response = requests.get(chain_config["API_BASE"], params=params, timeout=60)
    data = response.json()
    if data["status"] != "1":
        return []
    return data["result"]


def has_recent_transactions(address, chain_config):
    """Check if the address has made any transactions in the last 24 hours."""
    print(f"Checking address {address} for recent transactions... ", end="")
    transactions = get_transactions(address, chain_config)
    if transactions is None:
        print("inactive")
        return None

    if not transactions:
        print("inactive")
        return False

    latest_tx_timestamp = int(transactions[0]["timeStamp"])
    a_day_ago = int(time.time()) - 24 * 3600

    active = latest_tx_timestamp >= a_day_ago
    print("active" if active else "inactive")
    return active


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


def get_memeooorr_safes(chain_config):
    """Read service details from the service registry"""
    service_registry = load_contract(
        contract_address=chain_config["SERVICE_REGISTRY_ADDRES"],
        abi_file_name="service_registry/ServiceRegistry",
        chain_config=chain_config,
        has_abi_key=False,
    )

    n_services = service_registry.functions.totalSupply().call()

    safes = []
    for service_id in range(n_services):
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
        safes.append(multisig)
    return safes


def calculate_daas(chain_config):
    """Calculate the DAAs."""
    safes = get_memeooorr_safes(chain_config)
    print(f"Found {len(safes)} agents.fun safes")

    daas = [
        address for address in safes if has_recent_transactions(address, chain_config)
    ]
    return len(daas)


if __name__ == "__main__":
    n_daas_base = calculate_daas(CHAIN_CONFIGS["BASE"])
    n_daas_celo = calculate_daas(CHAIN_CONFIGS["CELO"])

    print(f"DAAS Base: {n_daas_base}")
    print(f"DAAS Celo: {n_daas_celo}")

#  engagement rate = total no. of engagements/total impressions
#  engagements = likes + retweets + replies + profile clicks + link clicks
