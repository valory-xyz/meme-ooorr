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

"""This package contains code to read Contribute streams on Ceramic."""
# pylint: disable=import-error

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import dotenv
import tzlocal
from rich.console import Console
from rich.table import Table
from web3 import Web3
from web3.contract import Contract

from scripts.staking_epoch_trigger import StakingContract
from scripts.test_subgraph import get_memeooorrs_from_subgraph


dotenv.load_dotenv(override=True)


EPOCH = "latest"
BASE_LEDGER_RPC = os.getenv("BASE_LEDGER_RPC_ALCHEMY")
GREEN = "bold green"
RED = "bold red"
YELLOW = "bold yellow"
POINTS_PER_UPDATE = 200
web3 = Web3(Web3.HTTPProvider(BASE_LEDGER_RPC))
UNSTAKED = "UNSTAKED"
EVICTED = "EVICTED"


STAKING_CONTRACTS = {
    "Agents.fun 1 (100 OLAS)": {
        "address": "0x2585e63df7bd9de8e058884d496658a030b5c6ce",
    },
    "Agents.fun 2 (1000 OLAS)": {
        "address": "0x26fa75ef9ccaa60e58260226a71e9d07564c01bf",
    },
    "Agents.fun 3 (5000 OLAS)": {
        "address": "0x4d4233ebf0473ca8f34d105a6256a2389176f0ce",
    },
}
STAKING_ABI_FILE = Path("scripts", "staking.json")


def get_contract_by_address(staking_contract_address) -> Optional[Dict]:
    """Get contract by address"""
    for contract_name, contract_data in STAKING_CONTRACTS.items():
        if contract_data["address"] == staking_contract_address:
            return contract_name
    return None


def load_contract(
    contract_address: str, abi_file_path: Path, has_abi_key: bool = True
) -> Contract:
    """Load a smart contract"""
    with open(abi_file_path, "r", encoding="utf-8") as abi_file:
        contract_abi = json.load(abi_file)
        if has_abi_key:
            contract_abi = contract_abi["abi"]

    contract = web3.eth.contract(address=contract_address, abi=contract_abi)
    return contract


def get_contract_info() -> Dict:
    """Get staking contract info"""

    local_tz = tzlocal.get_localzone()
    now = datetime.now(local_tz)

    contract_info = STAKING_CONTRACTS

    table = Table(title="Agents.fun staking contracts")
    columns = ["Name", "Adress", "Epoch", "Epoch end", "Used slots"]

    for column in columns:
        table.add_column(column)

    for contract_name, contract_data in STAKING_CONTRACTS.items():
        staking_contract = StakingContract(contract_data["address"], contract_name)

        contract_info[contract_name]["contract"] = staking_contract

        epoch_end_time = staking_contract.get_next_epoch_start().astimezone(local_tz)
        remaining_time_str = ""
        if now < epoch_end_time:
            remaining_seconds = int((epoch_end_time - now).total_seconds())
            remaining_time_str = f" [{remaining_seconds // 3600:02d}h {(remaining_seconds % 3600) // 60:02d}m]"

        row = [
            contract_name,
            contract_data["address"],
            str(staking_contract.get_epoch_counter()),
            f"{epoch_end_time.strftime('%Y-%m-%d %H:%M:%S')}{remaining_time_str}",
            f"{len(staking_contract.get_service_ids()):3d} / {staking_contract.max_num_services:3d}",
        ]
        table.add_row(*row, style=GREEN)

    console = Console()
    console.print(table, justify="center")

    return contract_info


def shorten_address(address: str) -> str:
    """Shorten address"""
    return address[:5] + "..." + address[-4:]


def print_table():
    """Prints the status table"""

    contract_info = get_contract_info()
    service_id_to_handle = get_memeooorrs_from_subgraph()

    table = Table(
        title=f"Agents.fun staking status [{datetime.now().strftime('%H:%M:%S %Y-%m-%d')}]"
    )
    columns = [
        "Service ID",
        "Handle",
        "Contract",
        "Epoch",
        "Activity (this epoch)",
        "Rewards (accrued)",
        "Multisig",
    ]

    for column in columns:
        table.add_column(column)

    for contract_name, contract_data in contract_info.items():
        epoch = contract_data["contract"].get_epoch_counter()

        for service_id in contract_data["contract"].get_service_ids():
            service_info = contract_data["contract"].get_service_info(service_id)

            row = [
                str(service_id),
                service_id_to_handle.get(str(service_id), None),
                contract_name,
                str(epoch),
                f"{service_info['activity_nonces']} / {service_info['required_nonces']}",
                f"{service_info['accrued_reward']/1e18:.2f}",
                service_info["multisig_address"],
            ]
            style = (
                GREEN
                if service_info["activity_nonces"] >= service_info["required_nonces"]
                else YELLOW
            )

            table.add_row(*row, style=style)

    console = Console()
    console.print(table, justify="center")


if __name__ == "__main__":
    print_table()
