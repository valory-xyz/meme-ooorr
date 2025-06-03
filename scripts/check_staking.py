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
from rich.console import Console
from rich.table import Table
from web3 import Web3
from web3.contract import Contract

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
        "slots": 20,
        "required_updates": 1,
    },
    "Agents.fun 2 (1000 OLAS)": {
        "address": "0x26fa75ef9ccaa60e58260226a71e9d07564c01bf",
        "slots": 20,
        "required_updates": 2,
    },
    "Agents.fun 3 (5000 OLAS)": {
        "address": "0x4d4233ebf0473ca8f34d105a6256a2389176f0ce",
        "slots": 20,
        "required_updates": 10,
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

    contract_info = STAKING_CONTRACTS

    table = Table(title="Agents.fun staking contracts")
    columns = ["Name", "Adress", "Epoch", "Epoch end", "Used slots"]

    for column in columns:
        table.add_column(column)

    for contract_name, contract_data in STAKING_CONTRACTS.items():
        staking_token_contract = load_contract(
            web3.to_checksum_address(contract_data["address"]), STAKING_ABI_FILE, False
        )

        epoch = staking_token_contract.functions.epochCounter().call()
        service_ids = staking_token_contract.functions.getServiceIds().call()
        next_epoch_start = datetime.fromtimestamp(
            staking_token_contract.functions.getNextRewardCheckpointTimestamp().call()
        )

        contract_info[contract_name]["contract"] = staking_token_contract
        contract_info[contract_name]["epoch"] = epoch
        contract_info[contract_name]["next_epoch_start"] = next_epoch_start.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        contract_info[contract_name]["slots"] = contract_data["slots"]
        contract_info[contract_name]["used_slots"] = len(service_ids)
        contract_info[contract_name]["free_slots"] = contract_data["slots"] - len(
            service_ids
        )
        contract_info[contract_name]["service_ids"] = service_ids

        row = [
            contract_name,
            contract_data["address"],
            str(epoch),
            next_epoch_start.strftime("%Y-%m-%d %H:%M:%S UTC"),
            f"{len(service_ids):3d} / {contract_data['slots']:3d}",
        ]
        table.add_row(*row, style=GREEN)

    console = Console()
    console.print(table, justify="center")

    return contract_info


def get_user_info(
    contract_info: Dict, contract_name: str, service_id: int
) -> Dict:  # pylint: disable=too-many-locals
    """Get user info"""

    is_evicted = (
        contract_info[contract_name]["contract"]
        .functions.getStakingState(service_id)
        .call()
        == 2
    )

    accrued_rewards = (
        contract_info[contract_name]["contract"]
        .functions.calculateStakingReward(service_id)
        .call()
    )
    this_epoch_rewards = (
        contract_info[contract_name]["contract"]
        .functions.calculateStakingLastReward(service_id)
        .call()
    )
    this_epoch = contract_info[contract_name]["epoch"]

    (
        multisig,
        _,
        _,
        _,
        _,
        _,
    ) = (
        contract_info[contract_name]["contract"]
        .functions.getServiceInfo(service_id)
        .call()
    )

    user_info = {
        "staked": True,
        "evicted": is_evicted,
        "staking_contract_name": contract_name,
        "epoch": str(this_epoch),
        "next_epoch_start": contract_info[contract_name]["next_epoch_start"],
        "this_epoch_rewards": f"{this_epoch_rewards / 1e18:6.2f}",
        "accrued_rewards": f"{accrued_rewards / 1e18:6.2f}",
        "multisig": multisig,
        "color": GREEN,
    }

    return user_info


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
        "Rewards (this epoch)",
        "Rewards (accrued)",
        "Multisig",
    ]

    for column in columns:
        table.add_column(column)

    for contract_name, contract_data in contract_info.items():
        for service_id in contract_data["service_ids"]:
            user_info = get_user_info(contract_info, contract_name, service_id)

            row = [
                str(service_id),
                service_id_to_handle.get(str(service_id), None),
                user_info["staking_contract_name"],
                user_info["epoch"],
                user_info["this_epoch_rewards"],
                user_info["accrued_rewards"],
                user_info["multisig"],
            ]
            style = user_info["color"]

            table.add_row(*row, style=style)

    console = Console()
    console.print(table, justify="center")


if __name__ == "__main__":
    print_table()
