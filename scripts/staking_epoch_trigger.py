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

"""This package contains code to interact with staking contracts."""
# pylint: disable=too-many-instance-attributes,unused-variable

import json
import math
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import dotenv
import tzlocal
from web3 import Web3
from web3.contract import Contract


dotenv.load_dotenv(override=True)


def load_contract(
    contract_address: str, web3: Web3, abi_file_path: Path, has_abi_key: bool = True
) -> Contract:
    """Load a smart contract"""
    with open(abi_file_path, "r", encoding="utf-8") as abi_file:
        contract_abi = json.load(abi_file)
        if has_abi_key:
            contract_abi = contract_abi["abi"]

    contract = web3.eth.contract(address=contract_address, abi=contract_abi)
    return contract


STAKING_CONTRACTS = {
    "Agents.fun 1": {
        "address": "0x2585e63df7bd9de8e058884d496658a030b5c6ce",
    },
    "Agents.fun 2": {
        "address": "0x26fa75ef9ccaa60e58260226a71e9d07564c01bf",
    },
    "Agents.fun 3": {
        "address": "0x4d4233ebf0473ca8f34d105a6256a2389176f0ce",
    },
}


class StakingContract:
    """Class to interact with the staking contract."""

    def __init__(self, contract_address: str, contract_name: str):
        self.contract_address = contract_address
        self.contract_name = contract_name
        self.web3 = Web3(Web3.HTTPProvider(os.getenv("BASE_LEDGER_RPC_ALCHEMY")))
        self.contract = load_contract(
            self.web3.to_checksum_address(contract_address),
            self.web3,
            Path("scripts", "staking.json"),
            False,
        )
        self.private_key = os.getenv("PRIVATE_KEY")
        self.account = (
            self.web3.eth.account.from_key(self.private_key).address
            if self.private_key
            else None
        )
        self.activity_checker_address = self.contract.functions.activityChecker().call()
        self.activity_checker = load_contract(
            self.web3.to_checksum_address(self.activity_checker_address),
            self.web3,
            Path("scripts", "activity_checker.json"),
            False,
        )
        self.max_num_services = self.contract.functions.maxNumServices().call()
        self.available_rewards = self.contract.functions.availableRewards().call()
        self.balance = self.contract.functions.balance().call()
        self.liveness_period = self.contract.functions.livenessPeriod().call()
        self.liveness_ratio = self.activity_checker.functions.livenessRatio().call()
        self.rewards_per_second = self.contract.functions.rewardsPerSecond().call()
        self.stop_thread = False
        self.trigger_thread = None

    def get_service_ids(self):
        """Get the satked service ids"""
        return self.contract.functions.getServiceIds().call()

    def calculate_accrued_staking_reward(self, service_id: int) -> int:
        """Calculate the accrued staking reward for a given service ID."""
        return self.contract.functions.calculateStakingLastReward(service_id).call()

    def calculate_staking_reward(self, service_id: int) -> int:
        """Calculate the current staking reward for a given service ID."""
        return self.contract.functions.calculateStakingReward(service_id).call()

    def get_epoch_counter(self) -> int:
        """Get the current epoch counter from the staking contract."""
        return self.contract.functions.epochCounter().call()

    def get_next_epoch_start(self) -> datetime:
        """Calculate the start time of the next epoch."""
        return datetime.fromtimestamp(
            self.contract.functions.getNextRewardCheckpointTimestamp().call(),
            tz=timezone.utc,
        )

    def get_service_info(self, service_id: int):
        """Get information about services in the staking contract."""
        (
            multisig_address,
            owner_address,
            nonces_on_last_checkpoint,
            ts_start,
            accrued_reward,
            inactivity,
        ) = self.contract.functions.getServiceInfo(service_id).call()
        total_nonces = self.get_multisig_nonces(multisig_address)
        nonces_since_last_checkpoint = total_nonces[0] - nonces_on_last_checkpoint[0]
        required_nonces = self.get_required_requests()
        return {
            "activity_nonces": nonces_since_last_checkpoint,
            "multisig_address": multisig_address,
            "accrued_reward": accrued_reward,
            "required_nonces": required_nonces,
            "has_enough_nonces": nonces_since_last_checkpoint >= required_nonces,
            "remaining_epoch_seconds": (
                self.get_next_epoch_start() - datetime.now(timezone.utc)
            ).total_seconds(),
        }

    def get_staking_state(self, service_id: int):
        """Get the staking state for a given service ID."""
        return self.contract.functions.getStakingState(service_id).call()

    def get_multisig_nonces(self, multisig: str) -> int:
        """Get the number of nonces for a multisig address."""
        return self.activity_checker.functions.getMultisigNonces(multisig).call()

    def is_ratio_pass(
        self, current_nonces: int, last_nonces: int, timestamp: int
    ) -> bool:
        """Check if the liveness ratio is passed."""
        return self.activity_checker.functions.isRatioPass(
            current_nonces, last_nonces, timestamp
        ).call()

    def get_ts_checkpoint(self) -> int:
        """Get the timestamp of the last checkpoint."""
        return self.contract.functions.tsCheckpoint().call()

    def get_required_requests(self) -> int:
        """Calculate the required requests for the next epoch."""
        REQUESTS_SAFETY_MARGIN = 1
        now_ts = time.time()
        return math.ceil(
            (
                max(self.liveness_period, now_ts - self.get_ts_checkpoint())
                * self.liveness_ratio
            )
            / 1e18
            + REQUESTS_SAFETY_MARGIN
        )

    def wait_for_no_pending_tx(
        self, max_wait_seconds: int = 60, poll_interval: float = 2.0
    ):
        """Wait for no pending transactions for a specified time."""
        start_time = time.time()
        while time.time() - start_time < max_wait_seconds:
            latest_nonce = self.web3.eth.get_transaction_count(
                self.account, block_identifier="latest"
            )
            pending_nonce = self.web3.eth.get_transaction_count(
                self.account, block_identifier="pending"
            )

            if pending_nonce == latest_nonce:
                return True

            time.sleep(poll_interval)

        return False

    def sign_and_send_transaction(self, transaction: dict) -> bool:
        """Sign and send a transaction."""
        signed_txn = self.web3.eth.account.sign_transaction(
            transaction, private_key=self.private_key
        )
        txn_hash = self.web3.eth.send_raw_transaction(signed_txn.raw_transaction)
        receipt = self.web3.eth.wait_for_transaction_receipt(txn_hash)
        if receipt.status == 1:
            self.wait_for_no_pending_tx()
            return True
        return False

    def estimate_gas(self, function_name: str, *args) -> int:
        """Estimate gas for a contract function call."""
        try:
            func = getattr(self.contract.functions, function_name)(*args)
            estimated_gas = func.estimate_gas({"from": self.account})
            return int(estimated_gas * 1.1)
        except Exception as e:
            print(f"Could not estimate gas for {function_name}: {e}")
            raise

    def calculate_transaction_params(self, function_name: str) -> dict:
        """Calculate transaction parameters for a contract function call."""
        params = {
            "from": self.account,
            "nonce": self.web3.eth.get_transaction_count(self.account),
            "gas": self.estimate_gas(function_name),
            "gasPrice": self.web3.eth.gas_price,
        }
        return params

    def trigger_epoch(self) -> bool:
        """Trigger the next epoch in the staking contract."""
        transaction = self.contract.functions.checkpoint().build_transaction(
            self.calculate_transaction_params("checkpoint")
        )
        return self.sign_and_send_transaction(transaction)

    def run_trigger_loop(self):
        """Run a loop to trigger the epoch at the next start time."""

        while not self.stop_thread:
            next_epoch_start = self.get_next_epoch_start()
            now = datetime.now(timezone.utc)

            # If the epoch start is in the future, the epoch has not finalized
            if next_epoch_start > now:
                wait_time = int((next_epoch_start - now).total_seconds() + 10)
                print(
                    f"[{self.contract_name}] Next epoch starts at {next_epoch_start.astimezone(tzlocal.get_localzone())}. Waiting for {wait_time} seconds.",
                    flush=True,
                )
                time.sleep(wait_time)
                continue

            success = self.trigger_epoch()
            if success:
                print("Epoch triggered successfully.")
            else:
                print("Failed to trigger the epoch.")

    def run_trigger_thread(self):
        """Run the trigger loop in a separate thread."""
        self.trigger_thread = threading.Thread(target=self.run_trigger_loop)
        self.trigger_thread.start()

    def stop_trigger_thread(self):
        """Stop the trigger thread."""
        self.stop_thread = True
        if self.trigger_thread:
            self.trigger_thread.join()
            self.trigger_thread = None


if __name__ == "__main__":
    for staking_contract_name, staking_contract_data in STAKING_CONTRACTS.items():
        staking_contract_data["contract"] = StakingContract(
            staking_contract_data["address"], staking_contract_name
        )
        staking_contract_data["contract"].run_trigger_thread()
