# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023-2024 Valory AG
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

"""Tests for the MemeFactory contract."""

# pylint: disable=redefined-outer-name,too-few-public-methods

import json
import re
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
import web3 as web3_module

from packages.dvilela.contracts.meme_factory.contract import (
    MemeFactoryContract,
    PUBLIC_ID,
)

DUMMY_ADDRESS = "0x" + "1" * 40
DUMMY_TX_HASH = "0x" + "a" * 64


@pytest.fixture
def mock_ledger_api() -> MagicMock:
    """Return a mocked EthereumApi."""
    return MagicMock()


@pytest.fixture
def mock_contract_instance() -> MagicMock:
    """Return a mocked contract instance."""
    return MagicMock()


class TestMemeFactoryContractMeta:
    """Test contract metadata."""

    def test_public_id(self) -> None:
        """Test public ID is set correctly."""
        assert str(PUBLIC_ID) == "dvilela/meme_factory:0.1.0"

    def test_contract_id(self) -> None:
        """Test contract_id matches PUBLIC_ID."""
        assert MemeFactoryContract.contract_id == PUBLIC_ID


class TestBuildTransactions:
    """Test transaction builder methods."""

    @patch.object(MemeFactoryContract, "get_instance")
    def test_build_summon_tx(
        self, mock_get_instance: MagicMock, mock_ledger_api: MagicMock
    ) -> None:
        """Test build_summon_tx encodes ABI correctly."""
        instance = MagicMock()
        instance.encode_abi.return_value = "0xdeadbeef"
        mock_get_instance.return_value = instance

        result = MemeFactoryContract.build_summon_tx(
            mock_ledger_api, DUMMY_ADDRESS, "TestToken", "TT", 1000
        )

        instance.encode_abi.assert_called_once_with(
            abi_element_identifier="summonThisMeme",
            args=["TestToken", "TT", 1000],
        )
        assert result == {"data": bytes.fromhex("deadbeef")}

    @patch.object(MemeFactoryContract, "get_instance")
    def test_build_summon_tx_default_supply(
        self, mock_get_instance: MagicMock, mock_ledger_api: MagicMock
    ) -> None:
        """Test build_summon_tx uses default token supply."""
        instance = MagicMock()
        instance.encode_abi.return_value = "0xaa"
        mock_get_instance.return_value = instance

        MemeFactoryContract.build_summon_tx(mock_ledger_api, DUMMY_ADDRESS, "Tok", "T")

        call_args = instance.encode_abi.call_args
        assert call_args[1]["args"][2] == 1000000000000000000000000

    @patch.object(MemeFactoryContract, "get_instance")
    def test_build_heart_tx(
        self, mock_get_instance: MagicMock, mock_ledger_api: MagicMock
    ) -> None:
        """Test build_heart_tx encodes ABI correctly."""
        instance = MagicMock()
        instance.encode_abi.return_value = "0xabcd"
        mock_get_instance.return_value = instance

        result = MemeFactoryContract.build_heart_tx(
            mock_ledger_api, DUMMY_ADDRESS, "42"
        )

        instance.encode_abi.assert_called_once_with(
            abi_element_identifier="heartThisMeme",
            args=["42"],
        )
        assert result == {"data": bytes.fromhex("abcd")}

    @patch.object(MemeFactoryContract, "get_instance")
    def test_build_unleash_tx(
        self, mock_get_instance: MagicMock, mock_ledger_api: MagicMock
    ) -> None:
        """Test build_unleash_tx encodes ABI correctly."""
        instance = MagicMock()
        instance.encode_abi.return_value = "0x1234"
        mock_get_instance.return_value = instance

        result = MemeFactoryContract.build_unleash_tx(
            mock_ledger_api, DUMMY_ADDRESS, "7"
        )

        instance.encode_abi.assert_called_once_with(
            abi_element_identifier="unleashThisMeme",
            args=["7"],
        )
        assert result == {"data": bytes.fromhex("1234")}

    @patch.object(MemeFactoryContract, "get_instance")
    def test_build_collect_tx(
        self, mock_get_instance: MagicMock, mock_ledger_api: MagicMock
    ) -> None:
        """Test build_collect_tx checksums address and encodes ABI."""
        instance = MagicMock()
        instance.encode_abi.return_value = "0x5678"
        mock_get_instance.return_value = instance

        meme_addr = "0x" + "ab" * 20
        result = MemeFactoryContract.build_collect_tx(
            mock_ledger_api, DUMMY_ADDRESS, meme_addr
        )

        call_args = instance.encode_abi.call_args
        assert call_args[1]["abi_element_identifier"] == "collectThisMeme"
        # Address should be checksummed
        checksummed = call_args[1]["args"][0]
        assert checksummed == web3_module.Web3.to_checksum_address(meme_addr)
        assert result == {"data": bytes.fromhex("5678")}

    @patch.object(MemeFactoryContract, "get_instance")
    def test_build_purge_tx(
        self, mock_get_instance: MagicMock, mock_ledger_api: MagicMock
    ) -> None:
        """Test build_purge_tx checksums address and encodes ABI."""
        instance = MagicMock()
        instance.encode_abi.return_value = "0xffee"
        mock_get_instance.return_value = instance

        meme_addr = "0x" + "cd" * 20
        result = MemeFactoryContract.build_purge_tx(
            mock_ledger_api, DUMMY_ADDRESS, meme_addr
        )

        call_args = instance.encode_abi.call_args
        assert call_args[1]["abi_element_identifier"] == "purgeThisMeme"
        checksummed = call_args[1]["args"][0]
        assert checksummed == web3_module.Web3.to_checksum_address(meme_addr)
        assert result == {"data": bytes.fromhex("ffee")}

    @patch.object(MemeFactoryContract, "get_instance")
    def test_build_burn_tx(
        self, mock_get_instance: MagicMock, mock_ledger_api: MagicMock
    ) -> None:
        """Test build_burn_tx encodes ABI with no args."""
        instance = MagicMock()
        instance.encode_abi.return_value = "0xaa00"
        mock_get_instance.return_value = instance

        result = MemeFactoryContract.build_burn_tx(mock_ledger_api, DUMMY_ADDRESS)

        instance.encode_abi.assert_called_once_with(
            abi_element_identifier="scheduleForAscendance",
            args=[],
        )
        assert result == {"data": bytes.fromhex("aa00")}


class TestGetTokenData:
    """Test get_token_data event parsing."""

    @patch.object(MemeFactoryContract, "get_instance")
    def test_get_token_data_found(
        self, mock_get_instance: MagicMock, mock_ledger_api: MagicMock
    ) -> None:
        """Test get_token_data returns parsed Summoned event data."""
        instance = MagicMock()
        mock_get_instance.return_value = instance

        mock_event = MagicMock()
        mock_event.args = {
            "summoner": "0xSummoner",
            "memeNonce": "1",
            "amount": 100,
        }
        instance.events.Summoned.return_value.process_log.return_value = mock_event

        mock_ledger_api.api.eth.get_transaction_receipt.return_value = {
            "logs": [MagicMock()]
        }

        result = MemeFactoryContract.get_token_data(
            mock_ledger_api, DUMMY_ADDRESS, DUMMY_TX_HASH
        )

        assert result == {
            "summoner": "0xSummoner",
            "token_nonce": "1",
            "eth_contributed": 100,
        }

    @patch.object(MemeFactoryContract, "get_instance")
    def test_get_token_data_not_found(
        self, mock_get_instance: MagicMock, mock_ledger_api: MagicMock
    ) -> None:
        """Test get_token_data returns None values when no Summoned event found."""
        instance = MagicMock()
        mock_get_instance.return_value = instance

        instance.events.Summoned.return_value.process_log.side_effect = (
            web3_module.exceptions.MismatchedABI()
        )

        mock_ledger_api.api.eth.get_transaction_receipt.return_value = {
            "logs": [MagicMock()]
        }

        result = MemeFactoryContract.get_token_data(
            mock_ledger_api, DUMMY_ADDRESS, DUMMY_TX_HASH
        )

        assert result == {
            "token_address": None,
            "summoner": None,
            "eth_contributed": None,
        }

    @patch.object(MemeFactoryContract, "get_instance")
    def test_get_token_data_empty_logs(
        self, mock_get_instance: MagicMock, mock_ledger_api: MagicMock
    ) -> None:
        """Test get_token_data with no logs returns None values."""
        instance = MagicMock()
        mock_get_instance.return_value = instance

        mock_ledger_api.api.eth.get_transaction_receipt.return_value = {"logs": []}

        result = MemeFactoryContract.get_token_data(
            mock_ledger_api, DUMMY_ADDRESS, DUMMY_TX_HASH
        )

        assert result == {
            "token_address": None,
            "summoner": None,
            "eth_contributed": None,
        }


class TestGetMemeSummonsInfo:
    """Test get_meme_summons_info."""

    @patch.object(MemeFactoryContract, "get_instance")
    def test_with_token_nonce(
        self, mock_get_instance: MagicMock, mock_ledger_api: MagicMock
    ) -> None:
        """Test fetching summons info with a known nonce."""
        instance = MagicMock()
        mock_get_instance.return_value = instance

        meme_summons = MagicMock()
        meme_summons.return_value.call.return_value = ["data1", "data2"]
        instance.functions.memeSummons = meme_summons

        result = MemeFactoryContract.get_meme_summons_info(
            mock_ledger_api, DUMMY_ADDRESS, token_nonce="5"
        )

        meme_summons.assert_called_once_with("5")
        assert result == {"token_data": ["data1", "data2"]}

    @patch.object(MemeFactoryContract, "get_instance")
    def test_with_token_address(
        self, mock_get_instance: MagicMock, mock_ledger_api: MagicMock
    ) -> None:
        """Test fetching summons info by resolving nonce from token address."""
        instance = MagicMock()
        mock_get_instance.return_value = instance

        # memeTokenNonces(address) -> nonce
        nonce_fn = MagicMock()
        nonce_fn.return_value.call.return_value = "10"
        instance.functions.memeTokenNonces = nonce_fn

        # memeSummons(nonce) -> data
        summons_fn = MagicMock()
        summons_fn.return_value.call.return_value = ["d1"]
        instance.functions.memeSummons = summons_fn

        result = MemeFactoryContract.get_meme_summons_info(
            mock_ledger_api, DUMMY_ADDRESS, token_address="0xToken"
        )

        nonce_fn.assert_called_once_with("0xToken")
        summons_fn.assert_called_once_with("10")
        assert result == {"token_data": ["d1"]}


class TestGetSummonData:
    """Test get_summon_data."""

    @patch.object(MemeFactoryContract, "get_events")
    @patch.object(MemeFactoryContract, "get_instance")
    def test_get_summon_data(
        self,
        mock_get_instance: MagicMock,
        mock_get_events: MagicMock,
        mock_ledger_api: MagicMock,
    ) -> None:
        """Test get_summon_data combines event data with on-chain summon data."""
        mock_get_events.return_value = {
            "events": [
                {
                    "summoner": "0xAlice",
                    "token_nonce": "1",
                    "eth_contributed": 100,
                },
            ]
        }

        instance = MagicMock()
        mock_get_instance.return_value = instance

        # memeSummons returns: (name, ticker, supply, eth, summon_time, unleash_time, hearts, pos_id, native_first)
        summons_fn = MagicMock()
        summons_fn.return_value.call.return_value = [
            "TestCoin",
            "TC",
            1000,
            100,
            1000000,
            0,
            5,
            42,
            True,
        ]
        instance.functions.memeSummons = summons_fn

        result = MemeFactoryContract.get_summon_data(
            mock_ledger_api, DUMMY_ADDRESS, from_block=100
        )

        assert len(result["tokens"]) == 1
        token = result["tokens"][0]
        assert token["summoner"] == "0xAlice"
        assert token["token_nonce"] == "1"
        assert token["token_name"] == "TestCoin"
        assert token["token_ticker"] == "TC"
        assert token["token_supply"] == 1000
        assert token["token_address"] is None
        assert token["decimals"] == 18


class TestGetPurgeData:
    """Test get_purge_data."""

    @patch.object(MemeFactoryContract, "get_events")
    def test_get_purge_data(
        self, mock_get_events: MagicMock, mock_ledger_api: MagicMock
    ) -> None:
        """Test get_purge_data extracts purged addresses from events."""
        mock_get_events.return_value = {
            "events": [
                {"token_address": "0xA"},
                {"token_address": "0xB"},
            ]
        }

        result = MemeFactoryContract.get_purge_data(
            mock_ledger_api, DUMMY_ADDRESS, from_block=50
        )

        assert result == {"purged_addresses": ["0xA", "0xB"]}


class TestGetEvents:
    """Test get_events with different event types."""

    @patch.object(MemeFactoryContract, "get_instance")
    def test_summoned_events(
        self, mock_get_instance: MagicMock, mock_ledger_api: MagicMock
    ) -> None:
        """Test parsing Summoned events."""
        instance = MagicMock()
        mock_get_instance.return_value = instance

        mock_ledger_api.api.eth.get_block_number.return_value = 10100

        mock_event_entry = MagicMock()
        mock_event_entry.args = {
            "summoner": "0xSumm",
            "memeNonce": "3",
            "amount": 500,
        }

        event_obj = MagicMock()
        event_obj.create_filter.return_value.get_all_entries.return_value = [
            mock_event_entry
        ]
        instance.events.Summoned = event_obj

        result = MemeFactoryContract.get_events(
            mock_ledger_api, DUMMY_ADDRESS, "Summoned", from_block=10000
        )

        events: List[Dict[str, Any]] = result["events"]  # type: ignore[assignment]
        assert len(events) == 1
        assert events[0]["summoner"] == "0xSumm"
        assert events[0]["token_nonce"] == "3"
        assert events[0]["eth_contributed"] == 500
        assert "latest_block" in result

    @patch.object(MemeFactoryContract, "get_instance")
    def test_unleashed_events(
        self, mock_get_instance: MagicMock, mock_ledger_api: MagicMock
    ) -> None:
        """Test parsing Unleashed events."""
        instance = MagicMock()
        mock_get_instance.return_value = instance

        mock_ledger_api.api.eth.get_block_number.return_value = 10100

        mock_event_entry = MagicMock()
        mock_event_entry.args = {
            "unleasher": "0xUnl",
            "memeNonce": "2",
            "memeToken": "0xToken",
            "lpTokenId": 99,
            "liquidity": 1000,
        }

        event_obj = MagicMock()
        event_obj.create_filter.return_value.get_all_entries.return_value = [
            mock_event_entry
        ]
        instance.events.Unleashed = event_obj

        result = MemeFactoryContract.get_events(
            mock_ledger_api, DUMMY_ADDRESS, "Unleashed", from_block=10000
        )

        events: List[Dict[str, Any]] = result["events"]  # type: ignore[assignment]
        assert len(events) == 1
        assert events[0]["token_unleasher"] == "0xUnl"
        assert events[0]["token_address"] == "0xToken"
        assert events[0]["position_id"] == 99

    @patch.object(MemeFactoryContract, "get_instance")
    def test_purged_events(
        self, mock_get_instance: MagicMock, mock_ledger_api: MagicMock
    ) -> None:
        """Test parsing Purged events."""
        instance = MagicMock()
        mock_get_instance.return_value = instance

        mock_ledger_api.api.eth.get_block_number.return_value = 10100

        mock_event_entry = MagicMock()
        mock_event_entry.args = {"memeToken": "0xPurged"}

        event_obj = MagicMock()
        event_obj.create_filter.return_value.get_all_entries.return_value = [
            mock_event_entry
        ]
        instance.events.Purged = event_obj

        result = MemeFactoryContract.get_events(
            mock_ledger_api, DUMMY_ADDRESS, "Purged", from_block=10000
        )

        events: List[Dict[str, Any]] = result["events"]  # type: ignore[assignment]
        assert len(events) == 1
        assert events[0]["token_address"] == "0xPurged"

    @patch.object(MemeFactoryContract, "get_instance")
    def test_unknown_event_returns_empty(
        self, mock_get_instance: MagicMock, mock_ledger_api: MagicMock
    ) -> None:
        """Test that an unrecognized event name returns empty dict."""
        instance = MagicMock()
        mock_get_instance.return_value = instance

        mock_ledger_api.api.eth.get_block_number.return_value = 10100

        event_obj = MagicMock()
        event_obj.create_filter.return_value.get_all_entries.return_value = []
        instance.events.Unknown = event_obj

        result = MemeFactoryContract.get_events(
            mock_ledger_api, DUMMY_ADDRESS, "Unknown", from_block=10000
        )

        assert not result

    @patch.object(MemeFactoryContract, "get_instance")
    def test_default_from_block(
        self, mock_get_instance: MagicMock, mock_ledger_api: MagicMock
    ) -> None:
        """Test that from_block defaults to ~48h ago when not provided."""
        instance = MagicMock()
        mock_get_instance.return_value = instance

        mock_ledger_api.api.eth.get_block_number.return_value = 100000

        event_obj = MagicMock()
        event_obj.create_filter.return_value.get_all_entries.return_value = []
        instance.events.Summoned = event_obj

        MemeFactoryContract.get_events(mock_ledger_api, DUMMY_ADDRESS, "Summoned")

        # from_block = 100000 - 86400 = 13600
        # The first batch should start at 13600
        first_call_args = event_obj.create_filter.call_args_list[0]
        assert first_call_args[1]["fromBlock"] == 13600

    @patch.object(MemeFactoryContract, "get_instance")
    def test_batch_processing_large_range(
        self, mock_get_instance: MagicMock, mock_ledger_api: MagicMock
    ) -> None:
        """Test that large block ranges are split into batches of 5000."""
        instance = MagicMock()
        mock_get_instance.return_value = instance

        mock_ledger_api.api.eth.get_block_number.return_value = 20000

        event_obj = MagicMock()
        event_obj.create_filter.return_value.get_all_entries.return_value = []
        instance.events.Summoned = event_obj

        MemeFactoryContract.get_events(
            mock_ledger_api, DUMMY_ADDRESS, "Summoned", from_block=0, to_block=12000
        )

        # Should have 3 batches: 0-5000, 5000-10000, 10000-12000
        assert event_obj.create_filter.call_count == 3

    @patch.object(MemeFactoryContract, "get_instance")
    def test_value_error_retry(
        self, mock_get_instance: MagicMock, mock_ledger_api: MagicMock
    ) -> None:
        """Test that ValueError during filtering triggers retry."""
        instance = MagicMock()
        mock_get_instance.return_value = instance

        mock_ledger_api.api.eth.get_block_number.return_value = 10100

        event_obj = MagicMock()
        # First call raises ValueError, second succeeds
        mock_filter = MagicMock()
        mock_filter.get_all_entries.side_effect = [
            ValueError("Filter does not exist"),
            [],
        ]
        event_obj.create_filter.return_value = mock_filter
        instance.events.Summoned = event_obj

        result = MemeFactoryContract.get_events(
            mock_ledger_api, DUMMY_ADDRESS, "Summoned", from_block=10000
        )

        assert result == {"events": [], "latest_block": 10099}

    @patch.object(MemeFactoryContract, "get_instance")
    def test_mismatched_abi_retry(
        self, mock_get_instance: MagicMock, mock_ledger_api: MagicMock
    ) -> None:
        """Test that MismatchedABI during filtering triggers retry."""
        instance = MagicMock()
        mock_get_instance.return_value = instance

        mock_ledger_api.api.eth.get_block_number.return_value = 10100

        event_obj = MagicMock()
        mock_filter = MagicMock()
        mock_filter.get_all_entries.side_effect = [
            web3_module.exceptions.MismatchedABI(),
            [],
        ]
        event_obj.create_filter.return_value = mock_filter
        instance.events.Summoned = event_obj

        result = MemeFactoryContract.get_events(
            mock_ledger_api, DUMMY_ADDRESS, "Summoned", from_block=10000
        )

        assert result == {"events": [], "latest_block": 10099}


class TestGetBurnableAmount:
    """Test get_burnable_amount."""

    @patch.object(MemeFactoryContract, "get_instance")
    def test_get_burnable_amount(
        self, mock_get_instance: MagicMock, mock_ledger_api: MagicMock
    ) -> None:
        """Test fetching burnable amount from contract."""
        instance = MagicMock()
        instance.functions.scheduledForAscendance.return_value.call.return_value = 42
        mock_get_instance.return_value = instance

        result = MemeFactoryContract.get_burnable_amount(mock_ledger_api, DUMMY_ADDRESS)

        assert result == {"burnable_amount": 42}


class TestGetCollectableAmount:
    """Test get_collectable_amount."""

    @patch.object(MemeFactoryContract, "get_instance")
    def test_get_collectable_amount(
        self, mock_get_instance: MagicMock, mock_ledger_api: MagicMock
    ) -> None:
        """Test fetching collectable amount for a wallet."""
        instance = MagicMock()
        instance.functions.memeHearters.return_value.call.return_value = 100
        mock_get_instance.return_value = instance

        result = MemeFactoryContract.get_collectable_amount(
            mock_ledger_api, DUMMY_ADDRESS, token_nonce=5, wallet_address="0xWallet"
        )

        instance.functions.memeHearters.assert_called_once_with(5, "0xWallet")
        assert result == {"collectable_amount": 100}


PACKAGE_DIR = Path(__file__).parent.parent


class TestABIConsistency:
    """Test that functions and events used in contract.py exist in the ABI."""

    @staticmethod
    def _get_abi_names() -> tuple:
        """Extract function and event names from ABI files."""
        functions: set = set()
        events: set = set()
        for abi_file in PACKAGE_DIR.glob("build/*.json"):
            with open(abi_file) as f:
                data = json.load(f)
            abi = data.get("abi", data)
            for entry in abi:
                if entry.get("type") == "function":
                    functions.add(entry["name"])
                elif entry.get("type") == "event":
                    events.add(entry["name"])
        return functions, events

    @staticmethod
    def _get_contract_references() -> tuple:
        """Extract function and event names referenced in contract.py."""
        source = (PACKAGE_DIR / "contract.py").read_text()
        function_patterns = [
            r"\.functions\.(\w+)",
            r"encode[_.]?[aA][bB][iI]\(\s*(?:abi_element_identifier\s*=\s*)?[\"'](\w+)[\"']",
            r"method_name\s*=\s*[\"'](\w+)[\"']",
        ]
        referenced_functions: set = set()
        for pattern in function_patterns:
            referenced_functions.update(re.findall(pattern, source))
        event_pattern = r"\.events\.(\w+)"
        referenced_events: set = set(re.findall(event_pattern, source))
        return referenced_functions, referenced_events

    def test_functions_present_in_abi(self) -> None:
        """All contract functions referenced in contract.py must exist in the ABI."""
        abi_functions, _ = self._get_abi_names()
        referenced_functions, _ = self._get_contract_references()
        missing = referenced_functions - abi_functions
        assert (
            not missing
        ), f"Functions used in contract.py but missing from ABI: {missing}"

    def test_events_present_in_abi(self) -> None:
        """All contract events referenced in contract.py must exist in the ABI."""
        _, abi_events = self._get_abi_names()
        _, referenced_events = self._get_contract_references()
        missing = referenced_events - abi_events
        assert (
            not missing
        ), f"Events used in contract.py but missing from ABI: {missing}"
