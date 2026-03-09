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

"""Tests for the ServiceRegistry contract."""

# pylint: disable=redefined-outer-name

from unittest.mock import MagicMock, patch

import pytest

from packages.dvilela.contracts.service_registry.contract import (
    PUBLIC_ID,
    ServiceRegistryContract,
)

DUMMY_ADDRESS = "0x" + "1" * 40


@pytest.fixture
def mock_ledger_api() -> MagicMock:
    """Return a mocked EthereumApi."""
    return MagicMock()


class TestServiceRegistryContractMeta:
    """Test contract metadata."""

    def test_public_id(self) -> None:
        """Test public ID is set correctly."""
        assert str(PUBLIC_ID) == "dvilela/service_registry:0.1.0"

    def test_contract_id(self) -> None:
        """Test contract_id matches PUBLIC_ID."""
        assert ServiceRegistryContract.contract_id == PUBLIC_ID


class TestGetServicesData:
    """Test get_services_data method."""

    @patch.object(ServiceRegistryContract, "get_instance")
    def test_no_services(
        self, mock_get_instance: MagicMock, mock_ledger_api: MagicMock
    ) -> None:
        """Test with zero registered services."""
        instance = MagicMock()
        instance.functions.totalSupply.return_value.call.return_value = 0
        mock_get_instance.return_value = instance

        result = ServiceRegistryContract.get_services_data(
            mock_ledger_api, DUMMY_ADDRESS
        )

        assert result == {"services_data": []}

    @patch.object(ServiceRegistryContract, "get_instance")
    def test_single_service(
        self, mock_get_instance: MagicMock, mock_ledger_api: MagicMock
    ) -> None:
        """Test with one registered service, verifying all fields are mapped."""
        instance = MagicMock()
        instance.functions.totalSupply.return_value.call.return_value = 1

        mock_hash = MagicMock()
        mock_hash.to_0x_hex.return_value = "abcdef"

        service_data = [
            1000,  # security_deposit
            "0xMultisig",  # multisig_address
            mock_hash,  # ipfs_hash
            3,  # threshold
            4,  # max_num_agent_instances
            2,  # num_agent_instances
            1,  # state
        ]
        instance.functions.mapServices.return_value.call.return_value = service_data
        mock_get_instance.return_value = instance

        result = ServiceRegistryContract.get_services_data(
            mock_ledger_api, DUMMY_ADDRESS
        )

        assert len(result["services_data"]) == 1
        svc = result["services_data"][0]
        assert svc["security_deposit"] == 1000
        assert svc["multisig_address"] == "0xMultisig"
        assert svc["ipfs_hash"] == "f01701220abcdef"
        assert svc["threshold"] == 3
        assert svc["max_num_agent_instances"] == 4
        assert svc["num_agent_instances"] == 2
        assert svc["state"] == 1

    @patch.object(ServiceRegistryContract, "get_instance")
    def test_multiple_services(
        self, mock_get_instance: MagicMock, mock_ledger_api: MagicMock
    ) -> None:
        """Test with multiple registered services iterates 1..n correctly."""
        instance = MagicMock()
        instance.functions.totalSupply.return_value.call.return_value = 3

        mock_hash = MagicMock()
        mock_hash.to_0x_hex.return_value = "aabb"

        service_data = [0, "0x0", mock_hash, 1, 1, 1, 0]
        instance.functions.mapServices.return_value.call.return_value = service_data
        mock_get_instance.return_value = instance

        result = ServiceRegistryContract.get_services_data(
            mock_ledger_api, DUMMY_ADDRESS
        )

        assert len(result["services_data"]) == 3
        # Verify mapServices was called with 1, 2, 3 (1-indexed)
        calls = instance.functions.mapServices.call_args_list
        assert [c[0][0] for c in calls] == [1, 2, 3]
