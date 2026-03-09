# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023-2025 Valory AG
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

"""Tests for handlers.py."""

from packages.valory.skills.abstract_round_abci.handlers import (
    ABCIRoundHandler as BaseABCIRoundHandler,
)
from packages.valory.skills.abstract_round_abci.handlers import (
    ContractApiHandler as BaseContractApiHandler,
)
from packages.valory.skills.abstract_round_abci.handlers import (
    HttpHandler as BaseHttpHandler,
)
from packages.valory.skills.abstract_round_abci.handlers import (
    IpfsHandler as BaseIpfsHandler,
)
from packages.valory.skills.abstract_round_abci.handlers import (
    LedgerApiHandler as BaseLedgerApiHandler,
)
from packages.valory.skills.abstract_round_abci.handlers import (
    SigningHandler as BaseSigningHandler,
)
from packages.valory.skills.abstract_round_abci.handlers import (
    TendermintHandler as BaseTendermintHandler,
)
from packages.valory.skills.agent_db_abci.handlers import (
    ABCIHandler,
    ContractApiHandler,
    HttpHandler,
    IpfsHandler,
    LedgerApiHandler,
    SigningHandler,
    TendermintHandler,
)


class TestHandlerAliases:
    """Tests that handler aliases map to the correct base classes."""

    def test_abci_handler(self) -> None:
        """Test ABCIHandler is BaseABCIRoundHandler."""
        assert ABCIHandler is BaseABCIRoundHandler

    def test_http_handler(self) -> None:
        """Test HttpHandler is BaseHttpHandler."""
        assert HttpHandler is BaseHttpHandler

    def test_signing_handler(self) -> None:
        """Test SigningHandler is BaseSigningHandler."""
        assert SigningHandler is BaseSigningHandler

    def test_ledger_api_handler(self) -> None:
        """Test LedgerApiHandler is BaseLedgerApiHandler."""
        assert LedgerApiHandler is BaseLedgerApiHandler

    def test_contract_api_handler(self) -> None:
        """Test ContractApiHandler is BaseContractApiHandler."""
        assert ContractApiHandler is BaseContractApiHandler

    def test_tendermint_handler(self) -> None:
        """Test TendermintHandler is BaseTendermintHandler."""
        assert TendermintHandler is BaseTendermintHandler

    def test_ipfs_handler(self) -> None:
        """Test IpfsHandler is BaseIpfsHandler."""
        assert IpfsHandler is BaseIpfsHandler
