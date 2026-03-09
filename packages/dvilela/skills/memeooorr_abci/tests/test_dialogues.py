# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024 David Vilela Freire
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

"""Tests for dialogues.py."""

from aea.skills.base import Model

from packages.dvilela.protocols.kv_store.dialogues import (
    KvStoreDialogue as BaseKvStoreDialogue,
)
from packages.dvilela.protocols.kv_store.dialogues import (
    KvStoreDialogues as BaseKvStoreDialogues,
)
from packages.dvilela.skills.memeooorr_abci.dialogues import (
    AbciDialogue,
    AbciDialogues,
    ContractApiDialogue,
    ContractApiDialogues,
    HttpDialogue,
    HttpDialogues,
    IpfsDialogue,
    IpfsDialogues,
    KvStoreDialogue,
    KvStoreDialogues,
    LedgerApiDialogue,
    LedgerApiDialogues,
    SigningDialogue,
    SigningDialogues,
    SrrDialogue,
    SrrDialogues,
    TendermintDialogue,
    TendermintDialogues,
)
from packages.valory.protocols.srr.dialogues import SrrDialogue as BaseSrrDialogue
from packages.valory.protocols.srr.dialogues import SrrDialogues as BaseSrrDialogues
from packages.valory.skills.abstract_round_abci.dialogues import (
    AbciDialogue as BaseAbciDialogue,
)
from packages.valory.skills.abstract_round_abci.dialogues import (
    AbciDialogues as BaseAbciDialogues,
)
from packages.valory.skills.abstract_round_abci.dialogues import (
    ContractApiDialogue as BaseContractApiDialogue,
)
from packages.valory.skills.abstract_round_abci.dialogues import (
    ContractApiDialogues as BaseContractApiDialogues,
)
from packages.valory.skills.abstract_round_abci.dialogues import (
    HttpDialogue as BaseHttpDialogue,
)
from packages.valory.skills.abstract_round_abci.dialogues import (
    HttpDialogues as BaseHttpDialogues,
)
from packages.valory.skills.abstract_round_abci.dialogues import (
    IpfsDialogue as BaseIpfsDialogue,
)
from packages.valory.skills.abstract_round_abci.dialogues import (
    IpfsDialogues as BaseIpfsDialogues,
)
from packages.valory.skills.abstract_round_abci.dialogues import (
    LedgerApiDialogue as BaseLedgerApiDialogue,
)
from packages.valory.skills.abstract_round_abci.dialogues import (
    LedgerApiDialogues as BaseLedgerApiDialogues,
)
from packages.valory.skills.abstract_round_abci.dialogues import (
    SigningDialogue as BaseSigningDialogue,
)
from packages.valory.skills.abstract_round_abci.dialogues import (
    SigningDialogues as BaseSigningDialogues,
)
from packages.valory.skills.abstract_round_abci.dialogues import (
    TendermintDialogue as BaseTendermintDialogue,
)
from packages.valory.skills.abstract_round_abci.dialogues import (
    TendermintDialogues as BaseTendermintDialogues,
)


class TestDialogueAliases:
    """Tests that dialogue aliases map to the correct base classes."""

    def test_abci_dialogue(self) -> None:
        """Test AbciDialogue alias."""
        assert AbciDialogue is BaseAbciDialogue

    def test_abci_dialogues(self) -> None:
        """Test AbciDialogues alias."""
        assert AbciDialogues is BaseAbciDialogues

    def test_http_dialogue(self) -> None:
        """Test HttpDialogue alias."""
        assert HttpDialogue is BaseHttpDialogue

    def test_http_dialogues(self) -> None:
        """Test HttpDialogues alias."""
        assert HttpDialogues is BaseHttpDialogues

    def test_signing_dialogue(self) -> None:
        """Test SigningDialogue alias."""
        assert SigningDialogue is BaseSigningDialogue

    def test_signing_dialogues(self) -> None:
        """Test SigningDialogues alias."""
        assert SigningDialogues is BaseSigningDialogues

    def test_ledger_api_dialogue(self) -> None:
        """Test LedgerApiDialogue alias."""
        assert LedgerApiDialogue is BaseLedgerApiDialogue

    def test_ledger_api_dialogues(self) -> None:
        """Test LedgerApiDialogues alias."""
        assert LedgerApiDialogues is BaseLedgerApiDialogues

    def test_contract_api_dialogue(self) -> None:
        """Test ContractApiDialogue alias."""
        assert ContractApiDialogue is BaseContractApiDialogue

    def test_contract_api_dialogues(self) -> None:
        """Test ContractApiDialogues alias."""
        assert ContractApiDialogues is BaseContractApiDialogues

    def test_tendermint_dialogue(self) -> None:
        """Test TendermintDialogue alias."""
        assert TendermintDialogue is BaseTendermintDialogue

    def test_tendermint_dialogues(self) -> None:
        """Test TendermintDialogues alias."""
        assert TendermintDialogues is BaseTendermintDialogues

    def test_ipfs_dialogue(self) -> None:
        """Test IpfsDialogue alias."""
        assert IpfsDialogue is BaseIpfsDialogue

    def test_ipfs_dialogues(self) -> None:
        """Test IpfsDialogues alias."""
        assert IpfsDialogues is BaseIpfsDialogues

    def test_srr_dialogue(self) -> None:
        """Test SrrDialogue alias."""
        assert SrrDialogue is BaseSrrDialogue

    def test_kv_store_dialogue(self) -> None:
        """Test KvStoreDialogue alias."""
        assert KvStoreDialogue is BaseKvStoreDialogue


class TestSrrDialogues:
    """Tests for SrrDialogues class."""

    def test_is_subclass_of_model(self) -> None:
        """Test SrrDialogues is subclass of Model."""
        assert issubclass(SrrDialogues, Model)

    def test_is_subclass_of_base_srr_dialogues(self) -> None:
        """Test SrrDialogues is subclass of BaseSrrDialogues."""
        assert issubclass(SrrDialogues, BaseSrrDialogues)


class TestKvStoreDialogues:
    """Tests for KvStoreDialogues class."""

    def test_is_subclass_of_model(self) -> None:
        """Test KvStoreDialogues is subclass of Model."""
        assert issubclass(KvStoreDialogues, Model)

    def test_is_subclass_of_base_kv_store_dialogues(self) -> None:
        """Test KvStoreDialogues is subclass of BaseKvStoreDialogues."""
        assert issubclass(KvStoreDialogues, BaseKvStoreDialogues)
