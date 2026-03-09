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

"""Tests for models.py."""

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from packages.dvilela.skills.memeooorr_abci.models import (
    AlternativeModelForTweets,
    RandomnessApi,
    SharedState,
)
from packages.dvilela.skills.memeooorr_abci.rounds import MemeooorrAbciApp
from packages.valory.skills.abstract_round_abci.models import ApiSpecs
from packages.valory.skills.abstract_round_abci.models import (
    BenchmarkTool as BaseBenchmarkTool,
)
from packages.valory.skills.abstract_round_abci.models import Requests as BaseRequests
from packages.valory.skills.abstract_round_abci.models import (
    SharedState as BaseSharedState,
)


class TestSharedState:
    """Tests for SharedState."""

    def test_is_subclass_of_base(self) -> None:
        """Test SharedState is a subclass of BaseSharedState."""
        assert issubclass(SharedState, BaseSharedState)

    def test_abci_app_cls(self) -> None:
        """Test that abci_app_cls is set to MemeooorrAbciApp."""
        assert SharedState.abci_app_cls is MemeooorrAbciApp


class TestModelAliases:
    """Tests for model aliases."""

    def test_requests_alias(self) -> None:
        """Test Requests alias."""
        from packages.dvilela.skills.memeooorr_abci.models import Requests

        assert Requests is BaseRequests

    def test_benchmark_tool_alias(self) -> None:
        """Test BenchmarkTool alias."""
        from packages.dvilela.skills.memeooorr_abci.models import BenchmarkTool

        assert BenchmarkTool is BaseBenchmarkTool

    def test_agent_db_client_alias(self) -> None:
        """Test AgentDBClient alias."""
        from packages.dvilela.skills.memeooorr_abci.models import AgentDBClient
        from packages.valory.skills.agent_db_abci.models import (
            AgentDBClient as BaseAgentDBClient,
        )

        assert AgentDBClient is BaseAgentDBClient

    def test_agents_fun_database_alias(self) -> None:
        """Test AgentsFunDatabase alias."""
        from packages.dvilela.skills.memeooorr_abci.models import AgentsFunDatabase
        from packages.valory.skills.agent_db_abci.models import (
            AgentsFunDatabase as BaseAgentsFunDatabase,
        )

        assert AgentsFunDatabase is BaseAgentsFunDatabase


class TestRandomnessApi:
    """Tests for RandomnessApi."""

    def test_is_subclass(self) -> None:
        """Test RandomnessApi is a subclass of ApiSpecs."""
        assert issubclass(RandomnessApi, ApiSpecs)


class TestAlternativeModelForTweets:
    """Tests for AlternativeModelForTweets."""

    def _make_data(self, **overrides: Any) -> Dict[str, Any]:
        """Create a data dict with sensible defaults."""
        data: Dict[str, Any] = {
            "url": "https://api.example.com",
            "api_key": "test_key",
            "model": "gpt-4",
            "max_tokens": 100,
            "top_p": 1,
            "top_k": 50,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "temperature": 0.7,
        }
        data.update(overrides)
        return data

    def test_from_dict_with_valid_key(self) -> None:
        """Test from_dict with a valid API key."""
        data = self._make_data()
        model = AlternativeModelForTweets.from_dict(data)
        assert model.use is True
        assert model.api_key == "test_key"
        assert model.url == "https://api.example.com"
        assert model.model == "gpt-4"
        assert model.max_tokens == 100
        assert model.top_p == 1
        assert model.top_k == 50
        assert model.presence_penalty == 0
        assert model.frequency_penalty == 0
        assert model.temperature == 0.7

    def test_from_dict_with_none_key(self) -> None:
        """Test from_dict with None API key."""
        data = self._make_data(api_key=None)
        model = AlternativeModelForTweets.from_dict(data)
        assert model.use is False
        assert model.api_key is None

    def test_from_dict_with_empty_key(self) -> None:
        """Test from_dict with empty string API key."""
        data = self._make_data(api_key="")
        model = AlternativeModelForTweets.from_dict(data)
        assert model.use is False
        assert model.api_key is None

    def test_from_dict_with_whitespace_key(self) -> None:
        """Test from_dict with whitespace-only API key."""
        data = self._make_data(api_key="   ")
        model = AlternativeModelForTweets.from_dict(data)
        assert model.use is False
        assert model.api_key is None

    def test_from_dict_with_placeholder_key(self) -> None:
        """Test from_dict with placeholder string API key."""
        data = self._make_data(api_key="${str:}")
        model = AlternativeModelForTweets.from_dict(data)
        assert model.use is False
        assert model.api_key is None

    def test_frozen(self) -> None:
        """Test that the dataclass is frozen."""
        data = self._make_data()
        model = AlternativeModelForTweets.from_dict(data)
        with pytest.raises(AttributeError):
            model.use = False  # type: ignore
