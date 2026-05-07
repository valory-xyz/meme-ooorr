# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2025-2026 Valory AG
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

"""Tests for models.py in agent_performance_summary_abci."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from packages.valory.skills.agent_performance_summary_abci.models import (
    AGENT_PERFORMANCE_SUMMARY_FILE,
    AgentPerformanceMetrics,
    AgentPerformanceSummary,
    AgentPerformanceSummaryParams,
    SharedState,
)


class TestAgentPerformanceMetrics:
    """Tests for AgentPerformanceMetrics dataclass."""

    def test_creation_with_defaults(self) -> None:
        """Test creation with default description."""
        m = AgentPerformanceMetrics(name="test", is_primary=True, value="42")
        assert m.name == "test"
        assert m.is_primary is True
        assert m.value == "42"
        assert m.description is None

    def test_creation_with_description(self) -> None:
        """Test creation with explicit description."""
        m = AgentPerformanceMetrics(
            name="test", is_primary=False, value="75%", description="some desc"
        )
        assert m.description == "some desc"


class TestAgentPerformanceSummary:
    """Tests for AgentPerformanceSummary dataclass."""

    def test_defaults(self) -> None:
        """Test default values."""
        s = AgentPerformanceSummary()
        assert s.timestamp is None
        assert s.metrics == []
        assert s.agent_behavior is None

    def test_from_dict(self) -> None:
        """Test from_dict with metrics."""
        data = {
            "timestamp": 123456,
            "metrics": [
                {"name": "m1", "is_primary": True, "value": "10"},
                {"name": "m2", "is_primary": False, "value": "20", "description": "d"},
            ],
            "agent_behavior": "some behavior",
        }
        s = AgentPerformanceSummary.from_dict(data)
        assert s.timestamp == 123456
        assert len(s.metrics) == 2
        assert isinstance(s.metrics[0], AgentPerformanceMetrics)
        assert s.metrics[0].name == "m1"
        assert s.agent_behavior == "some behavior"

    def test_from_dict_empty_metrics(self) -> None:
        """Test from_dict with no metrics key."""
        data = {"timestamp": 1, "agent_behavior": None}
        s = AgentPerformanceSummary.from_dict(data)
        assert s.metrics == []

    def test_to_json(self) -> None:
        """Test to_json serialization."""
        s = AgentPerformanceSummary(
            timestamp=100,
            metrics=[AgentPerformanceMetrics(name="x", is_primary=True, value="1")],
            agent_behavior="behave",
        )
        result = json.loads(s.to_json())
        assert result["timestamp"] == 100
        assert len(result["metrics"]) == 1
        assert result["metrics"][0]["name"] == "x"
        assert result["agent_behavior"] == "behave"


class TestAgentPerformanceSummaryParams:
    """Tests for AgentPerformanceSummaryParams."""

    @patch(
        "packages.valory.skills.agent_performance_summary_abci.models.BaseParams.__init__"
    )
    def test_init_success(self, mock_super_init: MagicMock, tmp_path: Path) -> None:
        """Test successful initialization with valid store path."""
        mock_super_init.return_value = None
        params = AgentPerformanceSummaryParams(
            store_path=str(tmp_path),
            is_agent_performance_summary_enabled=True,
            performance_summary_ttl=3600,
            name="test",
            skill_context=MagicMock(),
        )
        assert params.store_path == tmp_path
        assert params.is_agent_performance_summary_enabled is True
        assert params.performance_summary_ttl == 3600

    def test_init_invalid_store_path(self) -> None:
        """Test initialization with invalid store path raises ValueError."""
        with pytest.raises(ValueError, match="is not a directory or is not writable"):
            AgentPerformanceSummaryParams(
                store_path="/nonexistent/path/that/does/not/exist",
                is_agent_performance_summary_enabled=True,
                performance_summary_ttl=3600,
                name="test",
                skill_context=MagicMock(),
            )

    def test_init_empty_store_path(self) -> None:
        """Test initialization with empty store path raises ValueError."""
        with pytest.raises(ValueError, match="is not a directory or is not writable"):
            AgentPerformanceSummaryParams(
                store_path="",
                is_agent_performance_summary_enabled=False,
                performance_summary_ttl=100,
                name="test",
                skill_context=MagicMock(),
            )


class TestSharedState:
    """Tests for SharedState."""

    def _make_shared_state(self, tmp_path: Path) -> SharedState:
        """Create a SharedState with mocked context."""
        mock_params = MagicMock(spec=AgentPerformanceSummaryParams)
        mock_params.store_path = tmp_path

        mock_context = MagicMock()
        mock_context.params = mock_params

        mock_round_sequence = MagicMock()
        mock_round_sequence.last_round_transition_timestamp.timestamp.return_value = (
            1700000000.0
        )

        mock_state = MagicMock()
        mock_state.round_sequence = mock_round_sequence

        mock_context.state = mock_state

        state = SharedState.__new__(SharedState)
        state._context = mock_context
        return state

    @property
    def _mock_context(self) -> MagicMock:
        """Create mock context."""
        return MagicMock()

    def test_synced_timestamp(self, tmp_path: Path) -> None:
        """Test synced_timestamp returns int from round sequence."""
        state = self._make_shared_state(tmp_path)
        assert state.synced_timestamp == 1700000000

    def test_read_existing_performance_summary_success(self, tmp_path: Path) -> None:
        """Test reading a valid performance summary file."""
        state = self._make_shared_state(tmp_path)

        data = {
            "timestamp": 123,
            "metrics": [{"name": "m1", "is_primary": True, "value": "10"}],
            "agent_behavior": "some behavior",
        }
        file_path = tmp_path / AGENT_PERFORMANCE_SUMMARY_FILE
        file_path.write_text(json.dumps(data))

        result = state.read_existing_performance_summary()
        assert result.timestamp == 123
        assert len(result.metrics) == 1
        assert result.agent_behavior == "some behavior"

    def test_read_existing_performance_summary_file_not_found(
        self, tmp_path: Path
    ) -> None:
        """Test reading when file doesn't exist returns empty summary."""
        state = self._make_shared_state(tmp_path)
        result = state.read_existing_performance_summary()
        assert result.timestamp is None
        assert result.metrics == []

    def test_read_existing_performance_summary_invalid_json(
        self, tmp_path: Path
    ) -> None:
        """Test reading invalid JSON returns empty summary."""
        state = self._make_shared_state(tmp_path)

        file_path = tmp_path / AGENT_PERFORMANCE_SUMMARY_FILE
        file_path.write_text("not valid json{{{")

        result = state.read_existing_performance_summary()
        assert result.timestamp is None
        assert result.metrics == []

    def test_overwrite_performance_summary(self, tmp_path: Path) -> None:
        """Test writing a performance summary to file."""
        state = self._make_shared_state(tmp_path)

        summary = AgentPerformanceSummary(
            timestamp=999,
            metrics=[AgentPerformanceMetrics(name="m", is_primary=True, value="v")],
            agent_behavior="behave",
        )
        state.overwrite_performance_summary(summary)

        file_path = tmp_path / AGENT_PERFORMANCE_SUMMARY_FILE
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["timestamp"] == 999
        assert data["agent_behavior"] == "behave"

    def test_update_agent_behavior(self, tmp_path: Path) -> None:
        """Test update_agent_behavior updates existing data."""
        state = self._make_shared_state(tmp_path)

        # Write initial data
        initial = AgentPerformanceSummary(
            timestamp=100,
            metrics=[AgentPerformanceMetrics(name="m", is_primary=True, value="v")],
            agent_behavior="old",
        )
        state.overwrite_performance_summary(initial)

        # Update behavior
        state.update_agent_behavior("new behavior")

        # Verify
        file_path = tmp_path / AGENT_PERFORMANCE_SUMMARY_FILE
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["agent_behavior"] == "new behavior"
        assert data["timestamp"] == 1700000000
