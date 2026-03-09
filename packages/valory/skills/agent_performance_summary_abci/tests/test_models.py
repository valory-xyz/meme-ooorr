# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2025 Valory AG
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

"""Tests for the models module of the agent_performance_summary_abci skill."""

# pylint: disable=W0212,E0237

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from packages.valory.skills.agent_performance_summary_abci.models import (
    AGENT_PERFORMANCE_SUMMARY_FILE,
    AgentPerformanceMetrics,
    AgentPerformanceSummary,
    AgentPerformanceSummaryParams,
    SharedState,
)
from packages.valory.skills.agent_performance_summary_abci.rounds import (
    AgentPerformanceSummaryAbciApp,
)


class TestAgentPerformanceMetrics:
    """Tests for the AgentPerformanceMetrics dataclass."""

    def test_construction_without_description(self) -> None:
        """Test construction without optional description."""
        metric = AgentPerformanceMetrics(name="likes", is_primary=True, value="100")
        assert metric.name == "likes"
        assert metric.is_primary is True
        assert metric.value == "100"
        assert metric.description is None

    def test_construction_with_description(self) -> None:
        """Test construction with description."""
        metric = AgentPerformanceMetrics(
            name="impressions",
            is_primary=False,
            value="5000",
            description="Total impressions in the last 7 days.",
        )
        assert metric.name == "impressions"
        assert metric.is_primary is False
        assert metric.value == "5000"
        assert metric.description == "Total impressions in the last 7 days."

    def test_description_with_html(self) -> None:
        """Test that description can contain HTML tags."""
        desc = "Total <b>likes</b> received."
        metric = AgentPerformanceMetrics(
            name="likes", is_primary=True, value="42", description=desc
        )
        assert metric.description == desc


class TestAgentPerformanceSummary:
    """Tests for the AgentPerformanceSummary dataclass."""

    def test_default_construction(self) -> None:
        """Test default construction produces empty summary."""
        summary = AgentPerformanceSummary()
        assert summary.timestamp is None
        assert summary.metrics == []
        assert summary.agent_behavior is None

    def test_construction_with_values(self) -> None:
        """Test construction with explicit values."""
        metrics = [
            AgentPerformanceMetrics(name="likes", is_primary=True, value="10"),
        ]
        summary = AgentPerformanceSummary(
            timestamp=1234567890,
            metrics=metrics,
            agent_behavior="posting memes",
        )
        assert summary.timestamp == 1234567890
        assert len(summary.metrics) == 1
        assert summary.agent_behavior == "posting memes"

    def test_from_dict_with_metrics(self) -> None:
        """Test from_dict parses metrics correctly."""
        data = {
            "timestamp": 1000000,
            "metrics": [
                {"name": "likes", "is_primary": True, "value": "50"},
                {
                    "name": "impressions",
                    "is_primary": False,
                    "value": "2000",
                    "description": "Total impressions.",
                },
            ],
            "agent_behavior": "active",
        }
        summary = AgentPerformanceSummary.from_dict(data)
        assert summary.timestamp == 1000000
        assert len(summary.metrics) == 2
        assert isinstance(summary.metrics[0], AgentPerformanceMetrics)
        assert summary.metrics[0].name == "likes"
        assert summary.metrics[0].is_primary is True
        assert summary.metrics[1].description == "Total impressions."
        assert summary.agent_behavior == "active"

    def test_from_dict_without_metrics(self) -> None:
        """Test from_dict with no metrics key uses empty list."""
        data = {"timestamp": 999, "agent_behavior": None}
        summary = AgentPerformanceSummary.from_dict(data)
        assert summary.metrics == []

    def test_to_json(self) -> None:
        """Test to_json produces valid JSON."""
        metrics = [
            AgentPerformanceMetrics(
                name="likes", is_primary=True, value="10", description=None
            ),
        ]
        summary = AgentPerformanceSummary(
            timestamp=123, metrics=metrics, agent_behavior="idle"
        )
        result = summary.to_json()
        parsed = json.loads(result)
        assert parsed["timestamp"] == 123
        assert parsed["agent_behavior"] == "idle"
        assert len(parsed["metrics"]) == 1
        assert parsed["metrics"][0]["name"] == "likes"

    def test_to_json_round_trip(self) -> None:
        """Test to_json -> from_dict round-trip."""
        original = AgentPerformanceSummary(
            timestamp=555,
            metrics=[
                AgentPerformanceMetrics(
                    name="m1", is_primary=True, value="v1", description="d1"
                ),
            ],
            agent_behavior="behavior",
        )
        json_str = original.to_json()
        reconstructed = AgentPerformanceSummary.from_dict(json.loads(json_str))
        assert reconstructed.timestamp == original.timestamp
        assert reconstructed.agent_behavior == original.agent_behavior
        assert len(reconstructed.metrics) == len(original.metrics)
        assert reconstructed.metrics[0].name == original.metrics[0].name
        assert reconstructed.metrics[0].value == original.metrics[0].value


class TestAgentPerformanceSummaryParams:
    """Tests for the AgentPerformanceSummaryParams class."""

    def test_get_store_path_valid(self, tmp_path: Path) -> None:
        """Test get_store_path with a valid directory."""
        params = AgentPerformanceSummaryParams.__new__(AgentPerformanceSummaryParams)
        result = params.get_store_path({"store_path": str(tmp_path)})
        assert result == tmp_path

    def test_get_store_path_empty_string(self) -> None:
        """Test get_store_path with empty string raises ValueError."""
        params = AgentPerformanceSummaryParams.__new__(AgentPerformanceSummaryParams)
        with pytest.raises(ValueError, match="is not a directory or is not writable"):
            params.get_store_path({"store_path": ""})

    def test_get_store_path_nonexistent(self) -> None:
        """Test get_store_path with nonexistent path raises ValueError."""
        params = AgentPerformanceSummaryParams.__new__(AgentPerformanceSummaryParams)
        with pytest.raises(ValueError, match="is not a directory or is not writable"):
            params.get_store_path({"store_path": "/nonexistent/path/xyz"})

    def test_get_store_path_not_writable(self, tmp_path: Path) -> None:
        """Test get_store_path with non-writable directory raises ValueError."""
        read_only_dir = tmp_path / "readonly"
        read_only_dir.mkdir()
        os.chmod(read_only_dir, 0o444)
        params = AgentPerformanceSummaryParams.__new__(AgentPerformanceSummaryParams)
        try:
            with pytest.raises(
                ValueError, match="is not a directory or is not writable"
            ):
                params.get_store_path({"store_path": str(read_only_dir)})
        finally:
            os.chmod(read_only_dir, 0o755)  # nosec B103

    def test_get_store_path_missing_key(self) -> None:
        """Test get_store_path with missing key uses empty string default."""
        params = AgentPerformanceSummaryParams.__new__(AgentPerformanceSummaryParams)
        with pytest.raises(ValueError):
            params.get_store_path({})


class TestSharedState:
    """Tests for the SharedState class."""

    def test_abci_app_cls(self) -> None:
        """Test the abci_app_cls attribute."""
        assert SharedState.abci_app_cls is AgentPerformanceSummaryAbciApp

    def test_params_property(self) -> None:
        """Test the params property returns context.params cast."""
        state = SharedState.__new__(SharedState)
        mock_params = MagicMock(spec=AgentPerformanceSummaryParams)
        state._context = MagicMock()
        state.context.params = mock_params

        result = state.params
        assert result is mock_params

    def test_synced_timestamp(self) -> None:
        """Test synced_timestamp reads from round_sequence."""
        state = SharedState.__new__(SharedState)
        mock_ts = MagicMock()
        mock_ts.timestamp.return_value = 1234567890.5
        state._context = MagicMock()
        state.context.state.round_sequence.last_round_transition_timestamp = mock_ts

        result = state.synced_timestamp
        assert result == 1234567890

    def test_read_existing_performance_summary_valid_file(self, tmp_path: Path) -> None:
        """Test reading a valid performance summary file."""
        state = SharedState.__new__(SharedState)
        mock_params = MagicMock()
        mock_params.store_path = tmp_path
        state._context = MagicMock()
        state.context.params = mock_params

        file_path = tmp_path / AGENT_PERFORMANCE_SUMMARY_FILE
        data = {
            "timestamp": 999,
            "metrics": [{"name": "likes", "is_primary": True, "value": "50"}],
            "agent_behavior": "test",
        }
        file_path.write_text(json.dumps(data))

        result = state.read_existing_performance_summary()
        assert result.timestamp == 999
        assert len(result.metrics) == 1
        assert result.metrics[0].name == "likes"
        assert result.agent_behavior == "test"

    def test_read_existing_performance_summary_missing_file(
        self, tmp_path: Path
    ) -> None:
        """Test reading when file does not exist returns empty summary."""
        state = SharedState.__new__(SharedState)
        mock_params = MagicMock()
        mock_params.store_path = tmp_path
        state._context = MagicMock()
        state.context.params = mock_params

        result = state.read_existing_performance_summary()
        assert result.timestamp is None
        assert result.metrics == []
        assert result.agent_behavior is None

    def test_read_existing_performance_summary_invalid_json(
        self, tmp_path: Path
    ) -> None:
        """Test reading invalid JSON returns empty summary."""
        state = SharedState.__new__(SharedState)
        mock_params = MagicMock()
        mock_params.store_path = tmp_path
        state._context = MagicMock()
        state.context.params = mock_params

        file_path = tmp_path / AGENT_PERFORMANCE_SUMMARY_FILE
        file_path.write_text("not valid json{{{")

        result = state.read_existing_performance_summary()
        assert result.timestamp is None
        assert result.metrics == []

    def test_overwrite_performance_summary(self, tmp_path: Path) -> None:
        """Test writing a performance summary to file."""
        state = SharedState.__new__(SharedState)
        mock_params = MagicMock()
        mock_params.store_path = tmp_path
        state._context = MagicMock()
        state.context.params = mock_params

        summary = AgentPerformanceSummary(
            timestamp=12345,
            metrics=[
                AgentPerformanceMetrics(
                    name="likes", is_primary=True, value="100", description=None
                ),
            ],
            agent_behavior="posting",
        )
        state.overwrite_performance_summary(summary)

        file_path = tmp_path / AGENT_PERFORMANCE_SUMMARY_FILE
        assert file_path.exists()
        written = json.loads(file_path.read_text())
        assert written["timestamp"] == 12345
        assert written["metrics"][0]["name"] == "likes"
        assert written["agent_behavior"] == "posting"

    def test_update_agent_behavior(self, tmp_path: Path) -> None:
        """Test update_agent_behavior reads, updates, and writes back."""
        state = SharedState.__new__(SharedState)
        mock_params = MagicMock()
        mock_params.store_path = tmp_path
        state._context = MagicMock()
        state.context.params = mock_params

        # Write initial data
        file_path = tmp_path / AGENT_PERFORMANCE_SUMMARY_FILE
        file_path.write_text(
            json.dumps(
                {
                    "timestamp": 100,
                    "metrics": [
                        {
                            "name": "likes",
                            "is_primary": True,
                            "value": "5",
                            "description": None,
                        }
                    ],
                    "agent_behavior": "old behavior",
                }
            )
        )

        # Mock synced_timestamp
        mock_ts = MagicMock()
        mock_ts.timestamp.return_value = 200.0
        state.context.state.round_sequence.last_round_transition_timestamp = mock_ts

        state.update_agent_behavior("new behavior")

        # Read back and verify
        written = json.loads(file_path.read_text())
        assert written["agent_behavior"] == "new behavior"
        assert written["timestamp"] == 200
        # Metrics should be preserved
        assert len(written["metrics"]) == 1
        assert written["metrics"][0]["name"] == "likes"
