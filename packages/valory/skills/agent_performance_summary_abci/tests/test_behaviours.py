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

"""Tests for the behaviours module of the agent_performance_summary_abci skill."""

import pytest

from packages.valory.skills.abstract_round_abci.behaviours import (
    AbstractRoundBehaviour,
    BaseBehaviour,
)
from packages.valory.skills.agent_performance_summary_abci.behaviours import (
    FETCH_FOR_LAST_DAYS,
    IMPRESSIONS_METRIC_NAME,
    LIKES_METRIC_NAME,
    NA,
    AgentPerformanceSummaryRoundBehaviour,
    FetchPerformanceSummaryBehaviour,
    extract_metric_by_name,
)
from packages.valory.skills.agent_performance_summary_abci.models import (
    AgentPerformanceMetrics,
)
from packages.valory.skills.agent_performance_summary_abci.rounds import (
    AgentPerformanceSummaryAbciApp,
    FetchPerformanceDataRound,
)


class TestExtractMetricByName:
    """Tests for the extract_metric_by_name utility function."""

    def test_found(self) -> None:
        """Test extracting a metric that exists."""
        metrics = [
            AgentPerformanceMetrics(name="likes", is_primary=True, value="42"),
            AgentPerformanceMetrics(
                name="impressions", is_primary=False, value="1000"
            ),
        ]
        assert extract_metric_by_name(metrics, "likes") == "42"
        assert extract_metric_by_name(metrics, "impressions") == "1000"

    def test_not_found(self) -> None:
        """Test extracting a metric that does not exist returns None."""
        metrics = [
            AgentPerformanceMetrics(name="likes", is_primary=True, value="42"),
        ]
        assert extract_metric_by_name(metrics, "nonexistent") is None

    def test_empty_list(self) -> None:
        """Test extracting from an empty list returns None."""
        assert extract_metric_by_name([], "likes") is None

    def test_returns_first_match(self) -> None:
        """Test that the first matching metric is returned when duplicates exist."""
        metrics = [
            AgentPerformanceMetrics(name="likes", is_primary=True, value="10"),
            AgentPerformanceMetrics(name="likes", is_primary=False, value="20"),
        ]
        assert extract_metric_by_name(metrics, "likes") == "10"


class TestFetchPerformanceSummaryBehaviour:
    """Tests for the FetchPerformanceSummaryBehaviour class."""

    def test_matching_round(self) -> None:
        """Test matching_round attribute."""
        assert (
            FetchPerformanceSummaryBehaviour.matching_round
            is FetchPerformanceDataRound
        )


class TestAgentPerformanceSummaryRoundBehaviour:
    """Tests for the AgentPerformanceSummaryRoundBehaviour class."""

    def test_is_abstract_round_behaviour(self) -> None:
        """Test that it is an AbstractRoundBehaviour subclass."""
        assert issubclass(
            AgentPerformanceSummaryRoundBehaviour, AbstractRoundBehaviour
        )

    def test_initial_behaviour_cls(self) -> None:
        """Test initial_behaviour_cls attribute."""
        assert (
            AgentPerformanceSummaryRoundBehaviour.initial_behaviour_cls
            is FetchPerformanceSummaryBehaviour
        )

    def test_abci_app_cls(self) -> None:
        """Test abci_app_cls attribute."""
        assert (
            AgentPerformanceSummaryRoundBehaviour.abci_app_cls
            is AgentPerformanceSummaryAbciApp
        )

    def test_behaviours_set(self) -> None:
        """Test behaviours set contains the expected behaviour."""
        assert AgentPerformanceSummaryRoundBehaviour.behaviours == {
            FetchPerformanceSummaryBehaviour,
        }


class TestConstants:
    """Tests for module-level constants."""

    def test_fetch_for_last_days(self) -> None:
        """Test FETCH_FOR_LAST_DAYS constant."""
        assert FETCH_FOR_LAST_DAYS == 7

    def test_na_constant(self) -> None:
        """Test NA constant."""
        assert NA == "N/A"

    def test_likes_metric_name(self) -> None:
        """Test LIKES_METRIC_NAME constant."""
        assert LIKES_METRIC_NAME == "Weekly Likes"

    def test_impressions_metric_name(self) -> None:
        """Test IMPRESSIONS_METRIC_NAME constant."""
        assert IMPRESSIONS_METRIC_NAME == "Weekly Impressions"
