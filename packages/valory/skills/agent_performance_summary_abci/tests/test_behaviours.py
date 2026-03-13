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

"""Tests for behaviours.py in agent_performance_summary_abci."""

from typing import Any, Generator, List, Optional
from unittest.mock import MagicMock, patch

from packages.valory.skills.agent_performance_summary_abci.behaviours import (
    AgentPerformanceSummaryRoundBehaviour,
    FetchPerformanceSummaryBehaviour,
    IMPRESSIONS_METRIC_NAME,
    LAST_METRIC_FETCH_TIMESTAMP_KEY,
    LIKES_METRIC_NAME,
    NA,
    extract_metric_by_name,
)
from packages.valory.skills.agent_performance_summary_abci.models import (
    AgentPerformanceMetrics,
    AgentPerformanceSummary,
    AgentPerformanceSummaryParams,
    SharedState,
)
from packages.valory.skills.agent_performance_summary_abci.payloads import (
    FetchPerformanceDataPayload,
)
from packages.valory.skills.agent_performance_summary_abci.rounds import (
    FetchPerformanceDataRound,
)


def exhaust_generator(gen: Generator, values: Optional[List] = None) -> Any:
    """Exhaust a generator, sending values from list and returning the result."""
    values = values or []
    idx = 0
    try:
        next(gen)
        while True:
            send_val = values[idx] if idx < len(values) else None
            idx += 1
            gen.send(send_val)
    except StopIteration as e:
        return e.value


class TestExtractMetricByName:
    """Tests for extract_metric_by_name function."""

    def test_found(self) -> None:
        """Test extracting a metric that exists."""
        metrics = [
            AgentPerformanceMetrics(name="a", is_primary=True, value="1"),
            AgentPerformanceMetrics(name="b", is_primary=False, value="2"),
        ]
        assert extract_metric_by_name(metrics, "b") == "2"

    def test_not_found(self) -> None:
        """Test extracting a metric that doesn't exist."""
        metrics = [
            AgentPerformanceMetrics(name="a", is_primary=True, value="1"),
        ]
        assert extract_metric_by_name(metrics, "missing") is None

    def test_empty_list(self) -> None:
        """Test with empty metrics list."""
        assert extract_metric_by_name([], "any") is None


def _make_behaviour() -> FetchPerformanceSummaryBehaviour:
    """Create a FetchPerformanceSummaryBehaviour with fully mocked context."""
    mock_context = MagicMock()
    mock_context.agent_address = "test_agent_address"
    mock_context.logger = MagicMock()

    mock_params = MagicMock(spec=AgentPerformanceSummaryParams)
    mock_params.is_agent_performance_summary_enabled = True
    mock_params.performance_summary_ttl = 3600
    mock_context.params = mock_params

    mock_shared_state = MagicMock(spec=SharedState)
    mock_shared_state.synced_timestamp = 1700000000
    mock_shared_state.twitter_id = "12345"
    mock_context.state = mock_shared_state

    mock_benchmark = MagicMock()
    mock_benchmark.measure.return_value.local.return_value = MagicMock(
        __enter__=MagicMock(), __exit__=MagicMock(return_value=False)
    )
    mock_benchmark.measure.return_value.consensus.return_value = MagicMock(
        __enter__=MagicMock(), __exit__=MagicMock(return_value=False)
    )
    mock_context.benchmark_tool = mock_benchmark

    behaviour = FetchPerformanceSummaryBehaviour.__new__(
        FetchPerformanceSummaryBehaviour
    )
    behaviour._context = mock_context
    behaviour._agent_performance_summary = None
    return behaviour


class TestFetchPerformanceSummaryBehaviourInit:
    """Tests for FetchPerformanceSummaryBehaviour initialization."""

    def test_matching_round(self) -> None:
        """Test matching_round class attribute."""
        assert (
            FetchPerformanceSummaryBehaviour.matching_round == FetchPerformanceDataRound
        )

    @patch.object(
        FetchPerformanceSummaryBehaviour.__bases__[0],
        "__init__",
        return_value=None,
    )
    def test_init_calls_super_and_sets_summary(
        self, mock_super_init: MagicMock
    ) -> None:
        """Test __init__ calls super and sets _agent_performance_summary to None."""
        behaviour = FetchPerformanceSummaryBehaviour(
            name="test", skill_context=MagicMock()
        )
        assert behaviour._agent_performance_summary is None
        mock_super_init.assert_called_once()

    def test_shared_state_property(self) -> None:
        """Test shared_state property returns cast context.state."""
        behaviour = _make_behaviour()
        result = behaviour.shared_state
        assert result is behaviour.context.state

    def test_params_property(self) -> None:
        """Test params property returns cast context.params."""
        behaviour = _make_behaviour()
        result = behaviour.params
        assert result is behaviour.context.params


class TestGetTotalLikesAndRetweets:
    """Tests for _get_total_likes_and_retweets."""

    def test_success(self) -> None:
        """Test successful fetch of likes and retweets."""
        behaviour = _make_behaviour()

        tweets = [
            {"like_count": 10, "impression_count": 100},
            {"like_count": 5, "impression_count": 50},
        ]

        def mock_init_twitter(*args: Any, **kwargs: Any) -> Generator:
            yield

        def mock_call_tweepy(*args: Any, **kwargs: Any) -> Generator:
            yield
            return tweets

        behaviour.init_own_twitter_details = mock_init_twitter
        behaviour._call_tweepy = mock_call_tweepy

        gen = behaviour._get_total_likes_and_retweets(since_timestamp=100)
        result = exhaust_generator(gen)
        assert result == (15, 150)

    def test_error_response(self) -> None:
        """Test error response from tweepy returns None, None."""
        behaviour = _make_behaviour()

        def mock_init_twitter(*args: Any, **kwargs: Any) -> Generator:
            yield

        def mock_call_tweepy(*args: Any, **kwargs: Any) -> Generator:
            yield
            return {"error": "rate limited"}

        behaviour.init_own_twitter_details = mock_init_twitter
        behaviour._call_tweepy = mock_call_tweepy

        gen = behaviour._get_total_likes_and_retweets(since_timestamp=100)
        result = exhaust_generator(gen)
        assert result == (None, None)

    def test_empty_tweets(self) -> None:
        """Test empty tweets list returns zero counts."""
        behaviour = _make_behaviour()

        def mock_init_twitter(*args: Any, **kwargs: Any) -> Generator:
            yield

        def mock_call_tweepy(*args: Any, **kwargs: Any) -> Generator:
            yield
            return []

        behaviour.init_own_twitter_details = mock_init_twitter
        behaviour._call_tweepy = mock_call_tweepy

        gen = behaviour._get_total_likes_and_retweets(since_timestamp=100)
        result = exhaust_generator(gen)
        assert result == (0, 0)

    def test_tweets_missing_fields(self) -> None:
        """Test tweets with missing count fields default to 0."""
        behaviour = _make_behaviour()

        tweets = [{"like_count": 3}, {"impression_count": 7}, {}]

        def mock_init_twitter(*args: Any, **kwargs: Any) -> Generator:
            yield

        def mock_call_tweepy(*args: Any, **kwargs: Any) -> Generator:
            yield
            return tweets

        behaviour.init_own_twitter_details = mock_init_twitter
        behaviour._call_tweepy = mock_call_tweepy

        gen = behaviour._get_total_likes_and_retweets(since_timestamp=100)
        result = exhaust_generator(gen)
        assert result == (3, 7)


class TestShouldFetchMetricsAgain:
    """Tests for should_fetch_metrics_again."""

    def test_no_metrics_returns_true(self) -> None:
        """Test returns True when existing data has no metrics."""
        behaviour = _make_behaviour()
        behaviour.shared_state.read_existing_performance_summary.return_value = (
            AgentPerformanceSummary(metrics=[])
        )

        gen = behaviour.should_fetch_metrics_again()
        result = exhaust_generator(gen)
        assert result is True

    def test_na_metric_returns_true(self) -> None:
        """Test returns True when existing data has N/A metrics."""
        behaviour = _make_behaviour()
        behaviour.shared_state.read_existing_performance_summary.return_value = (
            AgentPerformanceSummary(
                metrics=[
                    AgentPerformanceMetrics(name="m1", is_primary=True, value=NA),
                ]
            )
        )

        gen = behaviour.should_fetch_metrics_again()
        result = exhaust_generator(gen)
        assert result is True

    def test_no_last_fetch_timestamp_returns_true(self) -> None:
        """Test returns True when no previous fetch timestamp."""
        behaviour = _make_behaviour()
        behaviour.shared_state.read_existing_performance_summary.return_value = (
            AgentPerformanceSummary(
                metrics=[
                    AgentPerformanceMetrics(name="m1", is_primary=True, value="10"),
                ]
            )
        )

        def mock_read_json_from_kv(*args: Any, **kwargs: Any) -> Generator:
            yield

        behaviour._read_json_from_kv = mock_read_json_from_kv

        gen = behaviour.should_fetch_metrics_again()
        result = exhaust_generator(gen)
        assert result is True

    def test_ttl_not_expired_returns_false(self) -> None:
        """Test returns False when TTL has not expired."""
        behaviour = _make_behaviour()
        behaviour.shared_state.read_existing_performance_summary.return_value = (
            AgentPerformanceSummary(
                metrics=[
                    AgentPerformanceMetrics(name="m1", is_primary=True, value="10"),
                ]
            )
        )

        # last_fetch_timestamp + ttl > synced_timestamp => skip
        # synced_timestamp = 1700000000, ttl = 3600
        # last_fetch = 1700000000 => 1700000000 + 3600 = 1700003600 > 1700000000 => skip
        def mock_read_json_from_kv(*args: Any, **kwargs: Any) -> Generator:
            yield
            return 1700000000

        behaviour._read_json_from_kv = mock_read_json_from_kv

        gen = behaviour.should_fetch_metrics_again()
        result = exhaust_generator(gen)
        assert result is False

    def test_ttl_expired_returns_true(self) -> None:
        """Test returns True when TTL has expired."""
        behaviour = _make_behaviour()
        behaviour.shared_state.read_existing_performance_summary.return_value = (
            AgentPerformanceSummary(
                metrics=[
                    AgentPerformanceMetrics(name="m1", is_primary=True, value="10"),
                ]
            )
        )

        # last_fetch_timestamp + ttl <= synced_timestamp => fetch again
        # synced_timestamp = 1700000000, ttl = 3600
        # last_fetch = 1699990000 => 1699990000 + 3600 = 1699993600 <= 1700000000 => fetch
        def mock_read_json_from_kv(*args: Any, **kwargs: Any) -> Generator:
            yield
            return 1699990000

        behaviour._read_json_from_kv = mock_read_json_from_kv

        gen = behaviour.should_fetch_metrics_again()
        result = exhaust_generator(gen)
        assert result is True


class TestFetchAgentPerformanceSummary:
    """Tests for _fetch_agent_performance_summary."""

    def test_success(self) -> None:
        """Test successful fetch stores summary with metric values."""
        behaviour = _make_behaviour()

        def mock_get_likes(*args: Any, **kwargs: Any) -> Generator:
            yield
            return (15, 200)

        def mock_write_kv(*args: Any, **kwargs: Any) -> Generator:
            yield
            return True

        behaviour._get_total_likes_and_retweets = mock_get_likes
        behaviour.write_kv = mock_write_kv

        gen = behaviour._fetch_agent_performance_summary()
        exhaust_generator(gen)

        summary = behaviour._agent_performance_summary
        assert summary is not None
        assert summary.timestamp == 1700000000
        assert len(summary.metrics) == 2
        assert summary.metrics[0].name == IMPRESSIONS_METRIC_NAME
        assert summary.metrics[0].value == "200"
        assert summary.metrics[0].is_primary is True
        assert summary.metrics[1].name == LIKES_METRIC_NAME
        assert summary.metrics[1].value == "15"
        assert summary.metrics[1].is_primary is False
        assert summary.agent_behavior is None

    def test_fetch_failure_uses_existing_data(self) -> None:
        """Test when likes/impressions are None, falls back to existing data."""
        behaviour = _make_behaviour()

        def mock_get_likes(*args: Any, **kwargs: Any) -> Generator:
            yield
            return (None, None)

        behaviour._get_total_likes_and_retweets = mock_get_likes

        existing = AgentPerformanceSummary(
            metrics=[
                AgentPerformanceMetrics(
                    name=LIKES_METRIC_NAME, is_primary=False, value="5"
                ),
                AgentPerformanceMetrics(
                    name=IMPRESSIONS_METRIC_NAME, is_primary=True, value="50"
                ),
            ]
        )
        behaviour.shared_state.read_existing_performance_summary.return_value = existing

        gen = behaviour._fetch_agent_performance_summary()
        exhaust_generator(gen)

        summary = behaviour._agent_performance_summary
        assert summary is not None
        # Falls back to existing values
        assert summary.metrics[0].value == "50"  # impressions
        assert summary.metrics[1].value == "5"  # likes

    def test_fetch_failure_no_existing_data(self) -> None:
        """Test when likes/impressions are None and no existing data."""
        behaviour = _make_behaviour()

        def mock_get_likes(*args: Any, **kwargs: Any) -> Generator:
            yield
            return (None, None)

        behaviour._get_total_likes_and_retweets = mock_get_likes

        existing = AgentPerformanceSummary(metrics=[])
        behaviour.shared_state.read_existing_performance_summary.return_value = existing

        gen = behaviour._fetch_agent_performance_summary()
        exhaust_generator(gen)

        summary = behaviour._agent_performance_summary
        assert summary is not None
        # extract_metric_by_name returns None for missing metrics
        assert summary.metrics[0].value == NA  # impressions not found => NA
        assert summary.metrics[1].value == NA  # likes not found => NA

    def test_success_writes_timestamp_to_kv(self) -> None:
        """Test that successful fetch writes timestamp to KV store."""
        behaviour = _make_behaviour()

        write_kv_calls = []

        def mock_get_likes(*args: Any, **kwargs: Any) -> Generator:
            yield
            return (10, 100)

        def mock_write_kv(data: Any, **kwargs: Any) -> Generator:
            write_kv_calls.append(data)
            yield
            return True

        behaviour._get_total_likes_and_retweets = mock_get_likes
        behaviour.write_kv = mock_write_kv

        gen = behaviour._fetch_agent_performance_summary()
        exhaust_generator(gen)

        assert len(write_kv_calls) == 1
        assert LAST_METRIC_FETCH_TIMESTAMP_KEY in write_kv_calls[0]
        assert write_kv_calls[0][LAST_METRIC_FETCH_TIMESTAMP_KEY] == 1700000000


class TestSaveAgentPerformanceSummary:  # pylint: disable=too-few-public-methods
    """Tests for _save_agent_performance_summary."""

    def test_preserves_existing_agent_behavior(self) -> None:
        """Test that save preserves existing agent_behavior."""
        behaviour = _make_behaviour()

        existing = AgentPerformanceSummary(
            agent_behavior="existing behavior",
            metrics=[],
        )
        behaviour.shared_state.read_existing_performance_summary.return_value = existing

        new_summary = AgentPerformanceSummary(
            timestamp=999,
            metrics=[AgentPerformanceMetrics(name="m", is_primary=True, value="v")],
            agent_behavior=None,
        )
        behaviour._save_agent_performance_summary(new_summary)

        assert new_summary.agent_behavior == "existing behavior"
        behaviour.shared_state.overwrite_performance_summary.assert_called_once_with(
            new_summary
        )


class TestAsyncAct:
    """Tests for async_act."""

    def test_disabled(self) -> None:
        """Test when performance summary is disabled."""
        behaviour = _make_behaviour()
        behaviour.params.is_agent_performance_summary_enabled = False

        payloads_sent = []

        def mock_finish(payload: Any) -> Generator:
            payloads_sent.append(payload)
            yield

        behaviour.finish_behaviour = mock_finish

        gen = behaviour.async_act()
        exhaust_generator(gen)

        assert len(payloads_sent) == 1
        assert payloads_sent[0].vote is False
        behaviour.context.logger.info.assert_called()

    def test_should_not_fetch(self) -> None:
        """Test when should_fetch_metrics_again returns False."""
        behaviour = _make_behaviour()
        behaviour.params.is_agent_performance_summary_enabled = True

        def mock_should_fetch() -> Generator:
            yield
            return False

        payloads_sent = []

        def mock_finish(payload: Any) -> Generator:
            payloads_sent.append(payload)
            yield

        behaviour.should_fetch_metrics_again = mock_should_fetch
        behaviour.finish_behaviour = mock_finish

        gen = behaviour.async_act()
        exhaust_generator(gen)

        assert len(payloads_sent) == 1
        assert payloads_sent[0].vote is False

    def test_fetch_success_all_metrics_valid(self) -> None:
        """Test successful fetch where all metrics have valid values."""
        behaviour = _make_behaviour()
        behaviour.params.is_agent_performance_summary_enabled = True

        def mock_should_fetch() -> Generator:
            yield
            return True

        def mock_fetch() -> Generator:
            behaviour._agent_performance_summary = AgentPerformanceSummary(
                timestamp=1700000000,
                metrics=[
                    AgentPerformanceMetrics(
                        name=IMPRESSIONS_METRIC_NAME,
                        is_primary=True,
                        value="100",
                    ),
                    AgentPerformanceMetrics(
                        name=LIKES_METRIC_NAME,
                        is_primary=False,
                        value="10",
                    ),
                ],
            )
            yield

        saved_summaries = []

        def mock_save(summary: Any) -> None:
            saved_summaries.append(summary)

        payloads_sent = []

        def mock_finish(payload: Any) -> Generator:
            payloads_sent.append(payload)
            yield

        behaviour.should_fetch_metrics_again = mock_should_fetch
        behaviour._fetch_agent_performance_summary = mock_fetch
        behaviour._save_agent_performance_summary = mock_save
        behaviour.finish_behaviour = mock_finish

        gen = behaviour.async_act()
        exhaust_generator(gen)

        assert len(payloads_sent) == 1
        assert payloads_sent[0].vote is True
        assert len(saved_summaries) == 1

    def test_fetch_with_na_metrics(self) -> None:
        """Test fetch where some metrics have N/A values sets vote=False."""
        behaviour = _make_behaviour()
        behaviour.params.is_agent_performance_summary_enabled = True

        def mock_should_fetch() -> Generator:
            yield
            return True

        def mock_fetch() -> Generator:
            behaviour._agent_performance_summary = AgentPerformanceSummary(
                timestamp=1700000000,
                metrics=[
                    AgentPerformanceMetrics(
                        name=IMPRESSIONS_METRIC_NAME,
                        is_primary=True,
                        value=NA,
                    ),
                    AgentPerformanceMetrics(
                        name=LIKES_METRIC_NAME,
                        is_primary=False,
                        value="10",
                    ),
                ],
            )
            yield

        payloads_sent = []

        def mock_finish(payload: Any) -> Generator:
            payloads_sent.append(payload)
            yield

        behaviour.should_fetch_metrics_again = mock_should_fetch
        behaviour._fetch_agent_performance_summary = mock_fetch
        behaviour._save_agent_performance_summary = MagicMock()
        behaviour.finish_behaviour = mock_finish

        gen = behaviour.async_act()
        exhaust_generator(gen)

        assert len(payloads_sent) == 1
        assert payloads_sent[0].vote is False
        behaviour.context.logger.warning.assert_called()


class TestFinishBehaviour:  # pylint: disable=too-few-public-methods
    """Tests for finish_behaviour."""

    def test_finish_behaviour(self) -> None:
        """Test finish_behaviour sends transaction and waits for round end."""
        behaviour = _make_behaviour()

        def mock_send_a2a(payload: Any) -> Generator:
            yield

        def mock_wait() -> Generator:
            yield

        behaviour.send_a2a_transaction = mock_send_a2a
        behaviour.wait_until_round_end = mock_wait
        behaviour.set_done = MagicMock()

        payload = FetchPerformanceDataPayload(sender="test_agent_address", vote=True)

        gen = behaviour.finish_behaviour(payload)
        exhaust_generator(gen)

        behaviour.set_done.assert_called_once()


class TestAgentPerformanceSummaryRoundBehaviour:  # pylint: disable=too-few-public-methods
    """Tests for AgentPerformanceSummaryRoundBehaviour."""

    def test_class_attributes(self) -> None:
        """Test round behaviour class attributes."""
        assert (
            AgentPerformanceSummaryRoundBehaviour.initial_behaviour_cls
            == FetchPerformanceSummaryBehaviour
        )
        assert FetchPerformanceSummaryBehaviour in (
            AgentPerformanceSummaryRoundBehaviour.behaviours
        )
