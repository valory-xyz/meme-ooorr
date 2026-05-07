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

"""Tests for the handlers module of the agent_performance_summary_abci skill."""

# pylint: disable=W0212

from unittest.mock import MagicMock, patch

from packages.valory.skills.abstract_round_abci.handlers import (
    ABCIRoundHandler,
)
from packages.valory.skills.abstract_round_abci.handlers import (
    ContractApiHandler as BaseContractApiHandler,
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
from packages.valory.skills.agent_performance_summary_abci.handlers import (
    AgentPerformanceSummaryABCIHandler,
    ContractApiHandler,
    HttpHandler,
    IpfsHandler,
    LedgerApiHandler,
    SigningHandler,
    TendermintHandler,
)
from packages.valory.skills.agent_performance_summary_abci.models import (
    AgentPerformanceSummary,
    SharedState,
)
from packages.valory.skills.memeooorr_abci.handlers import (
    HttpHandler as BaseHttpHandler,
)
from packages.valory.skills.memeooorr_abci.handlers import (
    HttpMethod,
)


class TestHandlerAliases:
    """Tests that handler aliases reference the correct base classes."""

    def test_abci_handler(self) -> None:
        """Test AgentPerformanceSummaryABCIHandler alias."""
        assert AgentPerformanceSummaryABCIHandler is ABCIRoundHandler

    def test_signing_handler(self) -> None:
        """Test SigningHandler alias."""
        assert SigningHandler is BaseSigningHandler

    def test_ledger_api_handler(self) -> None:
        """Test LedgerApiHandler alias."""
        assert LedgerApiHandler is BaseLedgerApiHandler

    def test_contract_api_handler(self) -> None:
        """Test ContractApiHandler alias."""
        assert ContractApiHandler is BaseContractApiHandler

    def test_tendermint_handler(self) -> None:
        """Test TendermintHandler alias."""
        assert TendermintHandler is BaseTendermintHandler

    def test_ipfs_handler(self) -> None:
        """Test IpfsHandler alias."""
        assert IpfsHandler is BaseIpfsHandler


class TestHttpHandler:
    """Tests for the HttpHandler class."""

    def test_is_base_http_handler(self) -> None:
        """Test that HttpHandler is a subclass of BaseHttpHandler."""
        assert issubclass(HttpHandler, BaseHttpHandler)

    def test_setup_creates_routes(self) -> None:
        """Test that setup() creates the correct routes."""
        handler = HttpHandler.__new__(HttpHandler)
        handler.routes = {}
        handler._context = MagicMock()
        handler.context.params.service_endpoint = "http://example.com:8000"

        with patch.object(BaseHttpHandler, "setup"):
            handler.setup()

        # Check that handler_url_regex was set
        assert hasattr(handler, "handler_url_regex")
        assert "example.com" in handler.handler_url_regex

        # Check that performance summary route was added
        get_head_key = (HttpMethod.GET.value, HttpMethod.HEAD.value)
        assert get_head_key in handler.routes
        route_list = handler.routes[get_head_key]
        route_patterns = [pattern for pattern, _ in route_list]
        assert any("performance-summary" in p for p in route_patterns)

    def test_shared_state_property(self) -> None:
        """Test the shared_state property returns the context state."""
        handler = HttpHandler.__new__(HttpHandler)
        mock_state = MagicMock(spec=SharedState)
        handler._context = MagicMock()
        handler.context.state = mock_state

        result = handler.shared_state
        assert result is mock_state

    def test_get_performance_summary(self) -> None:
        """Test _get_performance_summary reads and sends response."""
        handler = HttpHandler.__new__(HttpHandler)

        mock_summary = AgentPerformanceSummary(
            timestamp=1234567890,
            metrics=[],
            agent_behavior="test behavior",
        )

        mock_state = MagicMock(spec=SharedState)
        mock_state.read_existing_performance_summary.return_value = mock_summary
        handler._context = MagicMock()
        handler.context.state = mock_state

        mock_http_msg = MagicMock()
        mock_http_dialogue = MagicMock()

        handler._send_ok_response = MagicMock()  # type: ignore[method-assign]

        handler._get_performance_summary(mock_http_msg, mock_http_dialogue)

        mock_state.read_existing_performance_summary.assert_called_once()
        handler._send_ok_response.assert_called_once_with(
            mock_http_msg, mock_http_dialogue, mock_summary.to_json()
        )
