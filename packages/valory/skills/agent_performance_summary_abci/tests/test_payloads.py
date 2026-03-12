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

"""Tests for the payloads module of the agent_performance_summary_abci skill."""

import pytest

from packages.valory.skills.agent_performance_summary_abci.payloads import (
    FetchPerformanceDataPayload,
)

SENDER = "test_sender"


class TestFetchPerformanceDataPayload:
    """Tests for the FetchPerformanceDataPayload."""

    @pytest.mark.parametrize("vote", [True, False])
    def test_construction(self, vote: bool) -> None:
        """Test payload construction with vote=True and vote=False."""
        payload = FetchPerformanceDataPayload(sender=SENDER, vote=vote)
        assert payload.sender == SENDER
        assert payload.vote is vote

    def test_frozen(self) -> None:
        """Test that the payload is frozen (immutable)."""
        payload = FetchPerformanceDataPayload(sender=SENDER, vote=True)
        with pytest.raises(AttributeError):
            payload.vote = False  # type: ignore
