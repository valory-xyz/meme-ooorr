# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023-2026 Valory AG
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

"""Tests for payloads.py."""

from dataclasses import FrozenInstanceError

import pytest

from packages.valory.skills.agent_db_abci.payloads import AgentDBPayload

SENDER = "test_sender"


class TestAgentDBPayload:
    """Tests for AgentDBPayload."""

    def test_construction(self) -> None:
        """Test payload construction with required fields."""
        payload = AgentDBPayload(sender=SENDER, content="test_content")
        assert payload.sender == SENDER
        assert payload.content == "test_content"

    def test_content_field(self) -> None:
        """Test that the content field stores value correctly."""
        payload = AgentDBPayload(sender=SENDER, content='{"key": "value"}')
        assert payload.content == '{"key": "value"}'

    def test_frozen(self) -> None:
        """Test that the dataclass is frozen."""
        payload = AgentDBPayload(sender=SENDER, content="test")
        with pytest.raises(FrozenInstanceError):
            payload.content = "new_content"  # type: ignore

    def test_different_senders(self) -> None:
        """Test payloads with different senders."""
        p1 = AgentDBPayload(sender="sender_1", content="same")
        p2 = AgentDBPayload(sender="sender_2", content="same")
        assert p1.sender != p2.sender
        assert p1.content == p2.content
