# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2021-2024 David Vilela Freire
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

"""Tests for SrrDialogues in the tweepy connection."""

from unittest.mock import MagicMock, patch

from aea.configurations.base import PublicId
from aea.protocols.dialogue.base import Dialogues

from packages.dvilela.connections.tweepy.connection import SrrDialogues
from packages.valory.protocols.srr.dialogues import SrrDialogue


def test_role_from_first_message() -> None:
    """Test that SrrDialogues assigns the CONNECTION role for incoming messages.

    The SrrDialogues.__init__ defines an inner function `role_from_first_message`
    that always returns SrrDialogue.Role.CONNECTION. This test verifies that
    the callback is correctly wired and returns the expected role.
    """
    connection_id = PublicId.from_str("dvilela/tweepy:0.1.0")

    # Capture the role_from_first_message callback passed to BaseSrrDialogues.__init__
    with patch.object(Dialogues, "__init__", return_value=None) as mock_init:
        SrrDialogues(connection_id=connection_id)

        # Extract the role_from_first_message callback from the call args
        _, kwargs = mock_init.call_args
        role_fn = kwargs["role_from_first_message"]

        # Verify it returns CONNECTION for any message/address
        role = role_fn(MagicMock(), "any_address")
        assert role == SrrDialogue.Role.CONNECTION
