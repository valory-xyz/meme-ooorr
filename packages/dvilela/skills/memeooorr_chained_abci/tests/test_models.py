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

# pylint: disable=protected-access

from unittest.mock import MagicMock, patch

from packages.dvilela.skills.memeooorr_abci.rounds import Event as MemeooorrEvent
from packages.dvilela.skills.memeooorr_chained_abci.composition import (
    MemeooorrChainedSkillAbciApp,
)
from packages.dvilela.skills.memeooorr_chained_abci.models import (
    MARGIN,
    MULTIPLIER,
    MULTIPLIER_MECH,
    SharedState,
)
from packages.valory.skills.agent_performance_summary_abci.models import (
    SharedState as BaseSharedState,
)
from packages.valory.skills.mech_interact_abci.rounds import Event as MechInteractEvent
from packages.valory.skills.reset_pause_abci.rounds import Event as ResetPauseEvent


class TestSharedState:
    """Tests for SharedState."""

    def test_init_sets_env_var_status(self) -> None:
        """Test that __init__ sets env_var_status attribute."""
        mock_context = MagicMock()
        with patch.object(BaseSharedState, "__init__", return_value=None):
            shared_state = SharedState(skill_context=mock_context)
        assert shared_state.env_var_status == {
            "needs_update": False,
            "env_vars": {},
        }

    def test_setup_sets_reset_pause_round_timeout(self) -> None:
        """Test that setup sets ResetPauseEvent.ROUND_TIMEOUT."""
        mock_context = MagicMock()
        mock_context.params.round_timeout_seconds = 30.0
        mock_context.params.reset_pause_duration = 10

        with patch.object(BaseSharedState, "__init__", return_value=None):
            shared_state = SharedState(skill_context=mock_context)

        shared_state._context = mock_context

        with patch.object(BaseSharedState, "setup"):
            shared_state.setup()

        assert (
            MemeooorrChainedSkillAbciApp.event_to_timeout[ResetPauseEvent.ROUND_TIMEOUT]
            == 30.0
        )

    def test_setup_sets_reset_and_pause_timeout(self) -> None:
        """Test that setup sets ResetPauseEvent.RESET_AND_PAUSE_TIMEOUT."""
        mock_context = MagicMock()
        mock_context.params.round_timeout_seconds = 30.0
        mock_context.params.reset_pause_duration = 10

        with patch.object(BaseSharedState, "__init__", return_value=None):
            shared_state = SharedState(skill_context=mock_context)

        shared_state._context = mock_context

        with patch.object(BaseSharedState, "setup"):
            shared_state.setup()

        assert (
            MemeooorrChainedSkillAbciApp.event_to_timeout[
                ResetPauseEvent.RESET_AND_PAUSE_TIMEOUT
            ]
            == 10 + MARGIN
        )

    def test_setup_sets_memeooorr_round_timeout(self) -> None:
        """Test that setup sets MemeooorrEvent.ROUND_TIMEOUT with MULTIPLIER."""
        mock_context = MagicMock()
        mock_context.params.round_timeout_seconds = 30.0
        mock_context.params.reset_pause_duration = 10

        with patch.object(BaseSharedState, "__init__", return_value=None):
            shared_state = SharedState(skill_context=mock_context)

        shared_state._context = mock_context

        with patch.object(BaseSharedState, "setup"):
            shared_state.setup()

        assert (
            MemeooorrChainedSkillAbciApp.event_to_timeout[MemeooorrEvent.ROUND_TIMEOUT]
            == 30.0 * MULTIPLIER
        )

    def test_setup_sets_mech_interact_round_timeout(self) -> None:
        """Test that setup sets MechInteractEvent.ROUND_TIMEOUT with MULTIPLIER_MECH."""
        mock_context = MagicMock()
        mock_context.params.round_timeout_seconds = 30.0
        mock_context.params.reset_pause_duration = 10

        with patch.object(BaseSharedState, "__init__", return_value=None):
            shared_state = SharedState(skill_context=mock_context)

        shared_state._context = mock_context

        with patch.object(BaseSharedState, "setup"):
            shared_state.setup()

        assert (
            MemeooorrChainedSkillAbciApp.event_to_timeout[
                MechInteractEvent.ROUND_TIMEOUT
            ]
            == 30.0 * MULTIPLIER_MECH
        )

    def test_setup_calls_super_setup(self) -> None:
        """Test that setup calls super().setup()."""
        mock_context = MagicMock()
        mock_context.params.round_timeout_seconds = 30.0
        mock_context.params.reset_pause_duration = 10

        with patch.object(BaseSharedState, "__init__", return_value=None):
            shared_state = SharedState(skill_context=mock_context)

        shared_state._context = mock_context

        with patch.object(BaseSharedState, "setup") as mock_super_setup:
            shared_state.setup()
            mock_super_setup.assert_called_once()
