# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024-2026 Valory AG
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

"""Tests for composition.py."""

import packages.valory.skills.agent_performance_summary_abci.rounds as AgentPerformanceSummaryAbci
import packages.valory.skills.mech_interact_abci.states.final_states as MechFinalStates
import packages.valory.skills.mech_interact_abci.states.mech_version as MechVersionStates
import packages.valory.skills.mech_interact_abci.states.request as MechRequestStates
import packages.valory.skills.mech_interact_abci.states.response as MechResponseStates
import packages.valory.skills.memeooorr_abci.rounds as MemeooorrAbci
import packages.valory.skills.registration_abci.rounds as RegistrationAbci
import packages.valory.skills.reset_pause_abci.rounds as ResetAndPauseAbci
import packages.valory.skills.transaction_settlement_abci.rounds as TransactionSettlementAbci
from packages.valory.skills.memeooorr_chained_abci.composition import (
    MemeooorrChainedSkillAbciApp,
    abci_app_transition_mapping,
    termination_config,
)
from packages.valory.skills.termination_abci.rounds import (
    BackgroundRound,
    Event,
    TerminationAbciApp,
)


class TestAbciAppTransitionMapping:
    """Tests for the abci_app_transition_mapping."""

    def test_registration_finished_maps_to_performance_data(self) -> None:
        """Test FinishedRegistrationRound maps to FetchPerformanceDataRound."""
        assert (
            abci_app_transition_mapping[RegistrationAbci.FinishedRegistrationRound]
            is AgentPerformanceSummaryAbci.FetchPerformanceDataRound
        )

    def test_performance_finished_maps_to_load_database(self) -> None:
        """Test FinishedFetchPerformanceDataRound maps to LoadDatabaseRound."""
        assert (
            abci_app_transition_mapping[
                AgentPerformanceSummaryAbci.FinishedFetchPerformanceDataRound
            ]
            is MemeooorrAbci.LoadDatabaseRound
        )

    def test_memeooorr_finished_to_reset(self) -> None:
        """Test FinishedToResetRound maps to ResetAndPauseRound."""
        assert (
            abci_app_transition_mapping[MemeooorrAbci.FinishedToResetRound]
            is ResetAndPauseAbci.ResetAndPauseRound
        )

    def test_memeooorr_finished_to_settlement(self) -> None:
        """Test FinishedToSettlementRound maps to RandomnessTransactionSubmissionRound."""
        assert (
            abci_app_transition_mapping[MemeooorrAbci.FinishedToSettlementRound]
            is TransactionSettlementAbci.RandomnessTransactionSubmissionRound
        )

    def test_transaction_finished_maps_to_post_tx_decision(self) -> None:
        """Test FinishedTransactionSubmissionRound maps to PostTxDecisionMakingRound."""
        assert (
            abci_app_transition_mapping[
                TransactionSettlementAbci.FinishedTransactionSubmissionRound
            ]
            is MemeooorrAbci.PostTxDecisionMakingRound
        )

    def test_transaction_failed_maps_to_loop_check(self) -> None:
        """Test FailedRound maps to TransactionLoopCheckRound."""
        assert (
            abci_app_transition_mapping[TransactionSettlementAbci.FailedRound]
            is MemeooorrAbci.TransactionLoopCheckRound
        )

    def test_reset_finished_maps_to_performance_data(self) -> None:
        """Test FinishedResetAndPauseRound maps to FetchPerformanceDataRound."""
        assert (
            abci_app_transition_mapping[ResetAndPauseAbci.FinishedResetAndPauseRound]
            is AgentPerformanceSummaryAbci.FetchPerformanceDataRound
        )

    def test_reset_error_maps_to_reset(self) -> None:
        """Test FinishedResetAndPauseErrorRound maps to ResetAndPauseRound."""
        assert (
            abci_app_transition_mapping[
                ResetAndPauseAbci.FinishedResetAndPauseErrorRound
            ]
            is ResetAndPauseAbci.ResetAndPauseRound
        )

    def test_memeooorr_mech_request_maps_to_version_detection(self) -> None:
        """Test FinishedForMechRequestRound maps to MechVersionDetectionRound."""
        assert (
            abci_app_transition_mapping[MemeooorrAbci.FinishedForMechRequestRound]
            is MechVersionStates.MechVersionDetectionRound
        )

    def test_mech_legacy_detected_maps_to_request(self) -> None:
        """Test FinishedMechLegacyDetectedRound maps to MechRequestRound."""
        assert (
            abci_app_transition_mapping[MechFinalStates.FinishedMechLegacyDetectedRound]
            is MechRequestStates.MechRequestRound
        )

    def test_mech_information_maps_to_request(self) -> None:
        """Test FinishedMechInformationRound maps to MechRequestRound."""
        assert (
            abci_app_transition_mapping[MechFinalStates.FinishedMechInformationRound]
            is MechRequestStates.MechRequestRound
        )

    def test_failed_mech_information_maps_to_version_detection(self) -> None:
        """Test FailedMechInformationRound maps to MechVersionDetectionRound."""
        assert (
            abci_app_transition_mapping[MechFinalStates.FailedMechInformationRound]
            is MechVersionStates.MechVersionDetectionRound
        )

    def test_mech_request_finished_maps_to_settlement(self) -> None:
        """Test FinishedMechRequestRound maps to RandomnessTransactionSubmissionRound."""
        assert (
            abci_app_transition_mapping[MechFinalStates.FinishedMechRequestRound]
            is TransactionSettlementAbci.RandomnessTransactionSubmissionRound
        )

    def test_mech_response_finished_maps_to_post_mech(self) -> None:
        """Test FinishedMechResponseRound maps to PostMechResponseRound."""
        assert (
            abci_app_transition_mapping[MechFinalStates.FinishedMechResponseRound]
            is MemeooorrAbci.PostMechResponseRound
        )

    def test_mech_request_skip_maps_to_failed_request(self) -> None:
        """Test FinishedMechRequestSkipRound maps to FailedMechRequestRound."""
        assert (
            abci_app_transition_mapping[MechFinalStates.FinishedMechRequestSkipRound]
            is MemeooorrAbci.FailedMechRequestRound
        )

    def test_mech_response_timeout_maps_to_failed_response(self) -> None:
        """Test FinishedMechResponseTimeoutRound maps to FailedMechResponseRound."""
        assert (
            abci_app_transition_mapping[
                MechFinalStates.FinishedMechResponseTimeoutRound
            ]
            is MemeooorrAbci.FailedMechResponseRound
        )

    def test_memeooorr_mech_response_maps_to_mech_response(self) -> None:
        """Test FinishedForMechResponseRound maps to MechResponseRound."""
        assert (
            abci_app_transition_mapping[MemeooorrAbci.FinishedForMechResponseRound]
            is MechResponseStates.MechResponseRound
        )


class TestTerminationConfig:
    """Tests for termination_config."""

    def test_round_cls(self) -> None:
        """Test round_cls is BackgroundRound."""
        assert termination_config.round_cls is BackgroundRound

    def test_start_event(self) -> None:
        """Test start_event is Event.TERMINATE."""
        assert termination_config.start_event is Event.TERMINATE

    def test_abci_app(self) -> None:
        """Test abci_app is TerminationAbciApp."""
        assert termination_config.abci_app is TerminationAbciApp


class TestMemeooorrChainedSkillAbciApp:
    """Tests for the chained ABCI app."""

    def test_has_transition_function(self) -> None:
        """Test that the chained app has a transition_function."""
        assert hasattr(MemeooorrChainedSkillAbciApp, "transition_function")
        assert len(MemeooorrChainedSkillAbciApp.transition_function) > 0

    def test_initial_round_cls(self) -> None:
        """Test that the initial round is the registration round."""
        assert (
            MemeooorrChainedSkillAbciApp.initial_round_cls
            is RegistrationAbci.RegistrationStartupRound
        )

    def test_contains_registration_rounds(self) -> None:
        """Test that the chained app contains registration rounds."""
        all_rounds = set(MemeooorrChainedSkillAbciApp.transition_function.keys())
        assert RegistrationAbci.RegistrationStartupRound in all_rounds

    def test_contains_memeooorr_rounds(self) -> None:
        """Test that the chained app contains memeooorr rounds."""
        all_rounds = set(MemeooorrChainedSkillAbciApp.transition_function.keys())
        assert MemeooorrAbci.LoadDatabaseRound in all_rounds

    def test_contains_reset_pause_rounds(self) -> None:
        """Test that the chained app contains reset pause rounds."""
        all_rounds = set(MemeooorrChainedSkillAbciApp.transition_function.keys())
        assert ResetAndPauseAbci.ResetAndPauseRound in all_rounds

    def test_contains_transaction_settlement_rounds(self) -> None:
        """Test that the chained app contains transaction settlement rounds."""
        all_rounds = set(MemeooorrChainedSkillAbciApp.transition_function.keys())
        assert (
            TransactionSettlementAbci.RandomnessTransactionSubmissionRound in all_rounds
        )

    def test_contains_performance_summary_rounds(self) -> None:
        """Test that the chained app contains performance summary rounds."""
        all_rounds = set(MemeooorrChainedSkillAbciApp.transition_function.keys())
        assert AgentPerformanceSummaryAbci.FetchPerformanceDataRound in all_rounds

    def test_contains_mech_interact_rounds(self) -> None:
        """Test that the chained app contains mech interact rounds."""
        all_rounds = set(MemeooorrChainedSkillAbciApp.transition_function.keys())
        assert MechVersionStates.MechVersionDetectionRound in all_rounds
