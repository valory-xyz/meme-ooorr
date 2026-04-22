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

"""Tests for rounds_info.py."""

from packages.valory.skills.memeooorr_abci.rounds_info import ROUNDS_INFO


class TestRoundsInfo:
    """Tests for ROUNDS_INFO dict."""

    def test_all_entries_have_name_and_description(self) -> None:
        """Test all entries have 'name' and 'description' keys."""
        for key, value in ROUNDS_INFO.items():
            assert "name" in value, f"Missing 'name' for round: {key}"  # type: ignore[operator]
            assert "description" in value, f"Missing 'description' for round: {key}"  # type: ignore[operator]

    def test_all_names_are_strings(self) -> None:
        """Test all name values are non-empty strings."""
        for key, value in ROUNDS_INFO.items():
            assert isinstance(value["name"], str), f"name not str for round: {key}"  # type: ignore[index]
            assert len(value["name"]) > 0, f"empty name for round: {key}"  # type: ignore[index]

    def test_all_descriptions_are_strings(self) -> None:
        """Test all description values are non-empty strings."""
        for key, value in ROUNDS_INFO.items():
            assert isinstance(
                value["description"], str  # type: ignore[index]
            ), f"description not str for round: {key}"
            assert len(value["description"]) > 0, f"empty description for round: {key}"  # type: ignore[index]

    def test_expected_round_keys_present(self) -> None:
        """Test that key round names from MemeooorrAbciApp are present."""
        expected_keys = [
            "load_database_round",
            "check_staking_round",
            "pull_memes_round",
            "collect_feedback_round",
            "engage_twitter_round",
            "action_decision_round",
            "action_preparation_round",
            "action_tweet_round",
            "check_funds_round",
            "post_tx_decision_making_round",
            "call_checkpoint_round",
            "post_mech_response_round",
            "transaction_loop_check_round",
            "failed_mech_request_round",
            "failed_mech_response_round",
        ]
        for key in expected_keys:
            assert key in ROUNDS_INFO, f"Missing round key: {key}"

    def test_all_keys_are_snake_case(self) -> None:
        """Test all keys follow snake_case convention."""
        for key in ROUNDS_INFO:
            assert key == key.lower(), f"Key not lowercase: {key}"
            assert " " not in key, f"Key contains spaces: {key}"
