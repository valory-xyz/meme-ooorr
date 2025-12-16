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

"""This module contains the information about the rounds that is used by the Http handler."""


ROUNDS_INFO = {
    "action_decision_round": {
        "name": "Updating persona",
        "description": "Assesses the engagement from recent X posts and refines the agent's persona.",
    },
    "action_preparation_round": {
        "name": "Preparing Action Transaction",
        "description": "Constructs the exact on-chain command for the chosen token action, ensuring all parameters are set for successful execution",
    },
    "action_tweet_round": {
        "name": "Posting Action Tweet",
        "description": "Notifies followers about the executed token action, reinforcing engagement and transparency in the ongoing narrative",
    },
    "check_funds_round": {
        "name": "Checking Funds",
        "description": "The agent checks whether it has enough available funds for any required on-chain activity.",
    },
    "check_late_tx_hashes_round": {
        "name": "Reviewing Pending Transactions",
        "description": "The agent looks for any delayed or stuck transactions and updates their status.",
    },
    "check_transaction_history_round": {
        "name": "Reviewing Transaction History",
        "description": "The agent checks past transactions to ensure everything is in sync and up to date.",
    },
    "collect_feedback_round": {
        "name": "Reading Replies",
        "description": "The agent gathers replies to its latest post to understand audience reactions and adjust future tone or style.",
    },
    "collect_signature_round": {
        "name": "Preparing a Signature",
        "description": "The agent signs the action it’s about to take, completing a required step for sending certain transactions.",
    },
    "engage_twitter_round": {
        "name": "Engaging on X",
        "description": "The agent interacts with posts from other accounts—liking, replying, or reacting to stay active.",
    },
    "finalization_round": {
        "name": "Sending a transaction",
        "description": "The agent sends a required transaction and waits for it to be processed.",
    },
    "load_database_round": {
        "name": "Loading Persona & History",
        "description": "Loads Persona & History",
    },
    "pull_memes_round": {
        "name": "Pulling Meme Data",
        "description": "Fetches updated information on meme tokens and determines which ones to act on and what action to take.",
    },
    "randomness_transaction_submission_round": {
        "name": "Generating Randomness",
        "description": "The agent collects the randomness it needs to vary its behavior or decisions.",
    },
    "registration_round": {
        "name": "Setting Up the Agent",
        "description": "The agent completes its setup steps to begin operating correctly.",
    },
    "registration_startup_round": {
        "name": "Starting Up",
        "description": "The agent initializes and gets everything ready to begin its activities.",
    },
    "reset_and_pause_round": {
        "name": "Taking a Short Break",
        "description": "The agent pauses before continuing its next cycle of activity.",
    },
    "reset_round": {
        "name": "Cleaning up and resetting",
        "description": "The agent clears temporary data and resets before continuing.",
    },
    "select_keeper_transaction_submission_a_round": {
        "name": "Enabling agent to send the transaction",
        "description": "Aligns agent components for transaction submission.",
    },
    "select_keeper_transaction_submission_b_after_timeout_round": {
        "name": "Enabling agent to send the transaction",
        "description": "Aligns agent components for transaction submission.",
    },
    "select_keeper_transaction_submission_b_round": {
        "name": "Enabling agent to send the transaction",
        "description": "Aligns agent components for transaction submission.",
    },
    "synchronize_late_messages_round": {
        "name": "Syncing messages",
        "description": "The agent catches up on any delayed messages to stay up to date.",
    },
    "transaction_multiplexer_round": {
        "name": "Coordinating transactions",
        "description": "Coordinates multiple blockchain operations into a coherent sequence, ensuring efficient and timely execution of strategic steps",
    },
    "validate_transaction_round": {
        "name": "Validating the transaction",
        "description": "The agent checks that the sent transaction was completed successfully.",
    },
    "call_checkpoint_round": {
        "name": "Checking service status",
        "description": "The agent verifies whether the service is active and whether it’s time to perform a required checkpoint action.",
    },
    "check_staking_round": {
        "name": "Checking staking status",
        "description": "The agent confirms that staking conditions are met so it can continue running properly.",
    },
    "post_tx_decision_making_round": {
        "name": "Reviewing results",
        "description": "The agent looks at the outcome of the last transaction and decides what to do next.",
    },
    "transaction_loop_check_round": {
        "name": "Ensuring smooth operation",
        "description": "The agent checks for any stuck processes and resolves them if needed.",
    },
    "failed_mech_request_round": {
        "name": "Retrying a service request",
        "description": "The agent handles a failed request to an external service.",
    },
    "failed_mech_response_round": {
        "name": "Handling a failed response",
        "description": "The agent manages an error from an external service response.",
    },
    "mech_request_round": {
        "name": "Requesting a service",
        "description": "The agent sends a request to an external service it needs to operate.",
    },
    "mech_response_round": {
        "name": "Processing a service response",
        "description": "The agent handles the response it received from an external service.",
    },
    "post_mech_response_round": {
        "name": "Completing service handling",
        "description": "The agent wraps up tasks related to the service response.",
    },
    "mech_purchase_subscription_round": {
        "name": "Preparing an NVM purchase subscription tx",
        "description": "Preparing a transaction to purchase a Nevermined subscription, in order to interact with a Nevermind external service.",
        "transitions": {},
    },
    "fetch_performance_data_round": {
        "name": "Fetching agent performance summary",
        "description": "The agent gathers statistics about its recent performance and activity.",
    },
}
