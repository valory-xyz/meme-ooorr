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
    "ActionDecisionRound": {
        "name": "Taking a decision on what to do with the token",
        "description": "Takes a decision about whether to interact with other tokens",
        "transitions": {},
    },
    "ActionPreparationRound": {
        "name": "Preparing the token action transaction",
        "description": "Prepares a transaction to unleash, hearth, collect or burn a token",
        "transitions": {},
    },
    "ActionTweetRound": {
        "name": "Tweeting about the token action",
        "description": "Publishes a tweet announcing the performed token action",
        "transitions": {},
    },
    "AnalizeFeedbackRound": {
        "name": "Analyzing Twitter feedback",
        "description": "Analyzes responses to agent tweets and extracts feedback from them",
        "transitions": {},
    },
    "CheckFundsRound": {
        "name": "Checking funds",
        "description": "Checks the agetn has enough funds to send a transaction",
        "transitions": {},
    },
    "CheckLateTxHashesRound": {
        "name": "Checking late transaction hashes",
        "description": "Checks late transaction hashes",
        "transitions": {},
    },
    "CheckTransactionHistoryRound": {
        "name": "Checking the transaction history",
        "description": "Checks the transaction history",
        "transitions": {},
    },
    "CollectFeedbackRound": {
        "name": "Collecting feedback from Twitter",
        "description": "Collects responses to agent tweets",
        "transitions": {},
    },
    "CollectSignatureRound": {
        "name": "Collecting agent signatures",
        "description": "Collects agent signatures for a transaction",
        "transitions": {},
    },
    "DeploymentRound": {
        "name": "Preparing a token deployment",
        "description": "Prepares a token deployment transaction",
        "transitions": {},
    },
    "EngageRound": {
        "name": "Engaging with other agents",
        "description": "Responds to tweets from other agents",
        "transitions": {},
    },
    "FinalizationRound": {
        "name": "Sending a transaction",
        "description": "Sends a transaction for mining",
        "transitions": {},
    },
    "LoadDatabaseRound": {
        "name": "Loading the database",
        "description": "Loads the database",
        "transitions": {},
    },
    "PostAnnouncementRound": {
        "name": "Twitting about the token deployment",
        "description": "Publishes a tweet about the token deployment",
        "transitions": {},
    },
    "PostTweetRound": {
        "name": "Publishing a tweet",
        "description": "Publishes a tweet",
        "transitions": {},
    },
    "PullMemesRound": {
        "name": "Pulling other tokens from the subgraph",
        "description": "Pulls other tokens from the subgraph",
        "transitions": {},
    },
    "RandomnessTransactionSubmissionRound": {
        "name": "Getting some randomness",
        "description": "Gets randomness from a decentralized randomness source",
        "transitions": {},
    },
    "RegistrationRound": {
        "name": "Registering agents ",
        "description": "Initializes the agent registration process",
        "transitions": {},
    },
    "RegistrationStartupRound": {
        "name": "Registering agents at startup",
        "description": "Initializes the agent registration process",
        "transitions": {},
    },
    "ResetAndPauseRound": {
        "name": "Cleaning up and sleeping for some time",
        "description": "Cleans up and sleeps for some time before running again",
        "transitions": {},
    },
    "ResetRound": {
        "name": "Cleaning up and resetting",
        "description": "Cleans up and resets the agent",
        "transitions": {},
    },
    "SelectKeeperTransactionSubmissionARound": {
        "name": "Selecting an agent to send the transaction",
        "description": "Selects an agent to send the transaction",
        "transitions": {},
    },
    "SelectKeeperTransactionSubmissionBAfterTimeoutRound": {
        "name": "Selecting an agent to send the transaction",
        "description": "Selects an agent to send the transaction",
        "transitions": {},
    },
    "SelectKeeperTransactionSubmissionBRound": {
        "name": "Selecting an agent to send the transaction",
        "description": "Selects an agent to send the transaction",
        "transitions": {},
    },
    "SynchronizeLateMessagesRound": {
        "name": "Synchronizing late messages",
        "description": "Synchronizes late messages",
        "transitions": {},
    },
    "TransactionMultiplexerRound": {
        "name": "Selecting next round",
        "description": "Decides where to transition next based on the state previous to the transaction",
        "transitions": {},
    },
    "ValidateTransactionRound": {
        "name": "Validating the transaction",
        "description": "Checks that the transaction was succesful",
        "transitions": {},
    },
}
