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

"""This module contains the transaction payloads of the MemeooorrAbciApp."""

from dataclasses import dataclass
from typing import Optional
from packages.valory.skills.abstract_round_abci.base import BaseTxPayload

@dataclass(frozen=True)
class SearchTweetsPayload(BaseTxPayload):
    """Represent a transaction payload for the SearchTweetsRound."""

    pending_tweets: str


@dataclass(frozen=True)
class CheckFundsPayload(BaseTxPayload):
    """Represent a transaction payload for the CheckFundsRound."""

    event: str


@dataclass(frozen=True)
class DeploymentPayload(BaseTxPayload):
    """Represent a transaction payload for the DeploymentRound."""

    pending_tweets: str
    tx_hash: Optional[str]


@dataclass(frozen=True)
class PostTweetPayload(BaseTxPayload):
    """Represent a transaction payload for the PostTweetRound."""

    event: str