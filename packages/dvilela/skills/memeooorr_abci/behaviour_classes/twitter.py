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

"""This package contains round behaviours of MemeooorrAbciApp."""

import json
import random
import secrets
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional, Tuple, Type, Union
from uuid import uuid4

from twitter_text import parse_tweet  # type: ignore

from packages.dvilela.skills.memeooorr_abci.behaviour_classes.base import (
    MemeooorrBaseBehaviour,
)
from packages.dvilela.skills.memeooorr_abci.prompts import (
    ALTERNATIVE_MODEL_TWITTER_PROMPT,
    ENFORCE_ACTION_COMMAND,
    ENFORCE_ACTION_COMMAND_FAILED_MECH,
    MECH_RESPONSE_SUBPROMPT,
    TWITTER_DECISION_PROMPT,
    build_decision_schema,
)
from packages.dvilela.skills.memeooorr_abci.rounds import (
    ActionTweetPayload,
    ActionTweetRound,
    CollectFeedbackPayload,
    CollectFeedbackRound,
    EngageTwitterPayload,
    EngageTwitterRound,
    Event,
)
from packages.valory.skills.abstract_round_abci.base import AbstractRound
from packages.valory.skills.mech_interact_abci.states.base import MechMetadata
from packages.valory.skills.agent_db_abci.twitter_models import (
    TwitterPost,
    TwitterRewtweet,
    TwitterLike,
    TwitterFollow,
    TwitterAction,
)
from packages.valory.skills.agent_db_abci.agents_fun_db import AgentsFunAgent

MAX_TWEET_CHARS = 280
JSON_RESPONSE_REGEXES = [r"json.?({.*})", r"json({.*})", r"\`\`\`json(.*)\`\`\`"]
MAX_TWEET_PREPARATIONS_RETRIES = 3


def is_tweet_valid(tweet: str) -> bool:
    """Checks a tweet length"""
    return parse_tweet(tweet).asdict()["weightedLength"] <= MAX_TWEET_CHARS


# Define a context holder for interaction processing
@dataclass
class InteractionContext:
    """Holds the context required for processing multiple Twitter interactions within a single period."""

    pending_tweets: dict
    previous_tweets: Optional[List[dict]]
    persona: str
    new_interacted_tweet_ids: List[int]


class BaseTweetBehaviour(MemeooorrBaseBehaviour):  # pylint: disable=too-many-ancestors
    """Base behaviour for tweet-related actions."""

    matching_round: Type[AbstractRound] = None  # type: ignore

    def store_agent_action(self, action: TwitterAction):
        """Store agent action"""
        # get agent from db
        my_agent = yield from self.context.agents_fun_db.get_my_agent()
        if my_agent is None:
            self.context.logger.error("No agent found in AgentsFunDB")
            return
        try:
            import pdb

            pdb.set_trace()
            yield from my_agent.add_interaction(action)
        except Exception as e:
            self.context.logger.error(f"Error adding interaction: {e}")

    def _write_tweet_to_kv_store(
        self, tweet_data: Union[dict, List[dict]]
    ) -> Generator[None, None, bool]:
        """Store tweet data in the KV store under the 'tweets' key."""
        tweets = yield from self.get_tweets_from_db()
        if isinstance(tweet_data, list):
            tweets.extend(tweet_data)
        else:
            tweets.append(tweet_data)
        yield from self._write_kv({"tweets": json.dumps(tweets)})
        self.context.logger.info("Stored/Appended tweet data in KV store.")
        return True

    def post_tweet(
        self, tweet: Union[str, List[str]], store: bool = True
    ) -> Generator[None, None, Optional[Dict]]:
        """Post a tweet"""
        self.context.logger.info(f"Posting tweet: {tweet}")

        # Determine text for tweepy and for DB logging
        text_to_post = tweet[0] if isinstance(tweet, list) else tweet
        tweets_for_api = [{"text": text_to_post}]  # Tweepy expects a list of dicts

        # Post the tweet
        tweet_ids = yield from self._call_tweepy(
            method="post",
            tweets=tweets_for_api,
        )

        if not tweet_ids:
            self.context.logger.error("Failed posting to Twitter.")
            return None

        # Assuming only one tweet is posted at a time by this method based on tweet_ids[0]
        posted_tweet_id = str(tweet_ids[0])

        latest_tweet_data_for_kv = {
            "tweet_id": posted_tweet_id,
            "text": text_to_post,  # Store the actual text that was posted
            "timestamp": self.get_sync_timestamp(),
        }

        if store:
            # Write latest tweet to the KV store
            yield from self._write_tweet_to_kv_store(latest_tweet_data_for_kv)
            self.context.logger.info("Wrote latest tweet to KV db")

            # Store in AgentsFunDB
            try:
                post_action = TwitterPost(
                    tweet_id=posted_tweet_id,
                    text=text_to_post,  # Use the actual text posted
                    timestamp=datetime.now(timezone.utc),
                )

                yield from self.store_agent_action(post_action)
            except Exception as e:
                self.context.logger.error(
                    f"Error creating/storing TwitterPost action: {e}"
                )

        return latest_tweet_data_for_kv  # Return the structured data

    def respond_tweet(
        self,
        tweet_id: str,
        text: str,
        quote: bool = False,
        user_name: Optional[str] = None,
    ) -> Generator[None, None, bool]:
        """Like a tweet"""

        self.context.logger.info(f"Replying to tweet with ID: {tweet_id}")
        tweet = {"text": text}
        if quote:
            tweet["attachment_url"] = f"https://x.com/{user_name}/status/{tweet_id}"
        else:
            tweet["reply_to"] = tweet_id
        tweet_ids = yield from self._call_tweepy(
            method="post",
            tweets=[tweet],
        )
        if tweet_ids:
            # Store in AgentsFunDB
            try:
                post_action = TwitterPost(
                    tweet_id=str(
                        tweet_ids[0]
                    ),  # Assuming tweepy returns the new tweet's ID
                    text=text,
                    reply_to_tweet_id=str(tweet_id) if not quote else None,
                    timestamp=datetime.now(timezone.utc),
                )
                yield from self._store_agent_action_in_db(post_action)
            except Exception as e:
                self.context.logger.error(
                    f"Error creating/storing TwitterPost (reply/quote) action: {e}"
                )
        return tweet_ids is not None and tweet_ids

    def like_tweet(self, tweet_id: str) -> Generator[None, None, bool]:
        """Like a tweet"""
        self.context.logger.info(f"Liking tweet with ID: {tweet_id}")
        try:
            response = yield from self._call_tweepy(
                method="like_tweet", tweet_id=tweet_id
            )
            if response is None:
                self.context.logger.error(
                    f"Tweepy call for like_tweet ID {tweet_id} failed (returned None). See previous logs for details."
                )
                return False

            if not response.get("success", False):
                error_message = response.get("error", "Unknown error occurred.")
                self.context.logger.error(
                    f"Error liking tweet with ID {tweet_id}: {error_message}"
                )
                return False

            if response["success"]:
                # Store in AgentsFunDB
                try:
                    like_action = TwitterLike(
                        tweet_id=str(tweet_id), timestamp=datetime.now(timezone.utc)
                    )
                    yield from self._store_agent_action_in_db(like_action)
                except Exception as e:
                    self.context.logger.error(
                        f"Error creating/storing TwitterLike action: {e}"
                    )
            return response["success"]
        except Exception as e:  # pylint: disable=broad-except
            self.context.logger.error(f"Exception liking tweet with ID {tweet_id}: {e}")
            return False

    def retweet(self, tweet_id: str) -> Generator[None, None, bool]:
        """Retweet"""
        self.context.logger.info(f"Retweeting tweet with ID: {tweet_id}")
        try:
            response = yield from self._call_tweepy(method="retweet", tweet_id=tweet_id)
            if response is None:
                self.context.logger.error(
                    f"Tweepy call for retweet ID {tweet_id} failed (returned None). See previous logs for details."
                )
                return False

            if not response.get("success", False):
                error_message = response.get("error", "Unknown error occurred.")
                self.context.logger.error(
                    f"Error retweeting tweet with ID {tweet_id}: {error_message}"
                )
                return False

            if response["success"]:
                # Store in AgentsFunDB
                try:
                    retweet_action = TwitterRewtweet(
                        tweet_id=str(tweet_id), timestamp=datetime.now(timezone.utc)
                    )
                    yield from self._store_agent_action_in_db(retweet_action)
                except Exception as e:
                    self.context.logger.error(
                        f"Error creating/storing TwitterRetweet action: {e}"
                    )
            return response["success"]
        except Exception as e:  # pylint: disable=broad-except
            self.context.logger.error(
                f"Exception retweeting tweet with ID {tweet_id}: {e}"
            )
            return False

    def follow_user(self, user_name: str) -> Generator[None, None, bool]:
        """Follow user"""
        self.context.logger.info(f"Following user with user_name: {user_name}")
        try:
            response = yield from self._call_tweepy(
                method="follow_by_username", username=user_name
            )
            if response is None:
                self.context.logger.error(
                    f"Tweepy call for follow_user user_name {user_name} failed (returned None). See previous logs for details."
                )
                return False

            if not response.get("success", False):
                error_message = response.get("error", "Unknown error occurred.")
                self.context.logger.error(
                    f"Error Following user with user_name {user_name}: {error_message}"
                )
                return False

            # Store in AgentsFunDB
            # Assuming response from _call_tweepy contains the followed user's ID
            # For example: response.get("data", {}).get("id") or response.get("followed_user_id")
            followed_user_id = response.get("data", {}).get(
                "id"
            )  # Placeholder for actual key

            # The TwitterFollow model expects 'username', not 'followed_agent_id'
            # We will use the 'user_name' parameter that was passed to this method.
            if response["success"]:
                try:
                    follow_obj = TwitterFollow(
                        username=user_name,  # Use the passed user_name
                        timestamp=datetime.now(timezone.utc),
                    )
                    yield from self._store_agent_action_in_db(follow_obj)
                    self.context.logger.info(
                        f"Stored follow action for user {user_name} in AgentsFunDB."
                    )
                except Exception as e:
                    self.context.logger.error(
                        f"Failed to store follow action in AgentsFunDB for user {user_name}: {e}"
                    )
            # The old logging related to followed_user_id can be removed or adjusted if needed.
            # For now, we focus on storing the action with username.
            # if not followed_user_id:
            #     self.context.logger.warning(
            #         f"Could not extract followed user ID for {user_name} from tweepy response. "
            #         f"Follow action was successful but AgentsFunDB storage might be incomplete regarding user ID."
            #     )
            return response["success"]
        except Exception as e:  # pylint: disable=broad-except
            self.context.logger.error(
                f"Exception following user with user_name {user_name}: {e}"
            )
            return False

    def _format_previous_tweets_str(
        self, tweets: Optional[Union[List[TwitterPost], List[Dict[str, Any]]]]
    ) -> str:
        """Format previous tweets as a string"""
        if not tweets:
            return ""

        formatted_tweets: List[str] = []
        for post in tweets:
            if isinstance(post, TwitterPost):
                text = getattr(post, "text", "")
                timestamp = getattr(post, "timestamp", "")
                formatted_tweets.append(f"{text} ({timestamp})")
            elif isinstance(post, dict):
                text = post.get("text", "")
                timestamp = post.get("timestamp", "")
                formatted_tweets.append(f"{text} ({timestamp})")
        return "\\n".join(formatted_tweets)


class CollectFeedbackBehaviour(
    BaseTweetBehaviour
):  # pylint: disable=too-many-ancestors
    """CollectFeedbackBehaviour"""

    matching_round: Type[AbstractRound] = CollectFeedbackRound

    def async_act(self) -> Generator:
        """Do the act, supporting asynchronous execution."""

        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            feedback = yield from self.get_feedback()

            payload = CollectFeedbackPayload(
                sender=self.context.agent_address,
                feedback=(
                    json.dumps(feedback, sort_keys=True)
                    if feedback is not None
                    else json.dumps({"likes": 0, "retweets": 0, "replies": []})
                ),
            )

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()

    def _twitter_post_to_dict(self, post: TwitterPost) -> Dict[str, Any]:
        """Converts a TwitterPost object to a JSON-serializable dictionary."""
        timestamp = getattr(post, "timestamp", None)
        reply_to_tweet_id_val = getattr(post, "reply_to_tweet_id", None)

        return {
            "action": getattr(post, "action", None),
            "timestamp": timestamp.isoformat() if timestamp else None,
            "tweet_id": str(getattr(post, "tweet_id", None)),
            "text": getattr(post, "text", None),
            "reply_to_tweet_id": (
                str(reply_to_tweet_id_val)
                if reply_to_tweet_id_val is not None
                else None
            ),
        }

    def get_feedback(self) -> Generator[None, None, Optional[Dict[str, Any]]]:
        """Get the responses to our agent's tweets from MirrorDB."""

        self.context.logger.info("Attempting to get feedback for agent's tweets.")

        last_tweet = self._get_last_agent_tweet()
        if last_tweet is None:
            return {"likes": 0, "retweets": 0, "replies": []}
        # tweet_id_to_query = str(last_tweet.tweet_id)
        tweet_id_to_query = "1922648217251954876"
        # The following line is commented out as it seems to be for testing with a hardcoded ID
        self.context.logger.info(
            f"Last tweet: {last_tweet}, Querying feedback for tweet_id: {tweet_id_to_query}"
        )

        feedback_from_db = yield from self.context.agents_fun_db.get_tweet_feedback(
            tweet_id_to_query
        )

        if feedback_from_db is None:
            self.context.logger.warning(
                f"No feedback received from DB for tweet_id: {tweet_id_to_query}."
            )
            return {"likes": 0, "retweets": 0, "replies": []}

        processed_replies = self._process_raw_replies(
            feedback_from_db.get("replies", [])
        )

        feedback = {
            "likes": feedback_from_db.get("likes", 0),
            "retweets": feedback_from_db.get("retweets", 0),
            "replies": processed_replies,
        }
        self.context.logger.info(
            f"Processed feedback for tweet_id {tweet_id_to_query}: {feedback}"
        )
        return feedback

    def _get_last_agent_tweet(self) -> Optional[TwitterPost]:
        """Fetch and validate the last tweet posted by the agent."""
        if (
            not self.context.agents_fun_db.agents
            or not self.context.agents_fun_db.agents[0].posts
        ):
            self.context.logger.warning(
                "No agents or no posts found for the primary agent."
            )
            return None

        last_tweet = self.context.agents_fun_db.agents[0].posts[-1]

        if (
            not last_tweet
            or not hasattr(last_tweet, "tweet_id")
            or last_tweet.tweet_id is None
        ):
            self.context.logger.warning("Last tweet or its tweet_id is invalid.")
            return None

        return last_tweet

    def _process_raw_replies(self, raw_replies_list: Any) -> List[Dict[str, Any]]:
        """Process raw replies list from DB into a list of dictionaries."""
        processed_replies: List[Dict[str, Any]] = []

        if not isinstance(raw_replies_list, list):
            self.context.logger.warning(
                f"Replies in feedback_from_db is not a list: {type(raw_replies_list)}"
            )
            return processed_replies  # Return empty list

        for post_item in raw_replies_list:
            if isinstance(post_item, TwitterPost):
                processed_replies.append(self._twitter_post_to_dict(post_item))
            elif isinstance(post_item, dict):
                # If it's already a dict, ensure timestamp is ISO format if it's a datetime object
                if "timestamp" in post_item and isinstance(
                    post_item["timestamp"], datetime
                ):
                    post_item["timestamp"] = post_item["timestamp"].isoformat()
                processed_replies.append(post_item)
            else:
                self.context.logger.warning(
                    f"Unexpected item type in replies list: {type(post_item)}"
                )

        return processed_replies


class EngageTwitterBehaviour(BaseTweetBehaviour):  # pylint: disable=too-many-ancestors
    """EngageTwitterBehaviour"""

    matching_round: Type[AbstractRound] = EngageTwitterRound

    def async_act(self) -> Generator:
        """Do the act, supporting asynchronous execution."""

        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            event, new_mech_requests = yield from self.get_event()

            if new_mech_requests:
                mech_requests = json.dumps(new_mech_requests, sort_keys=True)
                self.context.logger.info(f"Mech Requests JSON: {mech_requests}")

            # Determine mech_request value based on event type
            mech_request = (
                json.dumps(new_mech_requests, sort_keys=True)
                if event == Event.MECH.value
                else None
            )

            # Create payload with appropriate mech_request value
            payload = EngageTwitterPayload(
                sender=self.context.agent_address,
                event=event,
                mech_request=mech_request,
                tx_submitter=self.matching_round.auto_round_id(),
            )

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()

    def get_event(self) -> Generator[None, None, Tuple[str, List]]:
        """
        Get the next event for Twitter engagement.

        Returns:
            Tuple[str, List]: Event type and any new mech requests.
        """
        new_mech_requests: List[Dict[str, Any]] = []

        # Handle differently based on mech_for_twitter state
        if self.synchronized_data.mech_for_twitter:
            (
                pending_tweets,
                interacted_tweet_ids,
            ) = yield from self._handle_mech_for_twitter()
        else:
            # Skip engagement if configured
            if self.params.skip_engagement:
                self.context.logger.info("Skipping engagement on Twitter")
                return Event.DONE.value, new_mech_requests
            (
                pending_tweets,
                interacted_tweet_ids,
            ) = yield from self._handle_regular_engagement()

        # Process interactions
        (
            event,
            new_interacted_tweet_ids,
            new_mech_requests,
        ) = yield from self.interact_twitter(pending_tweets)

        # Handle results based on event type
        if event == Event.MECH.value:
            return event, new_mech_requests

        if event == Event.DONE.value:
            yield from self._update_interacted_tweets(
                interacted_tweet_ids, new_interacted_tweet_ids
            )

        return event, new_mech_requests

    def _handle_mech_for_twitter(self) -> Generator[None, None, Tuple[dict, list]]:
        """
        Handle Twitter engagement when mech_for_twitter is True.

        Returns:
            Tuple[dict, list]: Pending tweets and interacted tweet IDs.
        """
        self.context.logger.info(
            "Mech for twitter detected, using Mech response for decision"
        )

        # Fetch pending tweets from db
        pending_tweets_data = yield from self._read_kv(
            keys=("pending_tweets_for_tw_mech",)
        )
        pending_tweets = {}
        if pending_tweets_data and pending_tweets_data.get(
            "pending_tweets_for_tw_mech"
        ):
            pending_tweets = json.loads(
                pending_tweets_data["pending_tweets_for_tw_mech"]
            )
        else:
            self.context.logger.warning(
                "No pending tweets found in KV store or value is empty."
            )

        # Fetch previously interacted tweets
        interacted_ids_data = yield from self._read_kv(
            keys=("interacted_tweet_ids_for_tw_mech",)
        )
        interacted_tweet_ids = []
        if interacted_ids_data and interacted_ids_data.get(
            "interacted_tweet_ids_for_tw_mech"
        ):
            interacted_tweet_ids = json.loads(
                interacted_ids_data["interacted_tweet_ids_for_tw_mech"]
            )
        else:
            self.context.logger.warning(
                "No interacted tweets found in KV store or value is empty."
            )

        return pending_tweets, interacted_tweet_ids

    def _handle_regular_engagement(self) -> Generator[None, None, Tuple[dict, list]]:
        """
        Handle Twitter engagement when mech_for_twitter is False.

        Returns:
            Tuple[dict, list]: Pending tweets and interacted tweet IDs.
        """

        self.context.logger.info("Entered Regular Engagement in EngageTwitterBehaviour")
        # Get other memeooorr handles
        active_agents = yield from self.get_agent_handles()
        self.context.logger.info(
            f"Found {len(active_agents) if active_agents else 0} active agents from agents_fun_db (or subgraph fallback)."
        )

        if not active_agents:
            self.context.logger.error("No valid Twitter agent data found.")
            return {}, []

        # Load previously interacted tweets
        interacted_tweet_ids = yield from self._get_interacted_tweet_ids()

        # Get latest tweets from each agent
        pending_tweets = yield from self._collect_pending_tweets(
            active_agents, set(interacted_tweet_ids)
        )

        # Store data for mech processing
        yield from self._store_engagement_data(interacted_tweet_ids, pending_tweets)

        return pending_tweets, interacted_tweet_ids

    def _get_interacted_tweet_ids(self) -> Generator[None, None, list]:
        """Get previously interacted tweet IDs from the database."""
        db_data = yield from self._read_kv(keys=("interacted_tweet_ids",))

        if db_data is None:
            self.context.logger.error("Error while loading the database")
            return []

        return json.loads(db_data["interacted_tweet_ids"] or "[]")

    def _collect_pending_tweets(
        self, active_agents: List["AgentsFunAgent"], interacted_tweet_ids: set[int]  # type: ignore
    ) -> Generator[None, None, dict]:
        """Collect pending tweets from active agents that haven't been interacted with."""
        pending_tweets: Dict[str, Dict[str, str]] = {}

        self.context.logger.info("Collecting pending tweets from active agents")

        if not active_agents:
            self.context.logger.info("No active agents to collect tweets from.")
            return pending_tweets

        for agent in active_agents:
            if not agent.loaded:
                # Ensure agent data, including posts, is loaded.
                # The load method in AgentsFunAgent is a generator, so we yield from it.
                self.context.logger.info(
                    f"Loading data for agent {agent.agent_instance.agent_id} (@{agent.twitter_username or 'Unknown'})."
                )
                yield from agent.load()  # Assuming agent.load() is a generator function

            if not agent.posts:
                self.context.logger.info(
                    f"No posts found for agent @{agent.twitter_username or 'Unknown'}."
                )
                continue

            # Assuming agent.posts is sorted with the latest post last
            latest_post = agent.posts[-1]
            tweet_id_str = str(latest_post.tweet_id)
            tweet_id_int = latest_post.tweet_id  # Keep as int for set comparison

            # Skip previously interacted tweets
            if tweet_id_int in interacted_tweet_ids:
                self.context.logger.info(
                    f"Tweet {tweet_id_str} from @{agent.twitter_username} was already interacted with"
                )
                continue

            if not agent.twitter_username:
                self.context.logger.warning(
                    f"Agent {agent.agent_instance.agent_id} has a post but no twitter_username. Skipping tweet {tweet_id_str}."
                )
                continue

            pending_tweets[tweet_id_str] = {
                "text": latest_post.text,
                "user_name": agent.twitter_username,
            }
            self.context.logger.info(
                f"Collected pending tweet {tweet_id_str} from @{agent.twitter_username}"
            )
        return pending_tweets

    def _store_engagement_data(
        self, interacted_tweet_ids: list[int], pending_tweets: dict[str, dict[str, str]]
    ) -> Generator[None, None, None]:
        """Store engagement data in the database for mech processing."""
        yield from self._write_kv(
            {
                "interacted_tweet_ids_for_tw_mech": json.dumps(
                    interacted_tweet_ids, sort_keys=True
                )
            }
        )
        self.context.logger.info("Wrote interacted tweet ids to db")

        yield from self._write_kv(
            {"pending_tweets_for_tw_mech": json.dumps(pending_tweets, sort_keys=True)}
        )
        self.context.logger.info("Wrote pending tweets to db")

    def _update_interacted_tweets(
        self, interacted_tweet_ids: list[int], new_interacted_tweet_ids: list[int]
    ) -> Generator[None, None, None]:
        """Update the list of interacted tweets in the database."""
        interacted_tweet_ids.extend(new_interacted_tweet_ids)
        yield from self._write_kv(
            {"interacted_tweet_ids": json.dumps(interacted_tweet_ids, sort_keys=True)}
        )
        self.context.logger.info("Updated interacted tweets in db")

    def interact_twitter(  # pylint: disable=too-many-locals
        self, pending_tweets: dict
    ) -> Generator[None, None, Tuple[str, List, List]]:
        """Decide whether to interact with tweets based on the persona's preferences."""
        new_interacted_tweet_ids: List[int] = []
        persona = yield from self.get_persona()

        # Track retry attempts
        max_retries = 3
        retry_count = 0
        valid_response = False
        json_response = None

        # Try to get the previously stored prompt first
        stored_prompt_data = yield from self._read_kv(keys=("last_prompt",))
        stored_prompt = (
            stored_prompt_data.get("last_prompt") if stored_prompt_data else None
        )

        # Check if we should use a stored prompt or generate a new one
        if stored_prompt and retry_count > 0:
            self.context.logger.info("Using previously stored prompt")
            prompt = stored_prompt
            # We still need previous_tweets for potential media handling
            previous_tweets = yield from self._get_stored_kv_data(
                "previous_tweets_for_tw_mech", {}
            )
        else:
            # Generate a new prompt and store it
            prompt, previous_tweets = yield from self._prepare_prompt_data(
                pending_tweets, persona
            )

        while not valid_response and retry_count < max_retries:
            # Get LLM decision about how to interact with tweets
            llm_response = yield from self._get_llm_decision(prompt)
            if llm_response is None:
                self.context.logger.error("Error getting a response from the LLM.")
                return Event.ERROR.value, new_interacted_tweet_ids, []

            # Parse LLM response
            try:
                json_response = json.loads(llm_response)
                self.context.logger.info(
                    f"LLM response after JSON parsing: {json_response}"
                )

                # Validate response format
                if json_response is not None and self._validate_llm_response(
                    json_response
                ):
                    valid_response = True
                else:
                    retry_count += 1
                    self.context.logger.warning(
                        f"Invalid response format from LLM (attempt {retry_count}/{max_retries})"
                    )
                    # If we need to retry, use the stored prompt
                    if retry_count > 0 and stored_prompt:
                        prompt = stored_prompt

            except json.JSONDecodeError as e:
                self.context.logger.error(f"Error decoding LLM response: {e}")
                self.context.logger.error(f"LLM Response: {llm_response}")
                retry_count += 1
                # If we need to retry, use the stored prompt
                if retry_count > 0 and stored_prompt:
                    prompt = stored_prompt
                continue

        # If we couldn't get a valid response after max retries
        if not valid_response or json_response is None:
            self.context.logger.error(
                f"Failed to get valid response after {max_retries} attempts"
            )
            return Event.ERROR.value, new_interacted_tweet_ids, []

        # At this point, json_response must be valid and not None
        assert json_response is not None, "json_response should not be None here"

        # Handle tool action if present
        if "tool_action" in json_response and json_response["tool_action"] is not None:
            # The validation should have already caught this, but double-check
            if self.synchronized_data.mech_for_twitter:
                self.context.logger.error(
                    "LLM provided a tool action when mech_for_twitter is True. "
                    "This should not happen after our validation."
                )
                return Event.ERROR.value, new_interacted_tweet_ids, []

            # Handle the tool action normally
            event, new_interacted_tweet_id, mech_request = self._handle_tool_action(
                json_response
            )
            return event, new_interacted_tweet_id, mech_request

        # Handle tweet actions if present
        if "tweet_action" in json_response:
            (
                event,
                new_interacted_tweet_id,
                mech_request,
            ) = yield from self._handle_tweet_actions(
                json_response,
                pending_tweets,
                previous_tweets,
                persona,
                new_interacted_tweet_ids,
            )
            return event, new_interacted_tweet_id, mech_request

        # This point should not be reached due to our validation
        self.context.logger.error("Invalid response from the LLM.")
        return Event.ERROR.value, new_interacted_tweet_ids, []

    def _prepare_prompt_data(
        self, pending_tweets: dict, persona: str
    ) -> Generator[None, None, Tuple[str, Optional[List[Dict]]]]:
        """Prepare the prompt data for LLM decision making."""
        if self.synchronized_data.mech_for_twitter:
            prompt, previous_tweets = yield from self._prepare_mech_prompt_data(persona)
        else:
            prompt, previous_tweets = yield from self._prepare_standard_prompt_data(
                pending_tweets, persona
            )

        # saving the prompt to the kv store for retrying if llm response is invalid
        yield from self._write_kv({"last_prompt": prompt})

        return prompt, previous_tweets

    def _prepare_standard_prompt_data(
        self, pending_tweets: dict, persona: str
    ) -> Generator[None, None, Tuple[str, Optional[List[Dict]]]]:
        """Prepare prompt data when mech_for_twitter is False."""
        self.context.logger.info(
            "Standard engagement: using prompt for decision and introducing tools to LLM"
        )

        # Prepare other tweets data
        items_for_formatting: List[Tuple[Any, Any]] = []
        if pending_tweets:
            items_for_formatting = list(
                dict(
                    random.sample(list(pending_tweets.items()), len(pending_tweets))
                ).items()
            )

        other_tweets = "\n\n".join(
            [
                f"tweet_id: {t_id}\ntweet_text: {t_data['text']}\nuser_name: {t_data['user_name']}"
                for t_id, t_data in items_for_formatting
            ]
        )

        # Get previous tweets
        tweets = self.context.agents_fun_db.get_my_agent_posts()

        tweets = tweets[-5:] if tweets else None
        previous_tweets_str = self._format_previous_tweets_str(tweets)

        # Convert TwitterPost objects to dictionaries for storage and return
        previous_tweets_for_return: Optional[List[Dict[str, Any]]] = None
        if tweets:
            if isinstance(tweets[0], TwitterPost):
                previous_tweets_for_return = [
                    {
                        "tweet_id": post.tweet_id,
                        "text": post.text,
                        "timestamp": (
                            post.timestamp.isoformat() if post.timestamp else None
                        ),
                    }
                    for post in tweets
                ]
            elif isinstance(tweets[0], dict):  # Already a list of dicts
                previous_tweets_for_return = tweets
            # else: handle other unexpected types or log error

        self.context.logger.info(f"Previous tweets: {previous_tweets_str}")

        is_staking_kpi_met = self.synchronized_data.is_staking_kpi_met
        # Here we want to make sure that even if mech response is not deilvered we do not skip the tweet action

        # we are providing context of the last prompt to the LLM so that it can use it to create a normal tweet

        if self.synchronized_data.failed_mech:
            self.context.logger.info(
                "It seems like the mech failed to deliver the response forcing agent to create a normal tweet"
            )
            last_prompt = yield from self._read_kv(keys=("last_prompt",))
            ENFORCE_ACTION_COMMAND_FAILED_MECH_SUBPROMPT = (
                ENFORCE_ACTION_COMMAND_FAILED_MECH.format(last_prompt=last_prompt)
            )
            extra_command = ENFORCE_ACTION_COMMAND_FAILED_MECH_SUBPROMPT
            tools_info = ""
        else:
            extra_command = (
                ENFORCE_ACTION_COMMAND if is_staking_kpi_met is False else ""
            )
            tools_info = (self.generate_mech_tool_info(),)  # type: ignore
            tools_info = ""

        prompt = TWITTER_DECISION_PROMPT.format(
            persona=persona,
            previous_tweets=previous_tweets_str,
            other_tweets=other_tweets,
            mech_response="",
            tools=tools_info,
            time=self.get_sync_time_str(),
            extra_command=extra_command,
        )

        # Save data for future mech responses
        yield from self._save_standard_kv_data(
            previous_tweets_for_return, pending_tweets
        )

        return prompt, previous_tweets_for_return

    def _save_standard_kv_data(
        self, tweets: Optional[List[Dict[str, Any]]], pending_tweets: dict
    ) -> Generator[None, None, None]:
        """Save data to KV store for potential future mech responses."""
        self.context.logger.info(
            "Saving standard prompt data (previous tweets, pending tweets) to KV store for potential future mech use"
        )

        # Ensure tweets are JSON serializable dictionaries before writing to KV
        # The `tweets` variable should already be List[Dict] due to prior conversion
        # in _prepare_standard_prompt_data
        serializable_tweets = tweets if tweets is not None else []

        yield from self._write_kv(
            {"previous_tweets_for_tw_mech": json.dumps(serializable_tweets)}
        )
        yield from self._write_kv(
            {"other_tweets_for_tw_mech": json.dumps(pending_tweets)}
        )

    def _prepare_mech_prompt_data(
        self, persona: str
    ) -> Generator[None, None, Tuple[str, Optional[List[Dict]]]]:
        """Prepare prompt data when mech_for_twitter is True."""
        # Read saved data for mech response
        previous_tweets_data = yield from self._get_stored_kv_data(
            "previous_tweets_for_tw_mech", []
        )
        other_tweets_data = yield from self._get_stored_kv_data(
            "other_tweets_for_tw_mech", {}
        )

        # Ensure previous_tweets is Optional[List[Dict]]
        previous_tweets = (
            previous_tweets_data if isinstance(previous_tweets_data, list) else None
        )
        previous_tweets_str = self._format_previous_tweets_str(previous_tweets)

        # Ensure other_tweets is str (formatted string)
        other_tweets_str = (
            "\n\n".join(
                [
                    f"tweet_id: {t_id}\ntweet_text: {t_data['text']}\nuser_name: {t_data['user_name']}"
                    for t_id, t_data in other_tweets_data.items()
                ]
            )
            if isinstance(other_tweets_data, dict)
            else "No other tweets found"
        )

        # Check mech responses (optional, for logging or validation)
        if not self.synchronized_data.mech_responses:
            self.context.logger.error("No mech responses found")

        # Determine media type summary for the prompt
        mech_summary = yield from self._determine_mech_summary()

        # Prepare prompt with mech response summary
        subprompt_with_mech_response = MECH_RESPONSE_SUBPROMPT.format(
            mech_response=mech_summary
        )

        prompt = TWITTER_DECISION_PROMPT.format(
            persona=persona,
            previous_tweets=previous_tweets_str,
            other_tweets=other_tweets_str,
            mech_response=subprompt_with_mech_response,
            tools=self.generate_mech_tool_info(),
            time=self.get_sync_time_str(),
            extra_command="",
        )

        # Clear stored data
        yield from self._clear_mech_kv_data()

        return prompt, previous_tweets

    def _clear_mech_kv_data(self) -> Generator[None, None, None]:
        """Clear KV store entries related to mech twitter interaction."""
        self.context.logger.info(
            "Clearing mech-related twitter data from KV store: previous_tweets, other_tweets, interacted_ids, pending_tweets"
        )
        yield from self._write_kv({"previous_tweets_for_tw_mech": ""})
        yield from self._write_kv({"other_tweets_for_tw_mech": ""})
        yield from self._write_kv({"interacted_tweet_ids_for_tw_mech": ""})
        yield from self._write_kv({"pending_tweets_for_tw_mech": ""})

    def _get_latest_media_info(self) -> Generator[None, None, Optional[Dict]]:
        """Reads and parses the 'latest_media_info' from the KV store."""
        try:
            media_info_data = yield from self._read_kv(keys=("latest_media_info",))
            if (
                not media_info_data
                or "latest_media_info" not in media_info_data
                or not media_info_data["latest_media_info"]
            ):
                self.context.logger.warning(
                    "Could not find valid 'latest_media_info' in KV store."
                )
                return None

            media_info = json.loads(media_info_data["latest_media_info"])
            return media_info
        except json.JSONDecodeError:
            self.context.logger.error(
                "Failed to parse 'latest_media_info' JSON from KV store."
            )
            return None
        except Exception as e:  # pylint: disable=broad-except
            self.context.logger.error(
                f"Error reading or processing 'latest_media_info' from KV: {e}"
            )
            return None

    def _determine_mech_summary(self) -> Generator[None, None, str]:
        """Determine the mech summary string based on media info in KV store."""
        mech_summary = "The mech response processing failed, proceed with tweet_with_media action."  # Default/Fallback
        media_info = yield from self._get_latest_media_info()

        if media_info:
            media_type = media_info.get("type")
            if media_type == "image":
                mech_summary = "The previous tool execution generated an image."
            elif media_type == "video":
                mech_summary = "The previous tool execution generated a video."
            else:
                self.context.logger.warning(
                    f"Found media info in KV store, but type was unexpected: {media_type}"
                )
        # If media_info is None, the error/warning is already logged by _get_latest_media_info

        return mech_summary

    def _get_stored_kv_data(
        self, key: str, default_value: Any
    ) -> Generator[None, None, Any]:
        """Helper to get and parse stored KV data."""
        data = yield from self._read_kv(keys=(key,))
        if not data:
            self.context.logger.error(f"No {key} found in KV store")
            return default_value
        return json.loads(data[key])

    def _get_llm_decision(self, prompt: str) -> Generator[None, None, Optional[str]]:
        """Get decision from LLM."""
        self.context.logger.info(f"Prompting the LLM for a decision: {prompt}")
        llm_response = yield from self._call_genai(
            prompt=prompt,
            schema=build_decision_schema(),
        )
        self.context.logger.info(f"LLM response for twitter decision: {llm_response}")
        return llm_response

    def _validate_mech_llm_response(self, json_response: dict) -> bool:
        """Validate LLM response when mech_for_twitter is True."""
        tweet_action = json_response.get("tweet_action")
        if not isinstance(tweet_action, dict):
            self.context.logger.error(
                "Invalid LLM response: expected tweet_action object when mech_for_twitter is True"
            )
            return False

        # Check both possible field names: action_type (schema format) and action (actual response format)
        action = tweet_action.get("action_type", tweet_action.get("action"))
        if action != "tweet_with_media":
            self.context.logger.error(
                f"Invalid action type: {action}. Only 'tweet_with_media' is allowed when mech_for_twitter is True"
            )
            return False

        if "text" not in tweet_action:
            self.context.logger.error(
                "Invalid tweet_with_media action: missing 'text' field"
            )
            return False

        return True

    def _validate_non_mech_tweet_action(
        self, tweet_action: Union[dict, List[dict]]
    ) -> bool:
        """Validate the tweet_action part of the LLM response when mech_for_twitter is False."""
        actions_to_check = (
            tweet_action if isinstance(tweet_action, list) else [tweet_action]
        )

        for action_item in actions_to_check:
            if not isinstance(action_item, dict):
                # Basic type check for safety, although schema should handle this.
                continue
            action_type = action_item.get("action_type", action_item.get("action"))
            if action_type == "tweet_with_media":
                self.context.logger.error(
                    "Invalid action: 'tweet_with_media' is not allowed when mech_for_twitter is False"
                )
                return False
        return True

    def _validate_non_mech_tool_action(self, tool_action: dict) -> bool:
        """Validate the tool_action part of the LLM response when mech_for_twitter is False."""
        if not isinstance(tool_action, dict):
            self.context.logger.error(
                f"Invalid tool_action format: expected dict, got {type(tool_action)}"
            )
            return False

        if "tool_name" not in tool_action or "tool_input" not in tool_action:
            self.context.logger.error(
                f"Invalid tool_action format: missing required fields. Got: {tool_action}"
            )
            return False
        return True

    def _validate_non_mech_llm_response(self, json_response: dict) -> bool:
        """Validate LLM response when mech_for_twitter is False."""
        tweet_action = json_response.get("tweet_action")
        if tweet_action is not None and self._validate_non_mech_tweet_action(
            tweet_action
        ):
            return True

        tool_action = json_response.get("tool_action")
        if tool_action is not None and self._validate_non_mech_tool_action(tool_action):
            return True

        self.context.logger.error(
            "Invalid LLM response: neither valid tweet_action nor valid tool_action found"
        )
        return False

    def _validate_llm_response(self, json_response: dict) -> bool:
        """Validate that the LLM response adheres to expected format based on context."""
        if not isinstance(json_response, dict):
            self.context.logger.error(
                f"Invalid LLM response format: expected dict, got {type(json_response)}"
            )
            return False

        if self.synchronized_data.mech_for_twitter:
            return self._validate_mech_llm_response(json_response)

        return self._validate_non_mech_llm_response(json_response)

    def _handle_tool_action(self, json_response: dict) -> Tuple[str, List, List]:
        """Handle tool action from LLM response."""
        self.context.logger.info("Tool action detected")

        # Validate that we have both tool_name and tool_input
        if (
            "tool_action" not in json_response
            or not json_response["tool_action"]
            or "tool_name" not in json_response["tool_action"]
            or "tool_input" not in json_response["tool_action"]
        ):
            self.context.logger.error(
                "Invalid tool action: missing tool_name or tool_input"
            )
            return Event.ERROR.value, [], []

        new_mech_requests = []
        nonce = str(uuid4())
        tool_name = json_response["tool_action"]["tool_name"]
        tool_input = json_response["tool_action"]["tool_input"]

        new_mech_requests.append(
            asdict(
                MechMetadata(
                    nonce=nonce,
                    tool=tool_name,
                    prompt=tool_input,
                )
            )
        )

        return Event.MECH.value, [], new_mech_requests

    def _handle_tweet_actions(  # pylint: disable=too-many-arguments
        self,
        json_response: dict,
        pending_tweets: dict,
        previous_tweets: Optional[List[dict]],
        persona: str,
        new_interacted_tweet_ids: List[int],
    ) -> Generator[None, None, Tuple[str, List, List]]:
        """Handle tweet actions from LLM response."""
        self.context.logger.info("Tweet action detected")
        tweet_actions = json_response["tweet_action"]

        # Ensure tweet_actions is a list
        if isinstance(tweet_actions, str):
            tweet_actions = [{"action": tweet_actions}]
        elif not isinstance(tweet_actions, list):
            tweet_actions = [tweet_actions]

        # Create context object
        context = InteractionContext(
            pending_tweets=pending_tweets,
            previous_tweets=previous_tweets,
            persona=persona,
            new_interacted_tweet_ids=new_interacted_tweet_ids,
        )

        for interaction in tweet_actions:
            # Process each interaction
            yield from self._process_single_interaction(
                interaction, context  # Pass the context object
            )

        return Event.DONE.value, context.new_interacted_tweet_ids, []

    def _process_single_interaction(  # pylint: disable=too-many-locals
        self,
        interaction: dict,
        context: InteractionContext,  # Use InteractionContext
    ) -> Generator[None, None, None]:
        """Process a single tweet interaction."""
        # Ensure interaction is a dictionary
        if not isinstance(interaction, dict):
            self.context.logger.error(f"Invalid interaction format: {interaction}")
            return

        tweet_id = interaction.get("selected_tweet_id", None)
        user_name = interaction.get("user_name", None)
        action = interaction.get("action", None)
        text = interaction.get("text", None)

        # Validate action and parameters (using context)
        if not self._validate_interaction(
            action, tweet_id, user_name, context.pending_tweets
        ):
            return

        # Add random delay to avoid rate limiting
        delay = secrets.randbelow(5)
        self.context.logger.info(f"Sleeping for {delay} seconds")
        yield from self.sleep(delay)

        # Handle tweet action based on type
        if action == "tweet":
            yield from self._handle_new_tweet(
                text, context.previous_tweets, context.persona
            )
        elif action == "tweet_with_media":
            # Delegate to the new handler
            success = yield from self._handle_media_tweet(text)
            if not success:
                self.context.logger.error("Failed to handle tweet_with_media action.")
                # Potentially return or handle error differently if needed
        else:
            yield from self._handle_tweet_interaction(
                action,
                tweet_id,
                text,
                user_name,
                context.pending_tweets,  # Pass needed context items
                context.new_interacted_tweet_ids,
            )

    def _handle_media_tweet(self, text: str) -> Generator[None, None, bool]:
        """Handles the 'tweet_with_media' action, including fetching, uploading, posting, and clearing KV."""
        # Read the combined media info from kv store
        media_info = yield from self._get_latest_media_info()

        if not media_info:
            # Error already logged by helper
            return False  # Indicate failure

        media_path = media_info.get("path")
        media_type = media_info.get("type")

        if not media_path or not media_type:
            self.context.logger.error(
                "Media info from KV store is missing 'path' or 'type'."
            )
            # Clear potentially incomplete info? Or leave it? Let's clear it for safety.
            yield from self._write_kv({"latest_media_info": ""})
            return False  # Indicate failure

        self.context.logger.info(
            f"Extracted media path: {media_path}, type: {media_type}"
        )

        # Clear the media info from KV store *after* successfully extracting path and type
        yield from self._write_kv({"latest_media_info": ""})
        self.context.logger.info("Cleared latest_media_info from KV store.")

        # Ensure media_path is a valid string path (already checked by .get implicitly somewhat)
        if not isinstance(media_path, str):
            self.context.logger.error(f"Invalid media path type: {type(media_path)}")
            return False  # Indicate failure

        # Post tweet with the uploaded media ID
        tweet_ids = yield from self._call_tweepy(
            method="post", tweets=[{"text": text, "image_paths": [media_path]}]
        )

        if not tweet_ids:
            self.context.logger.error("Failed posting tweet with media to Twitter.")
            return False  # Indicate failure

        latest_tweet = {
            "tweet_id": tweet_ids[0],
            "text": text,
            "media_path": media_path,
            "timestamp": self.get_sync_timestamp(),
        }

        # Write latest tweet to the database
        yield from self.store_tweet(latest_tweet)
        self.context.logger.info("Wrote latest tweet with media to db")
        return True  # Indicate success

    def _validate_interaction(
        self, action: str, tweet_id: str, user_name: str, pending_tweets: dict
    ) -> bool:
        """Validate tweet interaction parameters."""
        if action == "none":
            self.context.logger.error("Action is none")
            return False

        # Treat tweet_with_media like tweet - it doesn't need a tweet_id
        # Also, 'follow' action does not need a tweet_id for this specific check.
        if (
            action not in ["tweet", "tweet_with_media", "follow"]
            and str(tweet_id) not in pending_tweets.keys()
        ):
            self.context.logger.error(
                f"Action is {action} but tweet_id is not valid [{tweet_id}]"
            )
            return False

        if action == "follow" and (
            not user_name
            or user_name not in [t["user_name"] for t in pending_tweets.values()]
        ):
            self.context.logger.error(
                f"Action is {action} but user_name is not valid [{user_name}]"
            )
            return False

        return True

    def _handle_new_tweet(
        self, text: str, previous_tweets: Optional[List[Dict]], persona: str
    ) -> Generator[None, None, None]:
        """Handle creating a new tweet."""
        # Format the previous tweets list into a string for the prompt
        previous_tweets_str = self._format_previous_tweets_str(previous_tweets)

        # Optionally, replace the tweet with one generated by the alternative model
        new_text = yield from self.replace_tweet_with_alternative_model(
            ALTERNATIVE_MODEL_TWITTER_PROMPT.format(
                persona=persona,
                previous_tweets=previous_tweets_str,
            )
        )
        text = new_text or text

        if not is_tweet_valid(text):
            self.context.logger.error("The tweet is too long.")
            return

        yield from self.post_tweet(tweet=[text], store=True)

    def _handle_tweet_interaction(  # pylint: disable=too-many-arguments
        self,
        action: str,
        tweet_id: Optional[str],  # Can be None for follow
        text: Optional[str],
        user_name: Optional[str],  # Target for follow
        pending_tweets: dict,
        new_interacted_tweet_ids: List[int],
    ) -> Generator[None, None, None]:
        """Handle interaction with an existing tweet or user."""
        if text and not is_tweet_valid(text):
            self.context.logger.error("The tweet is too long.")
            return

        user_name_for_quote: Optional[str] = None

        if action == "follow":
            if not user_name:  # Safeguard, should be validated by _validate_interaction
                self.context.logger.error(
                    "Follow action initiated but no user_name provided."
                )
                return
            self.context.logger.info(f"Trying to {action} user {user_name}")
        else:  # For like, retweet, reply, quote
            if (
                tweet_id is None
            ):  # Safeguard, should be validated by _validate_interaction
                self.context.logger.error(
                    f"Action {action} initiated but no tweet_id provided."
                )
                return

            self.context.logger.info(f"Trying to {action} tweet {tweet_id}")
            str_tweet_id = str(tweet_id)  # tweet_id is not None here

            if str_tweet_id not in pending_tweets:
                self.context.logger.error(
                    f"Tweet ID {tweet_id} not found in pending tweets for {action} interaction."
                )
                return

            if action == "quote":
                user_name_for_quote = pending_tweets[str_tweet_id].get("user_name")
                if not user_name_for_quote:
                    self.context.logger.error(
                        f"User name for tweet {tweet_id} not found in pending_tweets, required for quote."
                    )
                    return

        success = False
        if action == "like":
            # tweet_id is guaranteed to be non-None here by the checks above for non-follow actions
            success = yield from self.like_tweet(tweet_id)  # type: ignore
        elif action == "follow" and user_name:
            success = yield from self.follow_user(user_name)
        elif action == "retweet":
            # tweet_id is guaranteed to be non-None here
            success = yield from self.retweet(tweet_id)  # type: ignore
        elif action == "reply":
            # tweet_id is guaranteed to be non-None here
            success = yield from self.respond_tweet(tweet_id, text)  # type: ignore
        elif action == "quote":
            # tweet_id and user_name_for_quote are guaranteed to be non-None if we reach here
            success = yield from self.respond_tweet(
                tweet_id, text, quote=True, user_name=user_name_for_quote  # type: ignore
            )

        if success and action != "follow" and tweet_id is not None:
            new_interacted_tweet_ids.append(int(tweet_id))

    def generate_mech_tool_info(self) -> str:
        """Generate tool info"""

        tools_dict = self.params.tools_for_mech
        tools_info = "\n" + "\n".join(
            [
                f"- {tool_name}: {tool_description}"
                for tool_name, tool_description in tools_dict.items()
            ]
        )
        self.context.logger.info(tools_info)
        return tools_info

    def get_agent_handles(self) -> Generator[None, None, List["AgentsFunAgent"]]:  # type: ignore
        """Get the agent handles from agents_fun_db."""
        # Forward type reference for AgentsFunAgent
        active_agents = yield self.context.agents_fun_db.get_active_agents()
        if not active_agents:
            self.context.logger.info(
                "No active memeooorr handles from agents_fun_db Now trying subgraph."
            )
            # fallback to subgraph
            active_agents = yield from self.get_memeooorr_handles_from_subgraph()

        # TODO: filter out suspended accounts (if AgentsFunAgent has a status field)
        return active_agents


class ActionTweetBehaviour(BaseTweetBehaviour):  # pylint: disable=too-many-ancestors
    """ActionTweetBehaviour"""

    matching_round: Type[AbstractRound] = ActionTweetRound

    def async_act(self) -> Generator:
        """Do the act, supporting asynchronous execution."""

        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            event = yield from self.get_event()

            payload = ActionTweetPayload(
                sender=self.context.agent_address,
                event=event,
            )

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()

    def get_event(self) -> Generator[None, None, str]:
        """Get the next event"""
        pending_tweet = self.synchronized_data.token_action["tweet"]
        if not pending_tweet:
            self.context.logger.info("Post-action tweet is missing")
            return Event.MISSING_TWEET.value
        self.context.logger.info("Sending the action tweet...")
        latest_tweet = yield from self.post_tweet(tweet=[pending_tweet], store=False)
        return Event.DONE.value if latest_tweet else Event.ERROR.value
