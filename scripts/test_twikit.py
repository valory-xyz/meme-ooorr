#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2021-2025 Valory AG
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

"""Test twikit"""


import asyncio
import os
import random
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import dotenv
import twikit


dotenv.load_dotenv(override=True)


def is_twitter_id(twitter_id: str) -> bool:
    """Check if a string is a valid Twitter ID."""
    return bool(re.match(r"^\d{1,20}$", twitter_id))


def tweet_to_json(tweet: Any) -> Dict:
    """Tweet to json"""
    return {
        "id": tweet.id,
        "user_name": tweet.user.name,
        "text": tweet.text,
        "created_at": tweet.created_at,
        "view_count": tweet.view_count,
        "retweet_count": tweet.retweet_count,
        "quote_count": tweet.quote_count,
        "view_count_state": tweet.view_count_state,
    }


async def password_login():
    """Login via password"""
    client = twikit.Client(language="en-US")
    await client.login(
        auth_info_1=os.getenv("TWIKIT_USERNAME"),
        auth_info_2=os.getenv("TWIKIT_EMAIL"),
        password=os.getenv("TWIKIT_PASSWORD"),
    )
    client.save_cookies(os.getenv("TWIKIT_COOKIES_PATH"))
    return client


async def cookie_login():
    """Login via cookie"""

    client = twikit.Client(language="en-US")
    await client.login(
        auth_info_1=os.getenv("TWIKIT_USERNAME"),
        auth_info_2=os.getenv("TWIKIT_EMAIL"),
        password=os.getenv("TWIKIT_PASSWORD"),
        cookies_file=os.getenv("TWIKIT_COOKIES_PATH"),
    )
    return client


async def validate_login() -> None:
    """Stress test"""
    client = await cookie_login()
    user = await client.get_user_by_screen_name("autonolas")
    if user.id != "1450081635559428107":
        print("Error")
    else:
        print("OK")


async def get_tweets(client) -> None:
    """Get tweets"""

    user = await client.get_user_by_screen_name(screen_name="autonolas")

    latest_tweets = await client.get_user_tweets(
        user_id=user.id, tweet_type="Tweets", count=1
    )

    return [tweet_to_json(t) for t in latest_tweets]


async def stress_test() -> None:
    """Stress test"""
    client = await cookie_login()
    while True:
        print("Getting tweets")
        tweets = await get_tweets(client)
        print(tweets)
        time.sleep(random.randint(30, 60))  # nosec


async def like_tweet() -> None:
    """Like tweet"""
    client = await cookie_login()
    await client.favorite_tweet("1868952161427882053")


async def create_tweet() -> None:
    """Like tweet"""
    client = await cookie_login()
    await client.create_tweet("Hello world!")


async def is_suspended() -> None:
    """Is suspended"""
    client = await cookie_login()
    try:
        await client.get_user_by_screen_name("autonolas")
        return False
    except twikit.errors.UserUnavailable:
        return True


async def search_tweet() -> None:
    """Search tweet"""
    client = await cookie_login()
    tweets = await client.search_tweet(query="$OLAS", product="Top", count=5)
    return tweets


async def get_followers_ids(user: str) -> Optional[List[str]]:
    """Get follower ids"""
    client = await cookie_login()

    if is_twitter_id(user):
        user_id = user
    else:
        try:
            user_obj = await client.get_user_by_screen_name(user)
        except twikit.errors.UserUnavailable:
            return None
        user_id = user_obj.id

    follower_ids = await client.get_followers_ids(user_id=user_id, count=5000)
    return follower_ids


async def get_tweet_engagement_rate(tweet, client=None) -> Optional[float]:
    """Get tweet engagement rate"""
    if client is None:
        client = await cookie_login()

    engagements = (
        tweet.favorite_count
        + tweet.retweet_count
        + tweet.quote_count
        + tweet.reply_count
    )
    impressions = int(tweet.view_count) if tweet.view_count else 0

    return engagements / impressions if impressions > 0 else 0


async def get_latest_user_tweets(user: str, client=None) -> List[Dict[str, Any]]:
    """Get user tweets"""
    if client is None:
        client = await cookie_login()

    if is_twitter_id(user):
        user_id = user
    else:
        try:
            user_obj = await client.get_user_by_screen_name(user)
        except twikit.errors.UserUnavailable:
            return None
        user_id = user_obj.id

    try:
        tweets = await client.get_user_tweets(
            user_id=user_id, tweet_type="Tweets", count=50
        )
    except KeyError:
        print("User might be restricted or banned")
        return None

    now = datetime.now(timezone.utc)
    this_week_tweets = [
        tweet
        for tweet in tweets
        if now - datetime.strptime(tweet.created_at, "%a %b %d %H:%M:%S %z %Y")
        <= timedelta(days=7)
    ]

    return this_week_tweets


async def calculate_user_engagement(user: str) -> float:
    """Calculate user engagement"""
    client = await cookie_login()
    tweets = await get_latest_user_tweets(user=user, client=client)

    if not tweets:
        print(f"No tweets found for {user}")
        return None

    print(f"Analizing {len(tweets)} tweets from {user}")
    engagements = {}
    for tweet in tweets:
        eng_rate = await get_tweet_engagement_rate(tweet, client)
        engagements[tweet.id] = eng_rate
        print(f"Tweet {tweet.id} engagement rate: {eng_rate:.2f}")
        time.sleep(1)

    return sum(engagements.values()) / len(engagements) if engagements else 0


if __name__ == "__main__":
    # print(asyncio.run(validate_login()))
    print(asyncio.run(get_latest_user_tweets("karab_olas1")))
