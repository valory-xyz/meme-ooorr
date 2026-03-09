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

"""Tests for prompts.py."""

import pickle  # nosec

import pytest
from pydantic import BaseModel

from packages.dvilela.skills.memeooorr_abci.prompts import (
    ALTERNATIVE_MODEL_TOKEN_PROMPT,
    ALTERNATIVE_MODEL_TWITTER_PROMPT,
    CHATUI_PROMPT,
    CHATUI_PROMPT_NO_MEMECOIN,
    Decision,
    ENFORCE_ACTION_COMMAND,
    ENFORCE_ACTION_COMMAND_FAILED_MECH,
    MECH_RESPONSE_SUBPROMPT,
    ONLY_PERSONA_UPDATE_PROMPT,
    PersonaAction,
    SUMMON_TOKEN_ACTION,
    TOKEN_DECISION_PROMPT,
    TWITTER_DECISION_PROMPT,
    TokenAction,
    TokenCollect,
    TokenHeart,
    TokenPurge,
    TokenSummon,
    TokenUnleash,
    ToolAction,
    ToolActionName,
    TwitterAction,
    TwitterActionName,
    UpdatedAgentConfig,
    UpdatedAgentConfigNoMemecoin,
    ValidActionName,
    build_decision_schema,
    build_persona_action_schema,
    build_token_action_schema,
    build_tool_action_schema,
    build_twitter_action_schema,
    build_updated_agent_config_schema,
    build_updated_agent_config_schema_no_memecoin,
)


class TestTwitterActionName:
    """Tests for TwitterActionName enum."""

    def test_values(self) -> None:
        """Test all enum values."""
        assert TwitterActionName.NONE.value == "none"
        assert TwitterActionName.TWEET.value == "tweet"
        assert TwitterActionName.LIKE.value == "like"
        assert TwitterActionName.RETWEET.value == "retweet"
        assert TwitterActionName.REPLY.value == "reply"
        assert TwitterActionName.QUOTE.value == "quote"
        assert TwitterActionName.FOLLOW.value == "follow"
        assert TwitterActionName.TWEET_WITH_MEDIA.value == "tweet_with_media"

    def test_member_count(self) -> None:
        """Test number of members."""
        assert len(TwitterActionName) == 8


class TestToolActionName:
    """Tests for ToolActionName enum."""

    def test_values(self) -> None:
        """Test all enum values."""
        assert ToolActionName.GOOGLE_IMAGE_GEN.value == "google_image_gen"
        assert ToolActionName.SHORT_MAKER.value == "short_maker"

    def test_member_count(self) -> None:
        """Test number of members."""
        assert len(ToolActionName) == 2


class TestValidActionName:
    """Tests for ValidActionName enum."""

    def test_values(self) -> None:
        """Test all enum values."""
        assert ValidActionName.NONE.value == "none"
        assert ValidActionName.SUMMON.value == "summon"
        assert ValidActionName.HEART.value == "heart"
        assert ValidActionName.UNLEASH.value == "unleash"
        assert ValidActionName.COLLECT.value == "collect"
        assert ValidActionName.PURGE.value == "purge"
        assert ValidActionName.BURN.value == "burn"

    def test_member_count(self) -> None:
        """Test number of members."""
        assert len(ValidActionName) == 7


class TestTwitterAction:
    """Tests for TwitterAction model."""

    def test_construction(self) -> None:
        """Test model construction."""
        action = TwitterAction(
            action=TwitterActionName.TWEET,
            selected_tweet_id="123",
            user_name="test_user",
            text="Hello world",
        )
        assert action.action == TwitterActionName.TWEET
        assert action.selected_tweet_id == "123"
        assert action.user_name == "test_user"
        assert action.text == "Hello world"

    def test_is_base_model(self) -> None:
        """Test it is a BaseModel."""
        assert issubclass(TwitterAction, BaseModel)


class TestToolAction:
    """Tests for ToolAction model."""

    def test_construction(self) -> None:
        """Test model construction."""
        action = ToolAction(
            tool_name=ToolActionName.GOOGLE_IMAGE_GEN,
            tool_input="generate a cat image",
        )
        assert action.tool_name == ToolActionName.GOOGLE_IMAGE_GEN
        assert action.tool_input == "generate a cat image"

    def test_is_base_model(self) -> None:
        """Test it is a BaseModel."""
        assert issubclass(ToolAction, BaseModel)


class TestDecision:
    """Tests for Decision model."""

    def test_construction_with_tool(self) -> None:
        """Test with tool action."""
        tool = ToolAction(tool_name=ToolActionName.GOOGLE_IMAGE_GEN, tool_input="input")
        decision = Decision(tool_action=tool, tweet_action=None)
        assert decision.tool_action == tool
        assert decision.tweet_action is None

    def test_construction_with_tweet(self) -> None:
        """Test with tweet action."""
        tweet = TwitterAction(
            action=TwitterActionName.TWEET,
            selected_tweet_id="1",
            user_name="user",
            text="text",
        )
        decision = Decision(tool_action=None, tweet_action=tweet)
        assert decision.tool_action is None
        assert decision.tweet_action == tweet

    def test_construction_both_none(self) -> None:
        """Test with both None."""
        decision = Decision(tool_action=None, tweet_action=None)
        assert decision.tool_action is None
        assert decision.tweet_action is None

    def test_is_base_model(self) -> None:
        """Test it is a BaseModel."""
        assert issubclass(Decision, BaseModel)


class TestTokenSummon:
    """Tests for TokenSummon model."""

    def test_construction(self) -> None:
        """Test model construction."""
        summon = TokenSummon(
            token_name="TestCoin",
            token_ticker="TC",
            token_supply=1000000,
            amount=100,
        )
        assert summon.token_name == "TestCoin"
        assert summon.token_ticker == "TC"
        assert summon.token_supply == 1000000
        assert summon.amount == 100


class TestTokenHeart:
    """Tests for TokenHeart model."""

    def test_construction(self) -> None:
        """Test model construction."""
        heart = TokenHeart(token_nonce="1", amount=50)
        assert heart.token_nonce == "1"
        assert heart.amount == 50


class TestTokenUnleash:
    """Tests for TokenUnleash model."""

    def test_construction(self) -> None:
        """Test model construction."""
        unleash = TokenUnleash(token_nonce="1")
        assert unleash.token_nonce == "1"


class TestTokenCollect:
    """Tests for TokenCollect model."""

    def test_construction(self) -> None:
        """Test model construction."""
        collect = TokenCollect(token_nonce="1", token_address="0xaddr")
        assert collect.token_nonce == "1"
        assert collect.token_address == "0xaddr"


class TestTokenPurge:
    """Tests for TokenPurge model."""

    def test_construction(self) -> None:
        """Test model construction."""
        purge = TokenPurge(token_nonce="1", token_address="0xaddr")
        assert purge.token_nonce == "1"
        assert purge.token_address == "0xaddr"


class TestTokenAction:
    """Tests for TokenAction model."""

    def test_construction_summon(self) -> None:
        """Test construction with summon action."""
        summon = TokenSummon(
            token_name="TC", token_ticker="TC", token_supply=1000, amount=10
        )
        action = TokenAction(
            action_name=ValidActionName.SUMMON,
            summon=summon,
            heart=None,
            unleash=None,
            collect=None,
            purge=None,
            new_persona=None,
            action_tweet="Summoning!",
        )
        assert action.action_name == ValidActionName.SUMMON
        assert action.summon == summon
        assert action.heart is None
        assert action.action_tweet == "Summoning!"

    def test_construction_none_action(self) -> None:
        """Test construction with none action."""
        action = TokenAction(
            action_name=ValidActionName.NONE,
            summon=None,
            heart=None,
            unleash=None,
            collect=None,
            purge=None,
            new_persona="new persona",
            action_tweet=None,
        )
        assert action.action_name == ValidActionName.NONE
        assert action.new_persona == "new persona"

    def test_is_base_model(self) -> None:
        """Test it is a BaseModel."""
        assert issubclass(TokenAction, BaseModel)


class TestPersonaAction:
    """Tests for PersonaAction model."""

    def test_construction_with_persona(self) -> None:
        """Test with new persona."""
        action = PersonaAction(new_persona="updated persona")
        assert action.new_persona == "updated persona"

    def test_construction_none(self) -> None:
        """Test with None persona."""
        action = PersonaAction(new_persona=None)
        assert action.new_persona is None

    def test_is_base_model(self) -> None:
        """Test it is a BaseModel."""
        assert issubclass(PersonaAction, BaseModel)


class TestUpdatedAgentConfig:
    """Tests for UpdatedAgentConfig model."""

    def test_construction(self) -> None:
        """Test model construction."""
        config = UpdatedAgentConfig(
            agent_persona="new persona",
            heart_cooldown_hours=48,
            summon_cooldown_seconds=3600,
            message="Updated successfully",
        )
        assert config.agent_persona == "new persona"
        assert config.heart_cooldown_hours == 48
        assert config.summon_cooldown_seconds == 3600
        assert config.message == "Updated successfully"

    def test_construction_none_fields(self) -> None:
        """Test with None optional fields."""
        config = UpdatedAgentConfig(
            agent_persona=None,
            heart_cooldown_hours=None,
            summon_cooldown_seconds=None,
            message="No changes",
        )
        assert config.agent_persona is None
        assert config.heart_cooldown_hours is None
        assert config.summon_cooldown_seconds is None

    def test_is_base_model(self) -> None:
        """Test it is a BaseModel."""
        assert issubclass(UpdatedAgentConfig, BaseModel)


class TestUpdatedAgentConfigNoMemecoin:
    """Tests for UpdatedAgentConfigNoMemecoin model."""

    def test_construction(self) -> None:
        """Test model construction."""
        config = UpdatedAgentConfigNoMemecoin(
            agent_persona="new persona", message="Updated"
        )
        assert config.agent_persona == "new persona"
        assert config.message == "Updated"

    def test_construction_none_persona(self) -> None:
        """Test with None persona."""
        config = UpdatedAgentConfigNoMemecoin(agent_persona=None, message="No change")
        assert config.agent_persona is None

    def test_is_base_model(self) -> None:
        """Test it is a BaseModel."""
        assert issubclass(UpdatedAgentConfigNoMemecoin, BaseModel)


class TestSchemaBuilders:
    """Tests for schema builder functions."""

    def test_build_twitter_action_schema(self) -> None:
        """Test build_twitter_action_schema returns valid schema."""
        schema = build_twitter_action_schema()
        assert "class" in schema
        assert "is_list" in schema
        assert schema["is_list"] is True
        # Verify we can deserialize the class back
        cls = pickle.loads(bytes.fromhex(schema["class"]))  # nosec
        assert cls is TwitterAction

    def test_build_tool_action_schema(self) -> None:
        """Test build_tool_action_schema returns valid schema."""
        schema = build_tool_action_schema()
        assert "class" in schema
        assert "is_list" in schema
        assert schema["is_list"] is False
        cls = pickle.loads(bytes.fromhex(schema["class"]))  # nosec
        assert cls is ToolAction

    def test_build_decision_schema(self) -> None:
        """Test build_decision_schema returns valid schema."""
        schema = build_decision_schema()
        assert "class" in schema
        assert "is_list" in schema
        assert schema["is_list"] is False
        cls = pickle.loads(bytes.fromhex(schema["class"]))  # nosec
        assert cls is Decision

    def test_build_token_action_schema(self) -> None:
        """Test build_token_action_schema returns valid schema."""
        schema = build_token_action_schema()
        assert "class" in schema
        assert "is_list" in schema
        assert schema["is_list"] is False
        cls = pickle.loads(bytes.fromhex(schema["class"]))  # nosec
        assert cls is TokenAction

    def test_build_persona_action_schema(self) -> None:
        """Test build_persona_action_schema returns valid schema."""
        schema = build_persona_action_schema()
        assert "class" in schema
        assert "is_list" in schema
        assert schema["is_list"] is False
        cls = pickle.loads(bytes.fromhex(schema["class"]))  # nosec
        assert cls is PersonaAction

    def test_build_updated_agent_config_schema(self) -> None:
        """Test build_updated_agent_config_schema returns valid schema."""
        schema = build_updated_agent_config_schema()
        assert "class" in schema
        assert "is_list" in schema
        assert schema["is_list"] is False
        cls = pickle.loads(bytes.fromhex(schema["class"]))  # nosec
        assert cls is UpdatedAgentConfig

    def test_build_updated_agent_config_schema_no_memecoin(self) -> None:
        """Test build_updated_agent_config_schema_no_memecoin returns valid schema."""
        schema = build_updated_agent_config_schema_no_memecoin()
        assert "class" in schema
        assert "is_list" in schema
        assert schema["is_list"] is False
        cls = pickle.loads(bytes.fromhex(schema["class"]))  # nosec
        assert cls is UpdatedAgentConfigNoMemecoin


class TestPromptStrings:
    """Tests for prompt string constants."""

    def test_enforce_action_command_is_string(self) -> None:
        """Test ENFORCE_ACTION_COMMAND is a non-empty string."""
        assert isinstance(ENFORCE_ACTION_COMMAND, str)
        assert len(ENFORCE_ACTION_COMMAND) > 0

    def test_enforce_action_command_failed_mech_is_string(self) -> None:
        """Test ENFORCE_ACTION_COMMAND_FAILED_MECH is a non-empty string."""
        assert isinstance(ENFORCE_ACTION_COMMAND_FAILED_MECH, str)
        assert len(ENFORCE_ACTION_COMMAND_FAILED_MECH) > 0
        assert "{last_prompt}" in ENFORCE_ACTION_COMMAND_FAILED_MECH

    def test_twitter_decision_prompt_has_placeholders(self) -> None:
        """Test TWITTER_DECISION_PROMPT has expected placeholders."""
        assert "{persona}" in TWITTER_DECISION_PROMPT
        assert "{tools}" in TWITTER_DECISION_PROMPT
        assert "{twitter_actions}" in TWITTER_DECISION_PROMPT
        assert "{previous_tweets}" in TWITTER_DECISION_PROMPT
        assert "{other_tweets}" in TWITTER_DECISION_PROMPT
        assert "{time}" in TWITTER_DECISION_PROMPT
        assert "{extra_command}" in TWITTER_DECISION_PROMPT
        assert "{mech_response}" in TWITTER_DECISION_PROMPT
        assert "{tweet_actions}" in TWITTER_DECISION_PROMPT
        assert "{tool_actions}" in TWITTER_DECISION_PROMPT

    def test_mech_response_subprompt_has_placeholder(self) -> None:
        """Test MECH_RESPONSE_SUBPROMPT has expected placeholder."""
        assert "{mech_response}" in MECH_RESPONSE_SUBPROMPT

    def test_alternative_model_twitter_prompt_has_placeholders(self) -> None:
        """Test ALTERNATIVE_MODEL_TWITTER_PROMPT has expected placeholders."""
        assert "{persona}" in ALTERNATIVE_MODEL_TWITTER_PROMPT
        assert "{previous_tweets}" in ALTERNATIVE_MODEL_TWITTER_PROMPT

    def test_token_decision_prompt_has_placeholders(self) -> None:
        """Test TOKEN_DECISION_PROMPT has expected placeholders."""
        assert "{summon_token_action}" in TOKEN_DECISION_PROMPT
        assert "{meme_coins}" in TOKEN_DECISION_PROMPT
        assert "{latest_tweet}" in TOKEN_DECISION_PROMPT
        assert "{tweet_responses}" in TOKEN_DECISION_PROMPT
        assert "{balance}" in TOKEN_DECISION_PROMPT
        assert "{ticker}" in TOKEN_DECISION_PROMPT

    def test_summon_token_action_is_string(self) -> None:
        """Test SUMMON_TOKEN_ACTION is a non-empty string."""
        assert isinstance(SUMMON_TOKEN_ACTION, str)
        assert "summon" in SUMMON_TOKEN_ACTION

    def test_only_persona_update_prompt_has_placeholders(self) -> None:
        """Test ONLY_PERSONA_UPDATE_PROMPT has expected placeholders."""
        assert "{latest_tweet}" in ONLY_PERSONA_UPDATE_PROMPT
        assert "{tweet_responses}" in ONLY_PERSONA_UPDATE_PROMPT

    def test_alternative_model_token_prompt_has_placeholders(self) -> None:
        """Test ALTERNATIVE_MODEL_TOKEN_PROMPT has expected placeholders."""
        assert "{persona}" in ALTERNATIVE_MODEL_TOKEN_PROMPT
        assert "{meme_coins}" in ALTERNATIVE_MODEL_TOKEN_PROMPT
        assert "{action}" in ALTERNATIVE_MODEL_TOKEN_PROMPT
        assert "{summon_token_action}" in ALTERNATIVE_MODEL_TOKEN_PROMPT

    def test_chatui_prompt_has_placeholders(self) -> None:
        """Test CHATUI_PROMPT has expected placeholders."""
        assert "{current_persona}" in CHATUI_PROMPT
        assert "{current_heart_cooldown_hours}" in CHATUI_PROMPT
        assert "{current_summon_cooldown_seconds}" in CHATUI_PROMPT
        assert "{user_prompt}" in CHATUI_PROMPT

    def test_chatui_prompt_no_memecoin_has_placeholders(self) -> None:
        """Test CHATUI_PROMPT_NO_MEMECOIN has expected placeholders."""
        assert "{current_persona}" in CHATUI_PROMPT_NO_MEMECOIN
        assert "{user_prompt}" in CHATUI_PROMPT_NO_MEMECOIN
