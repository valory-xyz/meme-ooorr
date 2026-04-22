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

"""Tests for prompts.py."""

# pylint: disable=too-few-public-methods

import pickle  # nosec

from packages.valory.skills.memeooorr_abci.prompts import (
    ALTERNATIVE_MODEL_TOKEN_PROMPT,
    ALTERNATIVE_MODEL_TWITTER_PROMPT,
    CHATUI_PROMPT,
    CHATUI_PROMPT_NO_MEMECOIN,
    Decision,
    MECH_RESPONSE_SUBPROMPT,
    ONLY_PERSONA_UPDATE_PROMPT,
    PersonaAction,
    SUMMON_TOKEN_ACTION,
    TOKEN_DECISION_PROMPT,
    TWITTER_DECISION_PROMPT,
    TokenAction,
    ToolAction,
    TwitterAction,
    UpdatedAgentConfig,
    UpdatedAgentConfigNoMemecoin,
    build_decision_schema,
    build_persona_action_schema,
    build_token_action_schema,
    build_tool_action_schema,
    build_twitter_action_schema,
    build_updated_agent_config_schema,
    build_updated_agent_config_schema_no_memecoin,
)


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
