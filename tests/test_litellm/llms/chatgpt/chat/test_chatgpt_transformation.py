"""
Tests for ChatGPT subscription Chat Completions transformation.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../../../../.."))

from litellm.llms.chatgpt.chat.transformation import ChatGPTConfig


def test_chatgpt_chat_request_rewrites_system_messages_to_developer():
    config = ChatGPTConfig()
    messages = [
        {"role": "system", "content": "Follow policy."},
        {"role": "user", "content": "Hello"},
    ]

    request = config.transform_request(
        model="gpt-5.3-chat-latest",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )

    assert request["messages"][0]["role"] == "developer"
    assert request["messages"][0]["content"] == "Follow policy."
    assert request["messages"][1] == {"role": "user", "content": "Hello"}
    assert messages[0]["role"] == "system"
