"""One serializer for PicoLLM SFT targets and generation prefixes."""
from tokenizer import BOS, EOS, PAD, IGNORE

CHAT_TEMPLATE = {"format_version": 1, "style": "role-lines-v1",
                 "roles": ["system", "user", "assistant"],
                 "header_prefix": "### ", "header_suffix": "\n", "message_suffix": "\n",
                 "bos_token_id": BOS, "eos_token_id": EOS, "pad_token_id": PAD}


def serialize(messages, generation=False):
    if not messages:
        raise ValueError("A conversation cannot be empty")
    ids, supervised, expected = [BOS], [False], "user"
    for index, message in enumerate(messages):
        role, content = message["role"], message["content"]
        if not isinstance(content, str):
            raise ValueError("Content must be text")
        if role == "system":
            if index != 0:
                raise ValueError("System message must be first")
        elif role == expected:
            expected = "assistant" if role == "user" else "user"
        else:
            raise ValueError("Expected alternating user/assistant roles")
        header = list(("### " + role + "\n").encode("utf-8"))
        body = list((content + "\n").encode("utf-8"))
        ids.extend(header + body)
        supervised.extend([False] * len(header) + [role == "assistant"] * len(body))
        if role == "assistant":
            ids.append(EOS)
            supervised.append(True)
    if generation:
        if messages[-1]["role"] != "user":
            raise ValueError("Generation history must end in a user message")
        header = list(b"### assistant\n")
        ids.extend(header)
        supervised.extend([False] * len(header))
    elif messages[-1]["role"] != "assistant":
        raise ValueError("Training conversation must end in an assistant answer")
    labels = [token if active else IGNORE for token, active in zip(ids[1:], supervised[1:])]
    return ids, labels
