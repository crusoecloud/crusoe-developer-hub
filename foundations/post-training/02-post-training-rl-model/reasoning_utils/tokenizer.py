"""A thin wrapper around the Qwen3 tokenizer file.

The heavy lifting (byte-pair encoding over a 151k vocabulary) is done by the
`tokenizers` library. This wrapper adds the two things the lessons need: the
end-of-sequence id so generation knows when to stop, and the chat template
the post-trained model expects.
"""

from __future__ import annotations

from pathlib import Path


class Qwen3Tokenizer:
    def __init__(self, tokenizer_file: str | Path, chat_template: bool = False, thinking: bool = True):
        from tokenizers import Tokenizer

        self._tok = Tokenizer.from_file(str(tokenizer_file))
        self.chat_template = chat_template
        self.thinking = thinking

        self.pad_token_id = self._tok.token_to_id("<|endoftext|>")
        # The base model was trained on raw documents separated by <|endoftext|>.
        # The post-trained model closes each chat turn with <|im_end|>.
        self.eos_token = "<|im_end|>" if chat_template else "<|endoftext|>"
        self.eos_token_id = self._tok.token_to_id(self.eos_token)

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def encode(self, text: str, chat_template: bool | None = None) -> list[int]:
        if chat_template is None:
            chat_template = self.chat_template
        if chat_template:
            text = self.wrap_chat(text)
        return self._tok.encode(text, add_special_tokens=False).ids

    def decode(self, token_ids: list[int]) -> str:
        return self._tok.decode(token_ids, skip_special_tokens=False)

    def wrap_chat(self, user_message: str) -> str:
        """Format one user turn and open the assistant turn.

        With thinking on, the model decides whether to emit a <think> block.
        With thinking off, an empty <think></think> block is pre-filled so the
        model answers directly.
        """
        text = f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
        if not self.thinking:
            text += "<think>\n\n</think>\n\n"
        return text
