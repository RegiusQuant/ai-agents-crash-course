import re
import logging
from typing import List, Dict, Any, Optional

# Configure a module-level logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class TagExtractionResult:
    """
    Represents the outcome of parsing out XML-like tag content.

    Attributes:
        items: The extracted strings (whitespace-trimmed).
        found: True if at least one item was extracted.
    """
    __slots__ = ('items', 'found')

    def __init__(self, items: List[str]):
        # Trim whitespace and set attributes
        cleaned = [s.strip() for s in items]
        self.items: List[str] = cleaned
        self.found: bool = bool(cleaned)


class TagParser:
    """
    Extracts all instances of content wrapped in a specific tag,
    e.g. <thought>…</thought>.
    """

    def __init__(self, tag_name: str):
        if not tag_name.isidentifier():
            raise ValueError(f"Invalid tag name: {tag_name!r}")
        self._tag = tag_name
        # Precompile regex for performance and safety
        self._pattern = re.compile(
            rf"<{re.escape(tag_name)}>(.*?)</{re.escape(tag_name)}>",
            re.DOTALL
        )

    def parse(self, text: str) -> TagExtractionResult:
        matches = self._pattern.findall(text)
        return TagExtractionResult(matches)


def create_message(role: str, content: str, tag: Optional[str] = None) -> Dict[str, str]:
    """
    Build a single message dict for the chat API, optionally wrapping
    the content in a tag.
    """
    if tag:
        content = f"<{tag}>{content}</{tag}>"
    return {"role": role, "content": content}


def add_to_history(
    history: List[Dict[str, str]],
    role: str,
    content: str,
    tag: Optional[str] = None
) -> None:
    """
    Append a new message (with optional tag) into the history list.
    """
    message = create_message(role, content, tag)
    history.append(message)


class MessageHistory:
    """
    Keeps a list of messages, automatically dropping the oldest
    when a max_size is set and reached.
    """

    def __init__(self, max_size: Optional[int] = None):
        self._msgs: List[Dict[str, str]] = []
        self._max = max_size if (max_size and max_size > 0) else None

    def append(self, message: Dict[str, str]) -> None:
        if self._max and len(self._msgs) >= self._max:
            self._msgs.pop(0)
        self._msgs.append(message)

    def extend(self, messages: List[Dict[str, str]]) -> None:
        for msg in messages:
            self.append(msg)

    def all(self) -> List[Dict[str, str]]:
        return self._msgs.copy()


class PinnedMessageHistory(MessageHistory):
    """
    Like MessageHistory but always retains the first (pinned) message
    even when full.
    """

    def append(self, message: Dict[str, str]) -> None:
        if self._max and len(self._msgs) >= self._max:
            # Remove second item, keeping index 0 pinned
            self._msgs.pop(1)
        self._msgs.append(message)
