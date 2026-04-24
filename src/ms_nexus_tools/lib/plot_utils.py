from typing import Any, NamedTuple


class AxCommand(NamedTuple):
    command: str
    kwargs: dict[str, Any]
