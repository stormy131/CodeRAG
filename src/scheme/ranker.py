from typing import Protocol


class Ranker(Protocol):
    async def ainvoke(self, query: str) -> tuple[str, list[str]]:
        ...
