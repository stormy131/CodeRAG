from typing import Protocol


class Ranker(Protocol):
    async def aretrieve(self, query: str) -> list[str]:
        ...

    async def aanswer(self, query: str) -> str:
        ...
