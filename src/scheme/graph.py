from typing import TypedDict, NamedTuple


TaskConfig = NamedTuple(
    "TaskConfig",
    [
        ("summarize", bool),
        ("expand_query", bool),
        ("verbose", bool),
        ("build_index", bool),
    ],
)


class RAGState(TypedDict):
    question: str
    retrieved: list[str]
    task_config: TaskConfig
    answer: str | None
