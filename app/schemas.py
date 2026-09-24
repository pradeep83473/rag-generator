from typing import Literal

from pydantic import BaseModel, Field


class CollectionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field("", max_length=1000)
    instructions: str = Field(
        "", max_length=4000,
        description="Optional extra guidance for answers, e.g. tone or audience.",
    )


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=20000)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[ChatMessage] = Field(default_factory=list)
    top_k: int | None = Field(None, ge=1, le=20)


class Source(BaseModel):
    number: int
    cited: bool
    document_id: str
    filename: str
    page: int | None
    text: str
    similarity: float


class QueryResponse(BaseModel):
    answer: str
    search_query: str
    grounded: bool
    sources: list[Source]
