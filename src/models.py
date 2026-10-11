try:
    from pydantic import BaseModel, Field
    from uuid import uuid4
    from typing import Optional
except Exception as e:
    print("=" * 5, "Import Error", "=" * 5)
    print(e)
    exit()


class MinimalSource(BaseModel):
    """A source chunk extracted from a file with its position metadata."""

    file_path: str
    content: Optional[str] = None
    first_character_index: int
    last_character_index: int

    def __str__(self) -> str:
        """Return a compact string representation including
        the source path and span.

        Args:
            None
        Returns:
            str: the source path and character span
        """
        return (self.file_path +
                f" [{self.first_character_index}:{self.last_character_index}]")


class UnansweredQuestion(BaseModel):
    """A question that has not yet been answered by the RAG pipeline."""

    question_id: str = Field(default_factory=lambda: str(uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question paired with its supporting sources and final answer."""

    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A collection of unanswered or answered questions for evaluation."""

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Search results for a question without the generated answer."""

    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """A search result enriched with an answer generated from the
    retrieved sources."""

    answer: str


class StudentSearchResults(BaseModel):
    """The retrieval output produced for a dataset of questions."""

    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """The retrieval output and answer for each question in a dataset."""

    search_results: list[MinimalAnswer]
    k: int
