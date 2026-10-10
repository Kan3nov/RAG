try:
    from pydantic import BaseModel, Field
    from uuid import uuid4
except Exception as e:
    print("=" * 5, "Import Error", "=" * 5)
    print(e)
    exit()


class MinimalSource(BaseModel):
    file_path: str
    content: str = None
    first_character_index: int
    last_character_index: int

    def __str__(self):
        return (self.file_path +
                f" [{self.first_character_index}:{self.last_character_index}]")


class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    answer: str


class StudentSearchResults(BaseModel):
    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    search_results: list[MinimalAnswer]
    k: int
