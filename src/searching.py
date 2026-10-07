try:
    import bm25s
    import json
    from config import index_dir
    from pathlib import Path
    from indexing import indexed
    from models import (MinimalSource, UnansweredQuestion,
                        MinimalSearchResults, StudentSearchResults,
                        AnsweredQuestion)
    from pydantic import TypeAdapter
except Exception as e:
    print("=" * 5, "Import Error", "=" * 5)
    print(e)


def search(query: str, k: int) -> list[MinimalSource]:
    sources = indexed()
    retriever = bm25s.BM25.load(index_dir, load_corpus=True)
    query_tokens = bm25s.tokenize(query)
    docs, scores = retriever.retrieve(query_tokens, k=k)
    top_sources = []
    for doc in docs[0]:
        id_ = doc["id"]
        top_sources.append(sources[id_])
    return top_sources


def search_dataset(dataset_path: str, k: int, save_dir: str) -> None:
    try:
        with open(dataset_path, "r") as f:
            content = f.read()
            adapter = TypeAdapter(
                dict[str, list[UnansweredQuestion | AnsweredQuestion]])
            if (adapter.validate_json(content)):
                questions_dict = json.loads(content)["rag_questions"]
                questions = []
                for q in questions_dict:
                    questions.append(UnansweredQuestion(
                        question_id=q["question_id"],
                        question=q["question"]))
    except Exception as e:
        print("=" * 5, "Error opening dataset", "=" * 5)
        print(e)
        exit()

    search_results = []
    for q in questions:
        min_srch_res = MinimalSearchResults(question_id=q.question_id,
                                            question=q.question,
                                            retrieved_sources=search(
                                                q.question,
                                                k)
                                            )
        search_results.append(min_srch_res)
    std_search_res = StudentSearchResults(search_results=search_results,
                                          k=k)
    json_file = std_search_res.model_dump_json(indent=4)
    try:
        if (not save_dir.endswith("/")):
            save_dir = save_dir + "/"
        save_dir += "search_results.json"
        save_path = Path(save_dir)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_dir, "w") as f:
            f.write(json_file)
    except Exception as e:
        print("=" * 5, "Error writing search results", "=" * 5)
        print(e)
        exit()
