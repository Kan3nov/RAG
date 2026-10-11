try:
    import bm25s
    import json
    from tqdm import tqdm
    from config import (index_dir, sources_file)
    from pathlib import Path
    from models import (MinimalSource, UnansweredQuestion,
                        MinimalSearchResults, StudentSearchResults,
                        RagDataset)
    from pydantic import TypeAdapter
    from utils import print_e
except Exception as e:
    print("=" * 5, "Import Error", "=" * 5)
    print(e)
    exit()


def load_src_and_ret() -> tuple[list[MinimalSource], bm25s.BM25]:
    """Load indexed source chunks and the BM25 retriever
     for retrieval tasks.

    Args:
        None
    Returns:
        tuple[list[MinimalSource], bm25s.BM25]: indexed chunks and the
            retriever used to search them
    """
    try:
        with open(sources_file, 'r') as f:
            content = f.read()
            sources = []
            sources_json = json.loads(content)
            for obj in sources_json:
                sources.append(MinimalSource(
                    file_path=obj["file_path"],
                    content=obj["content"],
                    first_character_index=obj["first_character_index"],
                    last_character_index=obj["last_character_index"])
                    )
    except Exception as e:
        print_e("Error Reading Sources File", e,
                "Index sources before searching")
    try:
        retriever = bm25s.BM25.load(index_dir, load_corpus=True)
    except Exception as e:
        print_e("Error Loading Indexed Chunks", e,
                "Index sources before searching")
    return (sources, retriever)


def search(query: str, k: int) -> list[MinimalSource]:
    """Search the indexed source corpus for the top matching chunks.

    Args:
        query: text to search for
        k: maximum number of matching chunks to return
    Returns:
        list[MinimalSource]: the top matching chunks
    """
    if (k <= 0):
        print_e("Invalid K Value", None,
                "K should be greater than zero")
    elif (query == ""):
        print_e("Invalid Query", None,
                "Query can't be empty")
    sources, retriever = load_src_and_ret()
    top_sources = search_query(query, k, retriever, sources)
    return top_sources


def search_query(query: str, k: int,
                 retriever: bm25s.BM25,
                 sources: list[MinimalSource]) -> list[MinimalSource]:
    """Retrieve the most relevant source chunks for a query using BM25.

    Args:
        query: text to search for
        k: maximum number of matching chunks to return
        retriever: BM25 retriever indexed over the source corpus
        sources: source chunks corresponding to the retriever's corpus
    Returns:
        list[MinimalSource]: the most relevant chunks, or an unknown-source
            placeholder when no relevant chunks are found
    """
    query_tokens = bm25s.tokenize(query)
    docs, scores = retriever.retrieve(query_tokens, k=k)
    score_list = scores.flatten().tolist()
    if (max(score_list) <= 0.1):
        return [MinimalSource(file_path="Unknown",
                              content="None",
                              first_character_index=0,
                              last_character_index=0)]
    top_sources = []
    for doc in docs[0]:
        id_ = doc["id"]
        top_sources.append(sources[id_])
    return top_sources


def search_dataset(dataset_path: str, k: int, save_dir: str) -> None:
    """Search every question in a dataset and save the results as JSON.

    Args:
        dataset_path: path to the dataset containing questions
        k: maximum number of matching chunks to retrieve per question
        save_dir: directory in which to save the search results
    """
    if (k <= 0):
        print_e("Invalid K Value", None,
                "K should be greater than zero")
    save_file = save_dir + dataset_path[dataset_path.rfind("/") + 1:]
    try:
        with open(dataset_path, "r") as f:
            content = f.read()
            adapter = TypeAdapter(
                dict[str, list[UnansweredQuestion]])
            if (adapter.validate_json(content)):
                questions_dict = json.loads(content)["rag_questions"]
                questions = []
                for q in questions_dict:
                    if q["question"] == "":
                        print("Questions can't be empty, " +
                              f"question {q["question_id"]} is skipped")
                        continue

                    questions.append(UnansweredQuestion(
                        question_id=q["question_id"],
                        question=q["question"]))
                q_dataset = RagDataset(rag_questions=questions)
    except Exception as e:
        print_e("Error Opening Questions Dataset", e)

    sources, retriever = load_src_and_ret()
    search_results = []
    for q in tqdm(q_dataset.rag_questions,
                  desc="Sources Retrieval",
                  unit="prompt"):
        min_srch_res = MinimalSearchResults(question_id=q.question_id,
                                            question=q.question,
                                            retrieved_sources=search_query(
                                                q.question,
                                                k,
                                                retriever,
                                                sources)
                                            )
        search_results.append(min_srch_res)
    std_search_res = StudentSearchResults(search_results=search_results,
                                          k=k)
    try:
        json_file = std_search_res.model_dump_json(indent=4)
        save_path = Path(save_file)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_file, "w") as f:
            f.write(json_file)
        print(f"Saved student_search_results to {save_file}")
    except Exception as e:
        print("=" * 5, "Error writing search results", "=" * 5)
        print(e)
        exit()
