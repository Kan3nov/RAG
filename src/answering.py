try:
    from searching import search
    import json
    from tqdm import tqdm
    from config import model_name
    from pathlib import Path
    from utils import print_e
    from models import (StudentSearchResults, MinimalAnswer,
                        StudentSearchResultsAndAnswer,
                        MinimalSource)
    from transformers import pipeline, AutoTokenizer
    from transformers.utils.logging import set_verbosity_error
    from pydantic import TypeAdapter
except Exception as e:
    print("=" * 5, "Import Error", "=" * 5)
    print(e)
    exit()


set_verbosity_error()


def answer(query: str, k: int):
    top_sources = search(query, k)
    response = answered(query, top_sources)
    return response


def answered(query: str, top_sources: list[MinimalSource]):
    context = "\n".join([src.content for src in top_sources])
    prompt = \
        "<|im_start|>system\nYou are an AI assitant that answers " +\
        f"user questions according to this corpus:\n{context}<|im_end|>" +\
        f"\n<|im_start|>user\n {query}<|im_end|>\n" +\
        "<|im_start|>assistant\n<think>\n</think>"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    context_length = 32_768 - len(tokenizer(prompt)["input_ids"])
    model = pipeline("text-generation", model_name,
                     max_new_tokens=context_length)
    response = model(prompt)[0]["generated_text"][len(prompt) + 1:]
    response = response if response[0] != "\n" else response[1:]


def answer_dataset(std_search_res_path: str, save_dir: str) -> None:
    try:
        with open(std_search_res_path, "r") as f:
            adapter = TypeAdapter(StudentSearchResults)
            content = f.read()
            search_results = adapter.validate_json(content)
            k = search_results.k
    except Exception as e:
        print_e("Reading Student Search Results Error", e)
    minimal_answers = []
    for search_result in tqdm(search_results.search_results,
                              desc="Answering questions", unit="answer"):
        min_answer = MinimalAnswer(
            question=search_result.question,
            question_id=search_result.question_id,
            retrieved_sources=search_result.retrieved_sources,
            answer=answer(search_result.question, k))
        minimal_answers.append(min_answer)
    std_search_answer = StudentSearchResultsAndAnswer(
        search_results=minimal_answers,
        k=k)
    try:
        if (not save_dir.endswith("/")):
            save_dir = save_dir + "/"
        save_dir += "answer_search_results.json"
        save_path = Path(save_dir)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_dir, "w") as f:
            f.write(json.dumps(std_search_answer, indent=4))
    except Exception as e:
        print_e("Error Writing Answer Dataset", e)
