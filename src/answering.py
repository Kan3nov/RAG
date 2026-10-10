try:
    from searching import search
    from tqdm import tqdm
    from typing import Any
    from config import model_name
    from pathlib import Path
    from utils import print_e
    from models import (StudentSearchResults, MinimalAnswer,
                        StudentSearchResultsAndAnswer,
                        MinimalSource)
    from transformers import pipeline, AutoTokenizer, TextGenerationPipeline
    from transformers.utils.logging import set_verbosity_error
    from pydantic import TypeAdapter
except Exception as e:
    print("=" * 5, "Import Error", "=" * 5)
    print(e)
    exit()


set_verbosity_error()


def answer(query: str, k: int) -> str:
    top_sources = search(query, k)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = pipeline("text-generation", model_name,
                     max_new_tokens=2000)
    response = answered(query, top_sources, tokenizer, model)
    return response


def answered(query: str, top_sources: list[MinimalSource], tokenizer: Any,
             model: TextGenerationPipeline):
    context = "\n".join([src.content for src in top_sources])
    prompt = \
        "<|im_start|>system\nYou are an AI assitant that answers " +\
        f"user questions according to this corpus:\n{context}<|im_end|>" +\
        f"\n<|im_start|>user\n {query}<|im_end|>\n" +\
        "<|im_start|>assistant\n<think>\n</think>"

    context_length = len(tokenizer.encode(prompt))
    if (context_length >= 30_768):
        prompt_tokens = tokenizer(prompt)["input_ids"]
        prompt = tokenizer.decode(prompt_tokens[:30_768])

    response = model(prompt)[0]["generated_text"][len(prompt) + 1:]
    response = response if response[0] != "\n" else response[1:]
    return response


def answer_dataset(std_search_res_path: str, save_dir: str) -> None:
    save_file = save_dir + \
        std_search_res_path[std_search_res_path.rfind("/") + 1:]
    try:
        with open(std_search_res_path, "r") as f:
            adapter = TypeAdapter(StudentSearchResults)
            content = f.read()
            search_results = adapter.validate_json(content)
            k = search_results.k
    except Exception as e:
        print_e("Reading Student Search Results Error", e,
                "Search Indexed Chunks First")
    minimal_answers = []
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = pipeline("text-generation", model_name,
                     max_new_tokens=2000)
    for search_result in tqdm(search_results.search_results,
                              desc="Answering questions", unit="answer"):
        min_answer = MinimalAnswer(
            question=search_result.question,
            question_id=search_result.question_id,
            retrieved_sources=search_result.retrieved_sources,
            answer=answered(search_result.question,
                            search_result.retrieved_sources,
                            tokenizer, model))
        minimal_answers.append(min_answer)
        print("\n", minimal_answers[-1].answer)
    std_search_answer = StudentSearchResultsAndAnswer(
        search_results=minimal_answers,
        k=k)
    try:
        save_path = Path(save_file)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_file, "w") as f:
            f.write(std_search_answer.model_dump_json(indent=4))
        print(f"Saved student to {save_file}")
    except Exception as e:
        print_e("Error Writing Answer Dataset", e)
