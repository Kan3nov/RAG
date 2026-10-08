try:
    from searching import search, search_dataset
    from transformers import pipeline, AutoTokenizer
    from transformers.utils.logging import set_verbosity_error
except Exception as e:
    print("=" * 5, "Import Error", "=" * 5)
    print(e)
    exit()


set_verbosity_error()


def answer(query: str, k: int):
    top_sources = search(query, k)
    model_name = "Qwen/Qwen3-0.6B"
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
    return response


def answer_dataset(std_search_res_path: str, save_dir: str):
    search_dataset(std_search_res_path, 5, save_dir)
