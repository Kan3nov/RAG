try:
    from searching import search, search_dataset
    from transformers import pipeline, AutoTokenizer
except Exception as e:
    print("=" * 5, "Import Error", "=" * 5)
    print(e)
    exit()


def answer(query: str, k: int):
    top_sources = search(query, k)
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    context = "\n\n".join([src.content for src in top_sources])

    chat = [
        {"role": "system", "content": "You are an AI assitant that answers " +
         f"user questions according to this corpus:\n{context}"},
        {"role": "user", "content": query}
    ]

    prompt = tokenizer.apply_chat_template(chat, tokenize=False)
    prompt += "<think></think>"
    print(prompt)

    model = pipeline("text-generation", model_name)
    response = model(prompt)
    print(f"response: {response[0]["generated_text"][len(prompt) + 1:]}")


def answer_dataset(std_search_res_path: str, save_dir: str):
    search_dataset(std_search_res_path, 5, save_dir)
