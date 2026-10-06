try:
    from fire import Fire
    from indexing import index
except Exception as e:
    print("=" * 5, "Import Error", "=" * 5)
    print(e)


def search(query: str, k: int):
    ...


def search_dataset(dataset_path: str, k: int, save_dir: str):
    ...


def answer(query: str, k: int):
    ...


def answer_dataset(std_search_res_path: str, save_dir: str):
    ...


def evaluate(std_search_res_path: str, dataset_path: str):
    ...


if __name__ == "__main__":
    Fire({"index": index,
          "search": search,
          "search_dataset": search_dataset,
          "answer": answer,
          "answer_dataset": answer_dataset,
          "evaluate": evaluate},
         name="RAG")
