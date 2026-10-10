try:
    from fire import Fire
    from indexing import index
    from searching import search_dataset, search
    from answering import answer, answer_dataset
    from evaluation import evaluate
except Exception as e:
    print("=" * 5, "Import Error", "=" * 5)
    print(e)
    exit()


if __name__ == "__main__":
    Fire({"index": index,
          "search": search,
          "search_dataset": search_dataset,
          "answer": answer,
          "answer_dataset": answer_dataset,
          "evaluate": evaluate},
         name="RAG")
