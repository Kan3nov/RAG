try:
    from utils import print_e
    from pydantic import TypeAdapter
    from models import (StudentSearchResults, AnsweredQuestion,
                        MinimalSource)
    from typing import Literal
except Exception as e:
    print("=" * 5, "Import Error", "=" * 5)
    print(e)
    exit()


def calc_recall(answer: MinimalSource, results: list[MinimalSource]):
    for source in results:
        diff = min(answer.last_character_index,
                   source.last_character_index) - \
                max(answer.first_character_index,
                    source.first_character_index)
        if (answer.file_path != source.file_path):
            continue
        elif (diff <= 0):
            continue
        elif (diff >= 0.05 *
              (answer.last_character_index - answer.first_character_index)):
            return 1
    return 0


def evaluate(std_search_res_path: str, dataset_path: str):
    try:
        with open(std_search_res_path, 'r') as f:
            content = f.read()
            adapter = TypeAdapter(StudentSearchResults)
            std_search_res = adapter.validate_json(content)
    except Exception as e:
        print_e("Error Reading Search Results Path", e,
                "Try Searching Sources Before")
    try:
        with open(dataset_path, 'r') as f:
            content = f.read()
            adapter = TypeAdapter(dict[Literal["rag_questions"],
                                       list[AnsweredQuestion]])
            search_ans: list[AnsweredQuestion] = \
                adapter.validate_json(content)["rag_questions"]
    except Exception as e:
        print_e("Error Reading Answered Questions Path", e,
                "Try Searching Sources Before")
    recall = 0
    print(len(search_ans))
    print(len(std_search_res.search_results))
    for answer, res in zip(search_ans, std_search_res.search_results):
        recall += calc_recall(answer.sources[0], res.retrieved_sources)
    return (recall / len(search_ans))
