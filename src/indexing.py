try:
    import os
    from pathlib import Path
    from enum import Enum
    from config import index_dir, sources_file
    from langchain_text_splitters import (RecursiveCharacterTextSplitter,
                                          Language)
    from tqdm import tqdm
    from utils import print_e
    import bm25s
    import json
    from models import MinimalSource
except Exception as e:
    print("=" * 5, "Import Error", "=" * 5)
    print(e)


class FileType(Enum):
    """Supported source file categories for indexing."""
    PY = "PYTHON"
    MD = "MARKDOWN"
    TXT = "TEXT"
    SH = "SHELL"


def find_files() -> dict[str, list[str]]:
    """Find all .py,.md,.txt,.sh files starting from the root directory

    Returns:
        dict[str, list[str]]: each file type is associated with its
            corresponding file paths
    """

    md_files = []
    txt_files = []
    py_files = []
    sh_files = []
    root_dir = "data/raw"
    if (not Path(root_dir).is_dir()):
        print_e("Raw Data Not Found", None,
                "Check files under dara/raw")
    try:
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                file_path = root + "/" + file
                if (Path(file_path).suffix == ".py"):
                    py_files.append(file_path)
                elif (Path(file_path).suffix == ".md"):
                    md_files.append(file_path)
                elif (Path(file_path).suffix == ".txt"):
                    txt_files.append(file_path)
                elif (Path(file_path).suffix == ".sh"):
                    sh_files.append(file_path)
    except Exception as e:
        print("=" * 5, "Error Finding Files", "=" * 5)
        print(e)
        exit()

    file_paths = {FileType.PY.value: py_files,
                  FileType.MD.value: md_files,
                  FileType.TXT.value: txt_files,
                  FileType.SH.value: sh_files}
    return file_paths


def index(max_chunk_size: int = 2000) -> None:
    """Serve as a CLI for indexing source files

    Args:
        max_chunk_size: maximum limit for chunk length
    """
    sources = indexed(max_chunk_size)
    print(f"Ingestion complete! Indexed {len(sources)}" +
          f" chunks under {index_dir}")


def indexed(max_chunk_size: int = 2000) -> list[MinimalSource]:
    """index all files and save the index into a pre-defined directory

    Args:
        max_chunk_size: maximum limit for chunk length
    Returns:
        list[MinimalSource]: all chunked sources
    """
    file_paths = find_files()
    sources = chunk(max_chunk_size, **file_paths)
    corpus = [src.content for src in sources]

    try:
        retriever = bm25s.BM25(corpus=corpus)
        retriever.index(bm25s.tokenize(corpus))
        index_files = Path(index_dir)
        index_files.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print_e("Error Indexing Corpus", e,
                "Check files under data/raw")
    try:
        retriever.save(index_files)
        sources_json = json.dumps(
            [src.model_dump(mode="json") for src in sources], indent=4
            )
        with open(sources_file, "w") as f:
            f.write(sources_json)
    except Exception as e:
        print_e("Error Saving Index", e)
    return sources


def chunk(max_chunk_size: int, **files: list[str]) -> list[MinimalSource]:
    """Create a list of all chunks produced from the files

    Args:
        max_chunk_size: maximum limit for chunk length
        files: paths of files to be split
    Returns:
        list[MinimalSource]: all chunked sources
    """
    if (sum([len(x) for x in files.values()]) == 0):
        print_e("Corpus Error", None,
                "Ensure that relevant files exist under data/raw")
    sources = []
    overlap = int(0.2 * max_chunk_size)

    pbar = tqdm(total=sum([len(x) for x in files.values()]),
                desc="Files chunking", unit="file")
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN, chunk_size=max_chunk_size,
        chunk_overlap=overlap)
    sources.extend(create_sources(files[FileType.MD.value], splitter, pbar))

    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=max_chunk_size,
        chunk_overlap=overlap)
    sources.extend(create_sources(files[FileType.PY.value], splitter, pbar))
    sources.extend(create_sources(files[FileType.SH.value], splitter, pbar))

    splitter = RecursiveCharacterTextSplitter(chunk_size=max_chunk_size,
                                              chunk_overlap=overlap)
    sources.extend(create_sources(files[FileType.TXT.value], splitter, pbar))
    pbar.close()
    return sources


def create_sources(files: list[str],
                   splitter: RecursiveCharacterTextSplitter,
                   pbar: tqdm) -> list[MinimalSource]:
    """Split all files into chunks

    Args:
        files: paths of files to be split
        splitter: the thing that splits file contents
        pbar: bar to indicate chunking progress
    Returns:
        list[MinimalSource]: list of chunked sources
    """
    sources = []
    for file in files:
        idx = 0
        try:
            with open(file, "r") as f:
                content = f.read()
        except Exception as e:
            print("=" * 5, "Reading Error", "=" * 5)
            print(e)
            exit()

        docs = splitter.create_documents([content])
        for doc in docs:
            lst_idx = idx + len(doc.page_content)
            if content[idx: lst_idx] == "":
                idx = lst_idx + 1
                continue
            source = MinimalSource(file_path=file,
                                   content=content[idx: lst_idx],
                                   first_character_index=idx,
                                   last_character_index=lst_idx)
            sources.append(source)
            idx = lst_idx + 1
        pbar.update(1)
    return sources
