try:
    import os
    from pathlib import Path
    from enum import Enum
    from config import index_dir
    from langchain_text_splitters import (RecursiveCharacterTextSplitter,
                                          Language)
    import bm25s
    from models import MinimalSource
except Exception as e:
    print("=" * 5, "Import Error", "=" * 5)
    print(e)


class FileType(Enum):
    PY = "PYTHON"
    TXT = "TEXT"
    MD = "MARKDOWN"
    SH = "BASH"


def find_files() -> dict[str: list[str]]:

    md_files = []
    py_files = []
    sh_files = []
    txt_files = []
    root_dir = "data/raw"
    try:
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                file_path = root + "/" + file
                if (Path(file_path).suffix == ".py"):
                    py_files.append(file_path)
                elif (Path(file_path).suffix == ".md"):
                    md_files.append(file_path)
                elif (Path(file_path).suffix == ".sh"):
                    sh_files.append(file_path)
                elif (Path(file_path).suffix == ".txt"):
                    txt_files.append(file_path)
    except Exception as e:
        print("=" * 5, "Error Finding Files", "=" * 5)
        print(e)
        exit()

    file_paths = {FileType.PY.value: py_files,
                  FileType.MD.value: md_files,
                  FileType.SH.value: sh_files,
                  FileType.TXT.value: txt_files}
    return file_paths


def index(max_chunk_size: int = 2000) -> None:
    sources = indexed(max_chunk_size)
    print(f"Ingestion complete! Indexed {len(sources)}" +
          f" chunks under {index_dir}")


def indexed(max_chunk_size: int = 2000) -> list[MinimalSource]:
    file_paths = find_files()
    sources = chunk(max_chunk_size, **file_paths)
    corpus = [src.content for src in sources]

    retriever = bm25s.BM25(corpus=corpus)
    retriever.index(bm25s.tokenize(corpus))
    index_files = Path(index_dir)
    index_files.parent.mkdir(parents=True, exist_ok=True)
    retriever.save(index_files)
    return sources


def chunk(max_chunk_size: int, **files) -> list[MinimalSource]:
    sources = []

    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN, chunk_size=max_chunk_size, chunk_overlap=0)
    sources.extend(create_sources(files[FileType.MD.value], splitter))

    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=max_chunk_size, chunk_overlap=0)
    sources.extend(create_sources(files[FileType.PY.value], splitter))

    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=max_chunk_size, chunk_overlap=0)
    sources.extend(create_sources(files[FileType.SH.value], splitter))

    splitter = RecursiveCharacterTextSplitter(chunk_size=max_chunk_size,
                                              chunk_overlap=0)
    sources.extend(create_sources(files[FileType.TXT.value], splitter))

    return sources


def create_sources(files: list[str],
                   splitter: RecursiveCharacterTextSplitter
                   ) -> list[MinimalSource]:
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

        md_docs = splitter.create_documents([content])
        for doc in md_docs:
            lst_idx = idx + len(doc.page_content) - 1
            if content[idx: lst_idx + 1] == "":
                idx += lst_idx + 1
                continue
            source = MinimalSource(file_path=file,
                                   content=content[idx: lst_idx + 1],
                                   first_character_index=idx,
                                   last_character_index=lst_idx)
            sources.append(source)
            idx += lst_idx + 1
    return sources
