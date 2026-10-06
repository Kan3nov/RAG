try:
    import os
    from pathlib import Path
    from enum import Enum
    from langchain_text_splitters import (RecursiveCharacterTextSplitter,
                                          Language)
    from models import MinimalSource
except Exception as e:
    print("=" * 5, "Import Error", "=" * 5)
    print(e)


class FileType(Enum):
    PY = "PYTHON"
    MD = "MARKDOWN"
    SH = "BASH"


def index(max_chunk_size: int = 2000):
    md_files = []
    py_files = []
    sh_files = []
    for root, dirs, files in os.walk("data"):
        for file in files:
            file_path = root + "/" + file
            if (Path(file_path).suffix == ".py"):
                py_files.append(file_path)
            elif (Path(file_path).suffix == ".md"):
                md_files.append(file_path)
            elif (Path(file_path).suffix == ".sh"):
                sh_files.append(file_path)
    chunk(max_chunk_size,
          **{FileType.PY.value: py_files,
             FileType.MD.value: md_files,
             FileType.SH.value: sh_files})


def chunk(max_chunk_size: int, **files):
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

    print(len(sources))


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
            source = MinimalSource(file_path=file,
                                   first_character_index=idx,
                                   last_character_index=idx +
                                   len(doc.page_content) - 1)
            sources.append(source)
            idx += len(doc.page_content)
            print(source)
    return sources
