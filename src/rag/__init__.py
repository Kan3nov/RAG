from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

with open("test.md", "r") as f:
    content = f.read()

md_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN, chunk_size=2000, chunk_overlap=0
)
md_docs = md_splitter.create_documents([content])
print("# Chunking Fininished")
print(f"# {len(md_docs)} chunks has been made")
for i, md_doc in enumerate(md_docs):
    print("=" * 5, f"chunk {i + 1}", "=" * 5)
    print("# Chunk length: ", len(md_doc.page_content))
    print(md_doc.page_content)
