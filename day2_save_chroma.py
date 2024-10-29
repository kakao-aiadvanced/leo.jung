import getpass
import os

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4
from langchain_core.documents import Document

llm = ChatOpenAI(model="gpt-4o-mini")


## STEP1. 블로그 포스팅 본문 load
loaders = WebBaseLoader([
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
])

docs = loaders.load()
#docs = [doc.page_content for doc in docs]

## STEP2. 불러온 본문을 Split (Chunking) 하기.
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
		separators=[
        "\n\n",
        "\n",
        " ",
        ".",
        ",",
        "\u200b",  # Zero-width space
        "\uff0c",  # Fullwidth comma
        "\u3001",  # Ideographic comma
        "\uff0e",  # Fullwidth full stop
        "\u3002",  # Ideographic full stop
        "",
    ]
)

#chunks = []
#	for t in text_splitter.split_text(doc):
#		chunks.append(t)
#

texts = text_splitter.split_documents(docs)

## STEP3. Chunks 를 임베딩하여 Vector store 저장.
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
)

vector_store = Chroma(
    collection_name="leojung_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not neccesary
)

vector_store.add_documents(texts)

#chunk_docs = [Document(page_content=chunk) for chunk in chunks]
#uuids = [str(uuid4()) for _ in range(len(chunk_docs))]
#vector_store.add_documents(documents=chunk_docs, ids=uuids)

