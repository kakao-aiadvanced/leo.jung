import getpass
import os

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4
from langchain_core.documents import Document
from langchain import hub
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap

llm = ChatOpenAI(model="gpt-4o-mini")

## STEP4. User query = 'agent memory' 를 받아 관련된 chunks를 retrieve
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
)

vector_store = Chroma(
    collection_name="leojung_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not neccesary
)

retriever = vector_store.as_retriever(
	search_type="similarity", search_kwargs={"k": 6}
)

## STEP 5
#prompt = hub.pull("rlm/rag-prompt")
#example_messages = prompt.invoke(
#    {"context": "filler context", "question": "filler question"}
#).to_messages()
parser = JsonOutputParser()

system = """
You are a grader assessing relevance
    of a retrieved document to a user question. If the document contains keywords related to the user question,
    grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary relavance 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary relavance as a JSON with a single key 'relavance' and no premable or explanation.
"""

prompt = ChatPromptTemplate.from_messages(
    [
			("system", system),
			("human", "question: {question}\n\n document: {document} "),
		]
)

retrieval_grader = prompt | llm | parser


## STEP 6. 모든 docs 에 대해 yes, no 가 나와야 하는 케이스
queries = [
  "Tell me about Faker in league of legends", # no
  "Tell me about Chain of thought",						# yes
  "Tell me about Jailbreak Prompting",				# yes
]

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def check_hallucination(generation):
	system = """You are a grader assessing whether
    an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate
    whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a
    single key 'score' and no preamble or explanation."""

	prompt = ChatPromptTemplate.from_messages(
	    [
	        ("system", system),
	        ("human", "documents: {documents}\n\n answer: {generation} "),
	    ]
	)
	
	hallucination_grader = prompt | llm | JsonOutputParser()
	is_hallucination = hallucination_grader.invoke({"documents": docs, "generation": generation})
	return is_hallucination['score'] == 'yes'

def generate(query, use_retrieve, is_retry):
	system = """You are an assistant for question-answering tasks.
	    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
	    Use three sentences maximum and keep the answer concise

			And Always answer with source
"""
	
	prompt = ChatPromptTemplate.from_messages(
	    [
	        ("system", system),
	        ("human", "question: {question}\n\n context: {context} "),
	    ]
	)
	
	# Chain
	rag_chain = prompt | llm | StrOutputParser()
	
	# Run
	docs = retriever.invoke(query)
	if use_retrieve == False:
		docs = ""
	generation = rag_chain.invoke({"context": docs, "question": query})
	if check_hallucination(generation) and not is_retry:
		print('generated hallucinated, regenerating...')
		generate(query, use_retrieve, True)
		return
	print(generation)



## STEP 8. 위 relavance = yes 면 context 에 첨부.
for query in queries:
	docs = retriever.invoke(query)
	doc_txt = docs[0].page_content
	ans = retrieval_grader.invoke({"question": query, "document": doc_txt})
	if ans['relavance'] == 'no':
		doc_txt = ""
	generate(query, ans['relavance'] == 'yes', False)

## STEP 8. 







