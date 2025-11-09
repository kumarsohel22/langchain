# from langchain_community.document_loaders import WebBaseLoader
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.vectorstores import Chroma
# from langchain_community import embeddings
# from langchain_community.chat_models import ChatOllama
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.output_parsers import PydanticOutputParser
# from langchain.text_splitter import CharacterTextSplitter

# model_local = ChatOllama(model="mistral")

# # 1. Split data into chunks
# urls = [
#     "https://ollama.com/",
#     "https://ollama.com/blog/windows-preview",
#     "https://ollama.com/blog/openai-compatibility",
# ]
# docs = [WebBaseLoader(url).load() for url in urls]
# docs_list = [item for sublist in docs for item in sublist]
# text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
# doc_splits = text_splitter.split_documents(docs_list)

# # 2. Convert documents to Embeddings and store them
# vectorstore = Chroma.from_documents(
#     documents=doc_splits,
#     collection_name="rag-chroma",
#     embedding=embeddings.OllamaEmbeddings(model='nomic-embed-text'),
# )
# retriever = vectorstore.as_retriever()

# # 3. Before RAG
# print("Before RAG\n")
# before_rag_template = "What is {topic}"
# before_rag_prompt = ChatPromptTemplate.from_template(before_rag_template)
# before_rag_chain = before_rag_prompt | model_local | StrOutputParser()
# print(before_rag_chain.invoke({"topic": "Ollama"}))

# # 4. After RAG
# print("\n########\nAfter RAG\n")
# after_rag_template = """Answer the question based only on the following context:
# {context}
# Question: {question}
# """
# after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
# after_rag_chain = (
#     {"context": retriever, "question": RunnablePassthrough()}
#     | after_rag_prompt
#     | model_local
#     | StrOutputParser()
# )
# print(after_rag_chain.invoke("What is Ollama?"))

# # loader = PyPDFLoader("Ollama.pdf")
# # doc_splits = loader.load_and_split()

## Pdf reader
from langchain_community.document_loaders import PyPDFLoader
loader=PyPDFLoader(r'D:\STUDY\Gen AI\Langchain\RAG\attention.pdf')
docs=loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
documents=text_splitter.split_documents(docs)
docs = documents[:20]

## Vector Embedding And Vector Store
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
db = Chroma.from_documents(documents,OllamaEmbeddings(model= 'nomic-embed-text'),collection_name='attention')
query = "Who are the authors of attention is all you need?"
retireved_results=db.similarity_search(query)
print("retireved_results", retireved_results)
print(retireved_results[0].page_content)