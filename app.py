import streamlit as st
# The line `from langchain_community.document_loaders import PyMuPDFLoader` is importing the
# `PyMuPDFLoader` class from the `document_loaders` module within the `langchain_community` package.
# This class is likely used for loading PDF documents and extracting text content from them for
# further processing within the code.
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import Tool
from langchain_community.llms import Ollama
from langchain.agents import AgentType,initialize_agent


wikipedia = WikipediaAPIWrapper()
wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia)
loader=PyMuPDFLoader('1706.03762v7.pdf')
documents=loader.load()

text_splitter=RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=300)
final_docs=text_splitter.split_documents(documents=documents)

embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
retriever=Chroma.from_documents(final_docs,embedding=embedding).as_retriever()

y
def retriever_tool_chroma_db(query: str) -> str:
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])[:3]
tool_rag= Tool(name="Research Paper",func=retriever_tool_chroma_db,description="Attention is all you need about transformers and architecture")
tools=[tool_rag,wikipedia_tool]
llm=Ollama(model='gemma3:latest')
agent_executor=initialize_agent(tools=tools,llm=llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True)

user_input = st.text_input("Ask a question:", placeholder="Transformer architecture")

if user_input:
    response = agent_executor.invoke(user_input)
    st.write(response)
else:
    st.write("didn't get the tool")