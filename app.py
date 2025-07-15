import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import Tool
from langchain_community.llms import Ollama
from langchain.agents import AgentType,initialize_agent
import datetime
import os
import tempfile
st.write("upload_a_rag_document")
uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'txt'])
if uploaded_file is not None:
    st.write("File uploaded:", uploaded_file.name)
 
    # Save file to a temp location in binary mode
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name  # Save path for PyPDFLoader

   

    loader=PyMuPDFLoader(temp_path)
    print(loader)

    documents=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=300)

    documents=text_splitter.split_documents(documents=documents)

    embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    retriever=Chroma.from_documents(documents=documents,embedding=embedding).as_retriever()

    def retriever_tool_chroma_db(query: str) -> str:

        docs = retriever.invoke(query)
        if not docs:
            return "Out_of_SCOPE"
        return "\n\n".join([doc.page_content for doc in docs[:3]])
    def extract_summary_description(docs, max_pages=5):
        for i, doc in enumerate(docs[:max_pages]):
            text = doc.page_content.strip()

        # Skip if empty or too short
            if not text or len(text) < 100:
                continue

        # Skip if contains typical author/affiliation noise
            if any(keyword in text.lower() for keyword in ['author', 'institute', 'affiliation', 'email']):
                continue

        # Try getting the first clean sentence
            sentences = text.split(".")
            for s in sentences:
                s = s.strip()
                if len(s) >= 150 and s[0].isupper():  
                    return s

        return "No meaningful summary found in the first few pages."
    summary = extract_summary_description(documents)

    if summary == "No meaningful summary found in the first few pages.":
        summary = extract_summary_description(docs=documents,max_pages=10)

    tool_description = f"Use this tool to answer questions about the document. It focuses on: {summary}"


    wikipedia = WikipediaAPIWrapper()
    wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia)

    tool_rag= Tool(name="Research Paper",func=retriever_tool_chroma_db,description=tool_description)
    tools=[tool_rag,wikipedia_tool]
    llm=Ollama(model='gemma3:latest')
    agent_executor=initialize_agent(tools=tools,llm=llm,agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,verbose=True)
    user_input = st.text_input("Ask a question:")

    if user_input:
        response = agent_executor.invoke(user_input)
        st.write(response)
    else:
        st.write("It is outside the tool scope")