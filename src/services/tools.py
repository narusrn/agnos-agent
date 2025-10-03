import json
import sys
from datetime import datetime
from typing import Literal

import pandas as pd
from langchain.tools import Tool
from langchain_chroma import Chroma
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, convert_to_messages
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

def prepare_documents_from_forum(file_path: str, group_size: int = 1) -> list[Document]:
    df = pd.read_excel(file_path)
    df = df.applymap(
        lambda x: str(x).replace("\xa0", " ").strip() if isinstance(x, str) else x
    )

    # เลือกเฉพาะแถวที่มีคำตอบจากแพทย์
    df = df[df['expert_answer'].notna()]

    documents = []
    # แบ่งเป็นกลุ่มละ group_size
    for i in range(0, len(df), group_size):
        chunk = df.iloc[i:i+group_size]
        content_lines = []

        for _, row in chunk.iterrows():
            # สร้าง text รวมคำถามและคำตอบ
            text = f"Title:{row['title'].strip()}\n"
            text += f"Info:{row['info'].strip()}\n"
            text += f"Tags:{row['tags'].strip()}\n"
            text += f"Question:{row['description'].strip()}\n"
            text += f"Expert({row['expert_role'.strip()]}, {row['expert_name'].strip()}): {row['expert_answer'].strip()}"
            content_lines.append(text)

        content = "\n\n---\n\n".join(content_lines)
        documents.append(Document(page_content=content, metadata={"source": "AgnosForum"}))

    # แบ่ง chunk ถ้าจำเป็น
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500
    )
    return splitter.split_documents(documents)

def get_retriever_from_forum(file_path: str):
    documents = prepare_documents_from_forum(file_path)
    vectorstore = Chroma(
        embedding_function=OpenAIEmbeddings(),
        persist_directory=None,
        collection_name="agnos_forum"
    )

    # add เป็น batch ย่อย ๆ
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        vectorstore.add_documents(documents[i:i+batch_size])

    return vectorstore.as_retriever(search_kwargs={"k": 5})



def document_search(question: str) -> dict:
    """
    Retrieve course documents
    """

    # Retrieval
    retriever = get_retriever_from_forum(file_path="src/data/data.xlsx")
    print("QUESTION: ", question)
    documents = retriever.invoke(question)

    return {"documents": documents, "question": question}

def initialize_default_tools() : 


    return [
        Tool.from_function(
            func=document_search,
            name="document_search",
            description="ใช้สำหรับค้นหาข้อมูล course จากฐานข้อมูลภายใน",
        ),
    ]