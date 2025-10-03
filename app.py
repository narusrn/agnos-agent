# __import__('pysqlite3')
import os 
import sys
from dotenv import load_dotenv
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
load_dotenv()

import streamlit as st
from src.services import StreamlitCallbackHandler, initialize_agent, initialize_combined_agent
from src.services import  AgnosAgent


def chat_completion(prompt, callback_handler, max_retries=3):
    attempts = 0
    while attempts < max_retries:
        response = st.session_state.agent.invoke(
            {"messages": [{"role": "human", "content": prompt}]},
            # {"chat_history": []},  # optionally add chat_history here
            {"callbacks": [callback_handler], "configurable": {"thread_id": "thread-1"}},
        )

        if response['messages'][-1].content != "Oops! Something went wrong. Please give it another try!":
            return response
        attempts += 1
    return "Failed after several attempts. Please try again later."

# Initialize Streamlit app
st.title("Chat with Data")

# Initial assistant message
initial_message = """
👋 สวัสดีค่ะ! ยินดีต้อนรับสู่ Agnos-Agent — ผู้ช่วยของคุณในการตอบคำถามสุขภาพและให้คำแนะนำ

คุณสามารถสอบถามเกี่ยวกับอาการหรือปัญหาสุขภาพ เช่น:
- "ช่วงนี้เครียดและนอนไม่หลับ ควรทำอย่างไร?"
- "มีวิธีบรรเทาอาการปวดหัวจากอารมณ์แปรปรวนไหม?"
- "อาการคลื่นไส้และแน่นหน้าอกเกิดจากอะไรได้บ้าง?"

พิมพ์คำถามของคุณได้เลยด้านล่างนี้! 🩺
"""


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    # st.session_state.agent =  AgnosAgent(verbose=False).creat_workflow()
    # st.session_state.agent = initialize_agent()
    st.session_state.agent = initialize_combined_agent()
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = "default_thread"

# Display message history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input and chat handling
if prompt := st.chat_input("พิมพ์คำถามของคุณที่นี่..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        callback_handler = StreamlitCallbackHandler(st.empty())
        with st.spinner("กำลังประมวลผล..."):

            response = chat_completion(prompt, callback_handler)

        st.session_state.messages.extend([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response['messages'][-1].content}])
        
else:
    with st.chat_message("assistant"):
        st.markdown(initial_message)