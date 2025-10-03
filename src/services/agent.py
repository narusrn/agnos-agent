from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langchain.tools import Tool
from langchain_core.messages import AIMessage


from src.services.tools import initialize_default_tools
from src.services.workflow import AgnosAgent


def initialize_agent():

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
            You are a trustworthy and helpful consultation assistant for Agnos.
            Your main responsibility is to assist users by providing advice and guidance based on previous answers by medical professionals.

            Guidelines:

            - Use document_search whenever you need to find information from the historical Q&A answered by doctors on Agnos.
            - If document_search does not return relevant information, respond politely saying that you could not find the information, instead of guessing.
            - Always provide answers clearly, politely, and in an easy-to-understand manner.
            - Respond in Thai.
            """),
        ("placeholder", "{messages}"),
    ])
    
    tools = initialize_default_tools()

    llm = ChatOpenAI(model="gpt-4o", streaming=True, temperature=0.7, max_tokens=256, verbose=True)

    checkpointer = InMemorySaver()

    # Creating the agent
    react_agent = create_react_agent(llm, tools, debug=True, prompt=prompt_template, checkpointer=checkpointer)

    return react_agent

def initialize_combined_agent():
    # สร้าง workflow agent
    agnos_workflow = AgnosAgent(verbose=False).creat_workflow()

    # สร้าง React agent
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
            You are a trustworthy and helpful consultation assistant for Agnos.
            Your main responsibility is to assist users by providing advice and guidance based on previous answers by medical professionals.

            Guidelines:
            - Use document_search whenever you need to find information from the historical Q&A answered by doctors on Agnos.
            - If document_search does not return relevant information, respond politely saying that you could not find the information, instead of guessing.
            - Always provide answers clearly, politely, and in an easy-to-understand manner.
            - Respond in Thai.
            """),
        ("placeholder", "{messages}"),
    ])

    llm = ChatOpenAI(model="gpt-4o", streaming=True, temperature=0.7, max_tokens=256, verbose=True)

    # React agent อาจใช้ tool ตัวเดียวที่เรียก workflow
    def document_search(messages):
        """
        Tool function ให้ React agent เรียก workflow
        """
        result = agnos_workflow.invoke({"messages": messages})
        # คืน candidate_answer จาก workflow
        return  result

    tools = [
        Tool.from_function(
            func=document_search,
            name="document_search",
            description="ใช้สำหรับค้นหาข้อมูลทางการแพทย์ จากฐานข้อมูลภายใน",
        ),
    ]
    react_agent = create_react_agent(
        llm,
        tools,
        debug=True,
        prompt=prompt_template,
    )

    return react_agent