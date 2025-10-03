from typing import Annotated, Iterator, List, Literal, TypedDict

from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, convert_to_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import END, StateGraph, add_messages
from langgraph.graph import add_messages
from .tools import get_retriever_from_forum

class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    documents: list[Document]
    question: str
    candidate_answer: str
    reason: str
    retries: int

class GraphConfig(TypedDict):
    max_retries: int

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


HALLUCINATION_GRADER_SYSTEM = (
"""
You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
Give a binary score 'yes' or 'no', where 'yes' means that the answer is grounded in / supported by the set of facts.

IF the generation includes code examples, make sure those examples are FULLY present in the set of facts, otherwise always return score 'no'.
"""
)

HALLUCINATION_GRADER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", HALLUCINATION_GRADER_SYSTEM),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)


class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


ANSWER_GRADER_SYSTEM = (
"""
You are a grader assessing whether an answer addresses / resolves a question.
Give a binary score 'yes' or 'no', where 'yes' means that the answer resolves the question.
"""
)

ANSWER_GRADER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", ANSWER_GRADER_SYSTEM),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

class AgnosAgent : 
    def __init__(self, max_retires=3, verbose=True):
        self.max_retires = max_retires
        self.verbose = verbose
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.retriever = get_retriever_from_forum(file_path="src/data/data.xlsx")

    def document_search(self, state: GraphState) -> dict:
        """
        Retrieve course documents
        """
        if self.verbose:
            print("---RETRIEVE---")

        question = convert_to_messages(state["messages"])[-1].content

        # Retrieval
        documents = self.retriever.invoke(question)

        return {"documents": documents, "question": question}

    def generate(self, state: GraphState):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        if self.verbose:
            print("---GENERATE---")
        
        question = state["question"]
        documents = state["documents"]
        retries = state["retries"] if state.get("retries") is not None else -1

        # # Generate  Answer
        # GENERATE_SYSTEM = (
        # """
        # You are a trustworthy and helpful consultation assistant for Agnos.
        # Your main responsibility is to assist users by providing advice and guidance based on previous answers by medical professionals.
        # If you don't know the answer, just say that you don't know.
        # Use three sentences maximum and keep the answer concise.
        # """
        # )

        # GENERATE_SYSTEM_PROMPT = ChatPromptTemplate.from_messages(
        #     [
        #         ("system", GENERATE_SYSTEM),
        #         ("human", "Question: \n\n {question} \n\n nContext: {context}\nAnswer:"),
        #     ]
        # )

        RAG_PROMPT: ChatPromptTemplate = hub.pull("rlm/rag-prompt")
        rag_chain = RAG_PROMPT | ChatOpenAI(model="gpt-4o", temperature=0) | StrOutputParser()
        generation = rag_chain.invoke({"context": documents, "question": question})
        return {"retries": retries + 1, "candidate_answer": generation}

    def transform_query(self, state: GraphState):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """

        if self.verbose:
            print("---TRANSFORM QUERY---")

        question = state["question"]

        # Re-write question
        QUERY_REWRITER_SYSTEM = (
        """
        You a question re-writer that converts an input question to a better version that is optimized for vectorstore retrieval.
        Look at the input and try to reason about the underlying semantic intent / meaning.
        """
        )

        QUERY_REWRITER_PROMPT = ChatPromptTemplate.from_messages(
            [
                ("system", QUERY_REWRITER_SYSTEM),
                (
                    "human",
                    "Here is the initial question: \n\n {question} \n Formulate an improved question.",
                ),
            ]
        )
        query_rewriter = QUERY_REWRITER_PROMPT | ChatOpenAI(model="gpt-4o", temperature=0) | StrOutputParser()
        better_question = query_rewriter.invoke({"question": question})
        return {"question": better_question}

    def finalize_response(self, state: GraphState):
        if self.verbose :
            print("---FINALIZING THE RESPONSE---")

        return {
            "messages": [AIMessage(content=state["candidate_answer"])],
            "documents": state["documents"]
        }

    def grade_generation_v_documents_and_question(self, state: GraphState, config) -> Literal["generate", "transform_query", "finalize_response"]:
        """
        Determines whether the generation is grounded in the document and answers question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """
        question = state["question"]
        documents = state["documents"]
        generation = state["candidate_answer"]
        retries = state["retries"] if state.get("retries") is not None else -1
        max_retries = config.get("configurable", {}).get("max_retries", self.max_retires)

        hallucination_grader = HALLUCINATION_GRADER_PROMPT | self.llm.with_structured_output(GradeHallucinations)
        hallucination_grade: GradeHallucinations = hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )

        # Check hallucination
        if hallucination_grade.binary_score == "no":
            return "generate" if retries < max_retries else "finalize_response"

        # Check question-answering
        answer_grader = ANSWER_GRADER_PROMPT | self.llm.with_structured_output(GradeAnswer)
        answer_grade: GradeAnswer = answer_grader.invoke({"question": question, "generation": generation})
        if answer_grade.binary_score == "yes":
            if self.verbose : print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "finalize_response"
        else:
            if self.verbose: print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "transform_query" if retries < max_retries else "finalize_response"
    
    def creat_workflow(self):  
        workflow = StateGraph(GraphState, config_schema=GraphConfig)

        # Define the nodes
        workflow.add_node("document_search", self.document_search)
        workflow.add_node("generate", self.generate)
        workflow.add_node("transform_query", self.transform_query)
        workflow.add_node("finalize_response", self.finalize_response)
        
        # Build graph
        workflow.set_entry_point("document_search")
        workflow.add_edge("document_search", "generate")
        workflow.add_edge("transform_query", "document_search")
        workflow.add_edge("finalize_response", END)

        workflow.add_conditional_edges(
            "generate",
            self.grade_generation_v_documents_and_question
        )

        # Compile
        graph = workflow.compile()

        return graph
