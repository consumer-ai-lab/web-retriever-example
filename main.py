from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv,find_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough
)
from langchain_community.chat_models import ChatOllama
from langchain.retrievers.tavily_search_api import TavilySearchAPIRetriever,SearchDepth
import os

load_dotenv(find_dotenv())

llm1 = ChatOllama(model="medllama2")
llm2 = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5,convert_system_message_to_human=True)
prompt = ChatPromptTemplate.from_template("""
     As a medical chatbot, please analyze the following context and question. Provide informative insights and advice strictly based on the given information. Remember, you cannot offer personalized diagnoses or treatment plans. Highlight the importance of seeking professional medical help.

    Context: {context}
    Question: {question}
""")
retriever = TavilySearchAPIRetriever(k=1,api_key=os.getenv("TAVILY_AI_API_KEY"),search_depth=SearchDepth.ADVANCED,include_generated_answer=True)

def get_context(res):
    result=retriever.invoke(res)
    return "\n".join(r.page_content for r in result) 


print(get_context("What is mango?"))

chain1 = (
    RunnableParallel({
        "question":RunnablePassthrough(),
        "context":get_context
    })
    | prompt
    | llm1
).with_types(input_type=str)

chain2 = (
    RunnableParallel({
        "question":RunnablePassthrough(),
        "context":get_context
    })
    | prompt
    | llm2
).with_types(input_type=str)




