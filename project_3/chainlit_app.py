# ---------------------------------------------------------
# Chainlit Web Search Agent

import chainlit as cl
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, ToolMessage
from ddgs import DDGS


# ---------------------------------------------------------
# Define the web search tool
# ---------------------------------------------------------
def search_web_call(query: str) -> str:
    ddgs = DDGS()
    results = ddgs.text(query, max_results=5)
    formatted_results = "\n".join([f"{res['title']}: {res['href']}" for res in results])
    return formatted_results

@tool
def search_web(query: str) -> str:
    """
    Search the web for a given query and return the top results.

    Args:
        query (str): The search query.
    Returns:
        str: A formatted string containing the top search results (titles and URLs).
    """
    return search_web_call(query)

# ---------------------------------------------------------
# Create the agent (once at startup)
# ---------------------------------------------------------

llm = ChatOllama(model="llama3.2:3b")
agent = create_react_agent(llm, tools=[search_web])


# ---------------------------------------------------------
# Chainlit message handler
# ---------------------------------------------------------
@cl.on_message
async def handle_message(message: cl.Message):
    """Handle user messages and stream agent responses."""

    # Send the user message to the agent and return the final response
    result = await agent.ainvoke({"messages": [{"role": "user", "content": message.content}]})
    messages = result.get("messages", [])
    if messages:
        last_message = messages[-1]
        if isinstance(last_message, ToolMessage):
            tool_result = await last_message.execute()
            await cl.Message(content=tool_result).send()
        elif isinstance(last_message, AIMessage):
            await cl.Message(content=last_message.content).send()


# ---------------------------------------------------------
# Welcome message
# ---------------------------------------------------------
@cl.on_chat_start
async def start(): 
    await cl.Message(content="Hello! Ask me anything and I'll search the web for you!").send()
