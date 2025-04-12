"""Define a custom Reasoning and Action agent with legal drafting capabilities.

Works with a chat model with tool calling support.
"""

from datetime import datetime, timezone
from typing import Dict, List, Literal, cast

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from react_agent.configuration import Configuration
from react_agent.state import InputState, State
from react_agent.tools import TOOLS  # This includes both create_word_doc and search
from react_agent.utils import load_chat_model
from react_agent.prompts import SYSTEM_PROMPT, LEGAL_DOCUMENT_PROMPT, LEGAL_RESEARCH_PROMPT, CONTRACT_DRAFTING_PROMPT

# Define the function that calls the model
async def call_model(
    state: State, config: RunnableConfig
) -> Dict[str, List[AIMessage]]:
    """Call the LLM powering our "agent".

    This function prepares the prompt, initializes the model, and processes the response.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the model run.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    configuration = Configuration.from_runnable_config(config)

    # Initialize the model with tool binding.
    model = load_chat_model(configuration.model).bind_tools(TOOLS)

    # Get the user's messages to analyze context and detect if we're in information gathering mode
    user_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
    ai_messages = [msg for msg in state.messages if isinstance(msg, AIMessage)]
    
    # Get the most recent user message
    last_user_message = user_messages[-1].content.lower() if user_messages else ""
    
    # Check if we're in an information gathering phase
    information_gathering_mode = False
    document_requested = False
    
    # Check if a document was initially requested
    for msg in user_messages:
        content = msg.content.lower()
        if any(term in content for term in [
            "draft", "create document", "write a", "document", "contract", 
            "agreement", "nda", "license", "lease", "letter"
        ]):
            document_requested = True
            break
    
    # Check if we're already in the information gathering phase
    if document_requested and ai_messages:
        for msg in ai_messages:
            content = msg.content.lower() if isinstance(msg.content, str) else ""
            # Check if previous AI messages asked questions about document details
            if any(term in content for term in [
                "what is the name", "what are the parties", "could you provide", 
                "what term", "what jurisdiction", "need some information", 
                "need to know", "please provide", "can you tell me",
                "would you like", "do you want", "should i include"
            ]):
                information_gathering_mode = True
                break
    
    # Select the appropriate system prompt based on the user's request
    system_prompt = configuration.system_prompt
    
    # Use specialized prompts based on detected intent
    if configuration.system_prompt == SYSTEM_PROMPT:  # Only apply if using default prompt
        if any(term in last_user_message for term in ["contract", "agreement", "nda", "license"]):
            system_prompt = CONTRACT_DRAFTING_PROMPT
        elif any(term in last_user_message for term in ["draft", "create document", "write a", "document"]):
            system_prompt = LEGAL_DOCUMENT_PROMPT
        elif any(term in last_user_message for term in ["research", "find cases", "legal precedent", "statute"]):
            system_prompt = LEGAL_RESEARCH_PROMPT
    
    # Format the system prompt with current time
    system_message = system_prompt.format(
        system_time=datetime.now(tz=timezone.utc).isoformat()
    )
    
    # If a document was requested and we're not in information gathering mode,
    # add a special instruction to ensure the model asks for information first
    if document_requested and not information_gathering_mode and len(user_messages) == 1:
        system_message += """
        
IMPORTANT INSTRUCTION: The user has requested a document to be created. 
DO NOT create the document yet. First, ask the user for all necessary information 
needed to customize the document to their specific needs. Ask about parties involved, 
dates, terms, jurisdiction, and other relevant details.
"""

    # Get the model's response
    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages], config
        ),
    )

    # If this is the first response to a document request, make sure we're not calling 
    # document creation tools yet - we should be gathering information first
    if document_requested and len(ai_messages) == 0 and response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call.name == "create_word_doc":
                # Replace with a response that asks for information instead
                return {
                    "messages": [
                        AIMessage(
                            id=response.id,
                            content="I'd be happy to help you draft that document. To make it properly tailored to your needs, I'll need to ask you a few questions first:\n\n"
                                   "1. Who are the parties involved in this document?\n"
                                   "2. What specific terms or conditions should be included?\n"
                                   "3. What is the effective date and duration?\n"
                                   "4. Is there a specific jurisdiction this should be governed by?\n"
                                   "5. Are there any special clauses or provisions you'd like to include?\n\n"
                                   "Once I have this information, I can create a customized document for you."
                        )
                    ]
                }

    # Handle the case when it's the last step and the model still wants to use a tool
    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not complete the legal drafting task in the specified number of steps.",
                )
            ]
        }

    # Return the model's response as a list to be added to existing messages
    return {"messages": [response]}


# Define a new graph
builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Define the two nodes we will cycle between
builder.add_node(call_model)
builder.add_node("tools", ToolNode(TOOLS))

# Set the entrypoint as `call_model`
# This means that this node is the first one called
builder.add_edge("__start__", "call_model")


def route_model_output(state: State) -> Literal["__end__", "tools"]:
    """Determine the next node based on the model's output.

    This function checks if the model's last message contains tool calls.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call ("__end__" or "tools").
    """
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        return "__end__"
    # Otherwise we execute the requested actions
    return "tools"


# Add a conditional edge to determine the next step after `call_model`
builder.add_conditional_edges(
    "call_model",
    # After call_model finishes running, the next node(s) are scheduled
    # based on the output from route_model_output
    route_model_output,
)

# Add a normal edge from `tools` to `call_model`
# This creates a cycle: after using tools, we always return to the model
builder.add_edge("tools", "call_model")

# Compile the builder into an executable graph
graph = builder.compile(
    interrupt_before=[],  # Add node names here to update state before they're called
    interrupt_after=[],  # Add node names here to update state after they're called
)
graph.name = "Legal ReAct Agent"  # This customizes the name in LangSmith

# Example runner function to make it easy to execute the agent
async def run_legal_agent(query: str, configuration: Configuration = None):
    """Run the legal agent with a specific query.
    
    Args:
        query: The user's question or request
        configuration: Optional configuration overrides
        
    Returns:
        The agent's response
    """
    if configuration is None:
        configuration = Configuration(
            model="anthropic/claude-3-5-haiku-latest",
            max_search_results=5,
            system_prompt=SYSTEM_PROMPT  # Uses the default prompt
        )
    
    input_state = {
        "messages": [HumanMessage(content=query)],
        "config": configuration
    }
    
    return await graph.ainvoke(input_state)


# Example usage showing how to run the agent:
"""
import asyncio
from react_agent.graph import run_legal_agent
from react_agent.configuration import Configuration

# Define a custom configuration (optional)
config = Configuration(
    model="anthropic/claude-3-5-haiku-latest",
    max_steps=10,
    max_search_results=5,
    system_prompt=LEGAL_DOCUMENT_PROMPT  # Use specialized legal document prompt
)

# Example query
query = "Draft a non-disclosure agreement for a software development project with a 2-year term."

# Run the agent
async def main():
    result = await run_legal_agent(query, config)
    for message in result["messages"]:
        print(f"[{message.type}]: {message.content}")

if __name__ == "__main__":
    asyncio.run(main())
"""













































# """Define a custom Reasoning and Action agent.

# Works with a chat model with tool calling support.
# """

# from datetime import datetime, timezone
# from typing import Dict, List, Literal, cast

# from langchain_core.messages import AIMessage
# from langchain_core.runnables import RunnableConfig
# from langgraph.graph import StateGraph
# from langgraph.prebuilt import ToolNode

# from react_agent.configuration import Configuration
# from react_agent.state import InputState, State
# from react_agent.tools import TOOLS
# from react_agent.utils import load_chat_model

# # Define the function that calls the model


# async def call_model(
#     state: State, config: RunnableConfig
# ) -> Dict[str, List[AIMessage]]:
#     """Call the LLM powering our "agent".

#     This function prepares the prompt, initializes the model, and processes the response.

#     Args:
#         state (State): The current state of the conversation.
#         config (RunnableConfig): Configuration for the model run.

#     Returns:
#         dict: A dictionary containing the model's response message.
#     """
#     configuration = Configuration.from_runnable_config(config)

#     # Initialize the model with tool binding. Change the model or add more tools here.
#     model = load_chat_model(configuration.model).bind_tools(TOOLS)

#     # Format the system prompt. Customize this to change the agent's behavior.
#     system_message = configuration.system_prompt.format(
#         system_time=datetime.now(tz=timezone.utc).isoformat()
#     )

#     # Get the model's response
#     response = cast(
#         AIMessage,
#         await model.ainvoke(
#             [{"role": "system", "content": system_message}, *state.messages], config
#         ),
#     )

#     # Handle the case when it's the last step and the model still wants to use a tool
#     if state.is_last_step and response.tool_calls:
#         return {
#             "messages": [
#                 AIMessage(
#                     id=response.id,
#                     content="Sorry, I could not find an answer to your question in the specified number of steps.",
#                 )
#             ]
#         }

#     # Return the model's response as a list to be added to existing messages
#     return {"messages": [response]}


# # def human_review_node(state) -> Command[Literal["call_llm", "run_tool"]]:
# #     # This is the value we'll be providing via Command(resume=<human_review>)
# #     human_review = interrupt(
# #         {
# #             "question": "Is this correct?",
# #             # Surface tool calls for review
# #             "tool_call": tool_call
# #         }
# #     )

# #     review_action, review_data = human_review

# #     # Approve the tool call and continue
# #     if review_action == "continue":
# #         return Command(goto="run_tool")

# #     # Modify the tool call manually and then continue
# #     elif review_action == "update":
# #         ...
# #         updated_msg = get_updated_msg(review_data)
# #         # Remember that to modify an existing message you will need
# #         # to pass the message with a matching ID.
# #         return Command(goto="run_tool", update={"messages": [updated_message]})

# #     # Give natural language feedback, and then pass that back to the agent
# #     elif review_action == "feedback":
# #         ...
# #         feedback_msg = get_feedback_msg(review_data)
# #         return Command(goto="call_llm", update={"messages": [feedback_msg]})




# # Define a new graph

# builder = StateGraph(State, input=InputState, config_schema=Configuration)

# # Define the two nodes we will cycle between
# builder.add_node(call_model)
# builder.add_node("tools", ToolNode(TOOLS))

# # Set the entrypoint as `call_model`
# # This means that this node is the first one called
# builder.add_edge("__start__", "call_model")


# def route_model_output(state: State) -> Literal["__end__", "tools"]:
#     """Determine the next node based on the model's output.

#     This function checks if the model's last message contains tool calls.

#     Args:
#         state (State): The current state of the conversation.

#     Returns:
#         str: The name of the next node to call ("__end__" or "tools").
#     """
#     last_message = state.messages[-1]
#     if not isinstance(last_message, AIMessage):
#         raise ValueError(
#             f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
#         )
#     # If there is no tool call, then we finish
#     if not last_message.tool_calls:
#         return "__end__"
#     # Otherwise we execute the requested actions
#     return "tools"


# # Add a conditional edge to determine the next step after `call_model`
# builder.add_conditional_edges(
#     "call_model",
#     # After call_model finishes running, the next node(s) are scheduled
#     # based on the output from route_model_output
#     route_model_output,
# )

# # Add a normal edge from `tools` to `call_model`
# # This creates a cycle: after using tools, we always return to the model
# builder.add_edge("tools", "call_model")

# # Compile the builder into an executable graph
# # You can customize this by adding interrupt points for state updates
# graph = builder.compile(
#     interrupt_before=[],  # Add node names here to update state before they're called
#     interrupt_after=[],  # Add node names here to update state after they're called
# )
# graph.name = "ReAct Agent"  # This customizes the name in LangSmith






# # from datetime import datetime, timezone
# # from typing import Dict, List, Literal, cast

# # from langchain_core.messages import AIMessage
# # from langchain_core.runnables import RunnableConfig
# # from langgraph.graph import StateGraph
# # from langgraph.prebuilt import ToolNode
# # from langgraph.graph import Command, interrupt

# # from react_agent.configuration import Configuration
# # from react_agent.state import InputState, State
# # from react_agent.tools import TOOLS
# # from react_agent.utils import load_chat_model
# # # Possible alternative 1













# # async def call_model(
# #     state: State, 
# #     config: RunnableConfig
# # ) -> Dict[str, List[AIMessage]]:
# #     """Call the LLM powering our 'agent'."""

# #     configuration = Configuration.from_runnable_config(config)

# #     # Initialize the model with tool binding
# #     model = load_chat_model(configuration.model).bind_tools(TOOLS)

# #     # Format the system prompt
# #     system_message = configuration.system_prompt.format(
# #         system_time=datetime.now(tz=timezone.utc).isoformat()
# #     )

# #     # Get the model's response
# #     response = cast(
# #         AIMessage,
# #         await model.ainvoke(
# #             [{"role": "system", "content": system_message}, *state.messages],
# #             config
# #         ),
# #     )

# #     # If it's our last step but the model still wants to call a tool, short-circuit
# #     if state.is_last_step and response.tool_calls:
# #         return {
# #             "messages": [
# #                 AIMessage(
# #                     id=response.id,
# #                     content=(
# #                         "Sorry, I could not find an answer to your question "
# #                         "in the specified number of steps."
# #                     ),
# #                 )
# #             ]
# #         }

# #     return {"messages": [response]}











# # def get_updated_msg(review_data: dict) -> AIMessage:
# #     """Given some user modifications from the HITL interface,
# #     create an updated AIMessage (e.g., with corrected tool call parameters).
# #     """
# #     # Example logic:
# #     new_content = review_data.get("updated_tool_call_instructions", "")
# #     return AIMessage(
# #         content=new_content, 
# #         tool_calls=[]  # Possibly you add or remove tool calls here
# #     )


# # def get_feedback_msg(review_data: dict) -> AIMessage:
# #     """Given some textual feedback from the human, wrap it in an AIMessage."""
# #     feedback = review_data.get("feedback", "")
# #     return AIMessage(content=feedback, tool_calls=[])


# # def human_review_node(state: State) -> Command[Literal["call_model", "tools"]]:
# #     """Interrupt to get a human review of the pending tool call."""
# #     # Check the last AIMessage (which should contain the tool call)
# #     last_message = state.messages[-1]
# #     tool_call = None
# #     if isinstance(last_message, AIMessage) and last_message.tool_calls:
# #         # For simplicity, just grab the first tool call if multiple exist
# #         tool_call = last_message.tool_calls[0]

# #     # The 'interrupt' function suspends execution and returns
# #     # the user's decision from a front-end or CLI
# #     human_review = interrupt(
# #         {
# #             "question": "Is this correct?",
# #             "tool_call": tool_call
# #         }
# #     )

# #     # The human_review might return something like ("continue", {...})
# #     review_action, review_data = human_review

# #     # Approve the tool call and continue
# #     if review_action == "continue":
# #         return Command(goto="tools")

# #     # Modify the tool call manually and then continue
# #     elif review_action == "update":
# #         updated_msg = get_updated_msg(review_data)
# #         # Overwrite the last AIMessage in state.messages
# #         return Command(
# #             goto="tools",
# #             update={"messages": [updated_msg]}  # Must have the same 'id' if we strictly track IDs
# #         )

# #     # Provide natural language feedback, then pass that back to the agent
# #     elif review_action == "feedback":
# #         feedback_msg = get_feedback_msg(review_data)
# #         return Command(
# #             goto="call_model",
# #             update={"messages": [feedback_msg]}
# #         )

# #     # Fallback if an unrecognized action came in
# #     return Command(goto="tools")



# # def get_updated_msg(review_data: dict) -> AIMessage:
# #     """Given some user modifications from the HITL interface,
# #     create an updated AIMessage (e.g., with corrected tool call parameters).
# #     """
# #     # Example logic:
# #     new_content = review_data.get("updated_tool_call_instructions", "")
# #     return AIMessage(
# #         content=new_content, 
# #         tool_calls=[]  # Possibly you add or remove tool calls here
# #     )


# # def get_feedback_msg(review_data: dict) -> AIMessage:
# #     """Given some textual feedback from the human, wrap it in an AIMessage."""
# #     feedback = review_data.get("feedback", "")
# #     return AIMessage(content=feedback, tool_calls=[])


# # def human_review_node(state: State) -> Command[Literal["call_model", "tools"]]:
# #     """Interrupt to get a human review of the pending tool call."""
# #     # Check the last AIMessage (which should contain the tool call)
# #     last_message = state.messages[-1]
# #     tool_call = None
# #     if isinstance(last_message, AIMessage) and last_message.tool_calls:
# #         # For simplicity, just grab the first tool call if multiple exist
# #         tool_call = last_message.tool_calls[0]

# #     # The 'interrupt' function suspends execution and returns
# #     # the user's decision from a front-end or CLI
# #     human_review = interrupt(
# #         {
# #             "question": "Is this correct?",
# #             "tool_call": tool_call
# #         }
# #     )

# #     # The human_review might return something like ("continue", {...})
# #     review_action, review_data = human_review

# #     # Approve the tool call and continue
# #     if review_action == "continue":
# #         return Command(goto="tools")

# #     # Modify the tool call manually and then continue
# #     elif review_action == "update":
# #         updated_msg = get_updated_msg(review_data)
# #         # Overwrite the last AIMessage in state.messages
# #         return Command(
# #             goto="tools",
# #             update={"messages": [updated_msg]}  # Must have the same 'id' if we strictly track IDs
# #         )

# #     # Provide natural language feedback, then pass that back to the agent
# #     elif review_action == "feedback":
# #         feedback_msg = get_feedback_msg(review_data)
# #         return Command(
# #             goto="call_model",
# #             update={"messages": [feedback_msg]}
# #         )

# #     # Fallback if an unrecognized action came in
# #     return Command(goto="tools")



# # builder = StateGraph(State, input=InputState, config_schema=Configuration)

# # # Add nodes
# # builder.add_node(call_model)
# # builder.add_node("human_review", human_review_node)
# # builder.add_node("tools", ToolNode(TOOLS))

# # # Entrypoint: start with call_model
# # builder.add_edge("__start__", "call_model")










# # def route_model_output(state: State) -> Literal["__end__", "human_review"]:
# #     """Route after model output: go to human review if there's a tool call,
# #     otherwise end the conversation.
# #     """
# #     last_message = state.messages[-1]
# #     if not isinstance(last_message, AIMessage):
# #         raise ValueError(
# #             f"Expected AIMessage, got {type(last_message).__name__}"
# #         )
# #     # No tool calls => conversation is done
# #     if not last_message.tool_calls:
# #         return "__end__"

# #     # If there's a tool call => go to our new 'human_review_node'
# #     return "human_review"


# # # After 'call_model' is called, we see if we must do HITL review or end
# # builder.add_conditional_edges("call_model", route_model_output)






# # builder.add_edge("tools", "call_model")





# # graph = builder.compile(
# #     interrupt_before=[],  # Insert node names here if you want state modifications BEFORE they run
# #     interrupt_after=[],   # Insert node names here if you want state modifications AFTER they run
# # )
# # graph.name = "ReAct Agent with Human-in-the-Loop"






# # # Or some other module
