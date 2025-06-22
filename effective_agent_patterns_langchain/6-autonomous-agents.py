from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
import json

# Define the agent state using TypedDict
class AgentState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    goal: str
    search_results: Optional[List[Dict[str, Any]]]
    current_plan: Optional[str]
    iteration_count: int
    max_iterations: int
    completed: bool

# Define structured output models
class SearchResult(BaseModel):
    title: str
    price: float
    url: str
    description: str
    
class PlanAction(BaseModel):
    action_type: str = Field(description="Type of action: 'search', 'analyze', 'complete'")
    query: Optional[str] = Field(description="Search query if action_type is 'search'")
    reasoning: str = Field(description="Reasoning for this action")
    completed: bool = Field(default=False, description="Whether the goal has been achieved")

# Define tools
@tool
def search_guitars(query: str) -> List[Dict[str, Any]]:
    """Search for guitars based on a query string."""
    # Mock search results - in practice, this would call a real search API
    mock_results = [
        {
            "title": "Schecter Omen-8 8-String Electric Guitar",
            "price": 649.99,
            "url": "https://example.com/schecter-omen-8",
            "description": "Used 8-string guitar in excellent condition"
        },
        {
            "title": "Ibanez RG8 8-String Electric Guitar", 
            "price": 599.99,
            "url": "https://example.com/ibanez-rg8",
            "description": "Pre-owned 8-string with minor wear"
        },
        {
            "title": "ESP LTD M-8 8-String Guitar",
            "price": 750.00,
            "url": "https://example.com/esp-ltd-m8", 
            "description": "8-string guitar, slightly over budget"
        }
    ]
    
    # Filter results based on query relevance
    filtered_results = [r for r in mock_results if query.lower() in r["title"].lower() or query.lower() in r["description"].lower()]
    return filtered_results if filtered_results else mock_results

# Initialize LLM with tool calling
llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
tools = [search_guitars]
llm_with_tools = llm.bind_tools(tools)

# Create prompt template
planner_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an autonomous agent helping users find items to purchase.
    
    Your goal: {goal}
    
    You can use the following tools:
    - search_guitars: Search for guitars with a query string
    
    Based on the current state and previous messages, decide what action to take next.
    If you've found suitable options that meet the criteria, mark completed as True.
    
    Consider the user's budget and requirements carefully."""),
    ("placeholder", "{messages}")
])

def should_continue(state: AgentState) -> str:
    """Determine whether to continue or end the conversation."""
    if state["completed"] or state["iteration_count"] >= state["max_iterations"]:
        return "end"
    return "continue"

def planning_node(state: AgentState) -> AgentState:
    """Plan the next action based on current state."""
    messages = state["messages"]
    goal = state["goal"]
    
    # Format the messages directly using the prompt template
    formatted_messages = planner_prompt.format_messages(
        goal=goal,
        messages=messages
    )
    
    # Get the planned action
    response = llm_with_tools.invoke(formatted_messages)
    
    # Update state
    updated_state = state.copy()
    updated_state["messages"].append(response)
    updated_state["iteration_count"] += 1
    
    return updated_state

def tool_node(state: AgentState) -> AgentState:
    """Execute tools based on the last AI message."""
    tool_node_instance = ToolNode(tools)
    result = tool_node_instance.invoke(state)
    return result

def analysis_node(state: AgentState) -> AgentState:
    """Analyze results and determine if goal is completed."""
    messages = state["messages"]
    goal = state["goal"]
    
    # Check if we have tool results to analyze
    last_message = messages[-1] if messages else None
    
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""Analyze the search results to determine if they meet the goal: {goal}
        
        Look for:
        1. 8-string guitars
        2. Under $700 price range
        3. Used/pre-owned condition acceptable
        
        If you find suitable options, provide a summary and mark as completed.
        If not, suggest what to search for next."""),
        ("human", "Based on the current results, have we achieved the goal? Provide analysis.")
    ])
    
    # Format the messages properly
    formatted_messages = analysis_prompt.format_messages()
    response = llm.invoke(formatted_messages)
    
    # Simple completion check - in practice, you'd want more sophisticated logic
    completed = "suitable" in response.content.lower() and "under" in response.content.lower()
    
    updated_state = state.copy()
    updated_state["messages"].append(response)
    updated_state["completed"] = completed
    
    return updated_state

# Build the graph
def create_autonomous_agent():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("planner", planning_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("analyzer", analysis_node)
    
    # Add edges
    workflow.add_edge(START, "planner")
    workflow.add_conditional_edges(
        "planner",
        lambda state: "tools" if state["messages"][-1].tool_calls else "analyzer"
    )
    workflow.add_edge("tools", "analyzer") 
    workflow.add_conditional_edges("analyzer", should_continue, {
        "continue": "planner",
        "end": END
    })
    
    return workflow.compile()

# Example usage
def run_autonomous_agent(goal: str, max_iterations: int = 5):
    """Run the autonomous agent with a given goal."""
    
    # Initialize state
    initial_state: AgentState = {
        "messages": [HumanMessage(content=f"Help me with this goal: {goal}")],
        "goal": goal,
        "search_results": None,
        "current_plan": None,
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "completed": False
    }
    
    # Create and run the agent
    agent = create_autonomous_agent()
    
    print(f"🎯 Goal: {goal}")
    print("🤖 Starting autonomous agent...\n")
    
    try:
        final_state = agent.invoke(initial_state)
        
        print("📊 Final Results:")
        print(f"- Iterations used: {final_state['iteration_count']}")
        print(f"- Goal completed: {final_state['completed']}")
        
        # Print the conversation
        print("\n💬 Conversation History:")
        for i, message in enumerate(final_state["messages"]):
            if hasattr(message, 'content'):
                role = "Human" if isinstance(message, HumanMessage) else "AI"
                print(f"{i+1}. {role}: {message.content[:200]}{'...' if len(message.content) > 200 else ''}")
                
        return final_state
        
    except Exception as e:
        print(f"❌ Error running agent: {e}")
        return None

if __name__ == "__main__":
    # Run the example
    result = run_autonomous_agent("Find me a used 8-string guitar under $700")