from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

# Initialize the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

@tool
def calc(expr: str) -> str:
    """Compute a simple math expression."""
    import math, operator as op
    try:
        return str(eval(expr, {"__builtins__": {}}, {"math": math, "op": op}))
    except Exception as e:
        return f"Error: {str(e)} - Please provide a valid mathematical expression"

@tool
def get_gdp(country: str) -> str:
    """Get the GDP of a country in trillion USD (approximate 2023 data)."""
    gdp_data = {
        "japan": 4.2,
        "germany": 4.5,
        "usa": 26.9,
        "china": 17.7,
        "united states": 26.9,
        "uk": 3.1,
        "united kingdom": 3.1,
        "france": 2.9,
        "india": 3.7,
        "italy": 2.1,
        "canada": 2.1,
        "south korea": 1.8,
        "russia": 2.2,
        "brazil": 2.1,
        "australia": 1.6
    }
    country_lower = country.lower()
    if country_lower in gdp_data:
        return f"GDP of {country}: ${gdp_data[country_lower]} trillion USD"
    else:
        return f"GDP data not available for {country}"

# Define the tools
tools = [calc, get_gdp]

# Create the react agent using LangGraph (most modern approach)
agent_executor = create_react_agent(llm, tools)

# Run the agent
messages = [HumanMessage(content="GDP of Japan / GDP of Germany")]
result = agent_executor.invoke({"messages": messages})

# Print the final response
print(result["messages"][-1].content) 