from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
import asyncio

# Use current model parameter name and structured approach
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

class TaskPlan(BaseModel):
    """A structured task plan with numbered steps."""
    tasks: List[str] = Field(description="List of specific, actionable tasks")

class TaskResult(BaseModel):
    """Result of executing a task."""
    task: str = Field(description="The original task")
    result: str = Field(description="The task execution result")

# Structured prompts using ChatPromptTemplate
planning_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a task planner. Break down complex goals into specific, actionable tasks. Return only the tasks as a numbered list, one per line."),
    ("user", "Goal: {goal}\n\nBreak this into 3-5 specific tasks:")
])

worker_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a task executor. Execute the given task and provide a concise, helpful result."),
    ("user", "Task: {task}")
])

# Create chains with proper output parsing
planning_chain = planning_prompt | llm | StrOutputParser()
worker_chain = worker_prompt | llm | StrOutputParser()

async def worker(task: str) -> TaskResult:
    """Execute a single task asynchronously."""
    result = await worker_chain.ainvoke({"task": task})
    return TaskResult(task=task, result=result)

async def manager(goal: str) -> List[TaskResult]:
    """Orchestrate task execution with proper planning and parallel execution."""
    print(f"🎯 Goal: {goal}\n")
    
    # Step 1: Plan the tasks
    print("📋 Planning tasks...")
    plan_result = await planning_chain.ainvoke({"goal": goal})
    
    # Parse tasks from the numbered list
    tasks = []
    for line in plan_result.strip().split('\n'):
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
            # Remove numbering and formatting
            task = line.split('.', 1)[-1].strip() if '.' in line else line.strip('- •').strip()
            if task:
                tasks.append(task)
    
    print(f"Found {len(tasks)} tasks to execute\n")
    
    # Step 2: Execute tasks concurrently
    print("⚡ Executing tasks...")
    task_results = await asyncio.gather(*[worker(task) for task in tasks])
    
    # Step 3: Display results
    print("\n📊 Results:")
    for i, result in enumerate(task_results, 1):
        print(f"\n{i}. Task: {result.task}")
        print(f"   Result: {result.result}")
    
    return task_results

# Example usage with proper async execution
async def main():
    await manager("Research three open-source LLM evaluation frameworks and compare their features")

if __name__ == "__main__":
    asyncio.run(main())
