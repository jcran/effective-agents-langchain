# Effective AI Agent Patterns with LangChain

A collection of proven patterns for building effective AI agents using LangChain and LangGraph. These patterns demonstrate different approaches to agent architecture, from simple tool usage to complex autonomous systems. These patterns are based on the Anthropic blog post ["Building Effective AI Agents"](https://www.anthropic.com/engineering/building-effective-agents)

## 🚀 Quick Start

```bash
# Install dependencies
uv sync

# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Run any pattern
python effective_agent_patterns_langchain/0-augmented-llm.py
```

## 📚 Pattern Overview

### 0. Augmented LLM Pattern
**Files:** `0-augmented-llm.py`, `0-augmented-llm-langgraph.py`

**What it does:** Extends LLM capabilities with external tools and functions.

**Key Features:**
- Tool integration (calculator, data lookup)
- ReAct (Reasoning + Acting) pattern
- Two implementations: traditional LangChain and modern LangGraph

**Use Cases:** 
- Mathematical computations
- Data retrieval and analysis
- API integrations

**Pattern:** Simple agent with predefined tools that can reason about when and how to use them.

---

### 1. Prompt Chaining Pattern
**File:** `1-prompt-chaining.py`

**What it does:** Sequences multiple LLM calls to create complex workflows.

**Key Features:**
- Sequential chain execution
- Output validation steps
- Multi-stage content generation

**Use Cases:**
- Content creation workflows
- Multi-step analysis
- Quality assurance processes

**Pattern:** `Input → Outline → Validate → Draft → Output`

---

### 2. Routing Pattern
**File:** `2-routing.py`

**What it does:** Intelligently routes queries to different models based on complexity.

**Key Features:**
- Keyword-based routing logic
- Cost optimization (fast vs. smart models)
- Dynamic model selection

**Use Cases:**
- Cost-efficient processing
- Performance optimization
- Resource management

**Pattern:** `Query Analysis → Route Decision → Model Selection → Response`

---

### 3. Parallelization Pattern
**File:** `3-parallelization.py`

**What it does:** Processes multiple tasks concurrently for improved performance.

**Key Features:**
- Async/await implementation
- Concurrent LLM calls
- Batch processing

**Use Cases:**
- Large document processing
- Bulk summarization
- Parallel analysis tasks

**Pattern:** `Split Work → Process Concurrently → Gather Results → Combine`

---

### 4. Orchestrator Pattern
**File:** `4-orchestrator.py`

**What it does:** Coordinates multiple agents or tasks with a central manager.

**Key Features:**
- Task planning and decomposition
- Parallel task execution
- Result aggregation

**Use Cases:**
- Complex project management
- Multi-step research tasks
- Coordinated analysis

**Pattern:** `Plan → Assign → Execute → Monitor → Aggregate`

---

### 5. Evaluator-Optimizer Pattern
**File:** `5-evaluator-optimizer.py`

**What it does:** Iteratively improves content through evaluation and refinement.

**Key Features:**
- Multi-perspective evaluation
- Structured feedback loops
- Quality scoring systems
- Iterative improvement

**Use Cases:**
- Content quality improvement
- Writing enhancement
- Performance optimization

**Pattern:** `Generate → Evaluate → Critique → Improve → Repeat`

---

### 6. Autonomous Agents Pattern
**File:** `6-autonomous-agents.py`

**What it does:** Creates self-directed agents that pursue goals independently.

**Key Features:**
- Goal-oriented behavior
- State management with LangGraph
- Decision-making loops
- Tool usage planning

**Use Cases:**
- Independent research tasks
- Shopping assistants
- Automated workflows

**Pattern:** `Goal → Plan → Act → Observe → Decide → Continue/Complete`

---

## 🔧 Technical Implementation

### Modern LangChain Updates

This project uses the latest LangChain patterns:

- ✅ `create_react_agent()` instead of deprecated `initialize_agent()`
- ✅ `@tool` decorator from `langchain_core.tools`
- ✅ Proper async support with `ainvoke()`
- ✅ Structured outputs with Pydantic models
- ✅ LangGraph for complex agent workflows

### Dependencies

```toml
dependencies = [
    "langchain-core",      # Core abstractions
    "langchain-community", # Community integrations  
    "langchain-openai",    # OpenAI integration
    "langchain",           # Main package + hub
    "langgraph"           # Graph-based agent framework
]
```

## 🎯 When to Use Each Pattern

| Pattern | Complexity | Use Case | Best For |
|---------|------------|----------|----------|
| **Augmented LLM** | Low | Tool integration | Simple Q&A with tools |
| **Prompt Chaining** | Low-Medium | Sequential workflows | Content creation |
| **Routing** | Low | Cost optimization | Multi-model systems |
| **Parallelization** | Medium | Performance | Bulk processing |
| **Orchestrator** | Medium-High | Task coordination | Complex projects |
| **Evaluator-Optimizer** | High | Quality improvement | Content refinement |
| **Autonomous** | High | Independent operation | Goal-driven agents |

## 🚦 Getting Started Guide

1. **Start Simple**: Begin with `0-augmented-llm.py` to understand basic tool usage
2. **Add Workflow**: Try `1-prompt-chaining.py` for multi-step processes  
3. **Optimize Performance**: Use `3-parallelization.py` for concurrent processing
4. **Scale Complexity**: Move to `4-orchestrator.py` for coordinated tasks
5. **Advanced Applications**: Explore `6-autonomous-agents.py` for independent agents

## 💡 Best Practices

- **Start with the simplest pattern** that meets your needs
- **Use async patterns** for performance-critical applications
- **Implement proper error handling** and timeouts
- **Monitor costs** when using multiple models
- **Test thoroughly** with edge cases
- **Consider fallback strategies** for tool failures

## 🔍 Pattern Selection Guide

**Choose Augmented LLM when:**
- You need basic tool integration
- Simple question-answering with data lookup
- Starting with agent development

**Choose Prompt Chaining when:**
- You have sequential, dependent steps
- Need validation or quality checks
- Building content generation pipelines

**Choose Routing when:**
- You want to optimize costs/performance
- Different complexity levels need different models
- Load balancing across models

**Choose Parallelization when:**
- Processing large volumes of similar tasks
- Time is critical
- Tasks are independent

**Choose Orchestrator when:**
- Managing complex, multi-step projects
- Coordinating different types of agents
- Need centralized task management

**Choose Evaluator-Optimizer when:**
- Quality is paramount
- Iterative improvement is valuable
- Need multi-perspective analysis

**Choose Autonomous when:**
- Agents need to operate independently
- Goal-directed behavior is required
- Complex decision-making is needed

## 📈 Migration from Legacy LangChain

If migrating from older LangChain code:

- Replace `initialize_agent()` → `create_react_agent()`
- Update imports: `langchain.tools` → `langchain_core.tools`
- Use `.invoke()` instead of `.run()`
- Consider migrating to LangGraph for complex workflows

## 🤝 Contributing

Feel free to contribute additional patterns or improvements to existing ones. Each pattern should be:
- Self-contained and runnable
- Well-documented with clear use cases
- Following modern LangChain best practices 
