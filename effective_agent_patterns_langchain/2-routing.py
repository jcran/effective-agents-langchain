"""
Advanced Routing Pattern with Modern LangChain Best Practices

This example demonstrates intelligent routing between different LLM models
based on query characteristics, using the latest recommended approaches.
"""

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, Any


# Initialize models with latest recommended versions
# gpt-4o-mini is the recommended replacement for gpt-3.5-turbo (faster, cheaper, more capable)
# gpt-4o is the latest stable version of GPT-4
fast_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
smart_model = ChatOpenAI(model="gpt-4o", temperature=0)


def intelligent_router(input_data: Dict[str, Any]) -> Any:
    """
    Intelligent routing function that selects the appropriate model
    based on query characteristics and complexity indicators.
    
    Args:
        input_data: Dictionary containing the query and routing decision
        
    Returns:
        The appropriate model chain for the given input
    """
    query = input_data["query"].lower()
    
    # Enhanced routing logic with more sophisticated patterns
    complexity_indicators = [
        "explain in detail", "comprehensive", "thorough analysis", 
        "step by step", "complex", "intricate", "elaborate"
    ]
    
    quick_indicators = [
        "quick", "brief", "short", "tldr", "summary", 
        "simple", "basic", "overview"
    ]
    
    # Route to smart model for complex tasks
    if any(indicator in query for indicator in complexity_indicators):
        return smart_model | StrOutputParser()
    
    # Route to fast model for quick tasks
    if any(indicator in query for indicator in quick_indicators):
        return fast_model | StrOutputParser()
    
    # Analyze query length and complexity
    word_count = len(query.split())
    if word_count > 20 or any(word in query for word in ["analyze", "compare", "evaluate", "research"]):
        return smart_model | StrOutputParser()
    
    # Default to fast model for efficiency
    return fast_model | StrOutputParser()


def create_routing_chain():
    """
    Creates a routing chain using modern LCEL patterns.
    
    Returns:
        A runnable chain that routes queries to appropriate models
    """
    # Create the routing chain using LCEL
    routing_chain = (
        {"query": lambda x: x["query"]}
        | RunnableLambda(intelligent_router)
    )
    
    return routing_chain


def get_user_question() -> str:
    """Get a question from the user via command line input"""
    print("\n" + "="*60)
    print("🤖 INTELLIGENT MODEL ROUTER")
    print("="*60)
    print("This system automatically routes your query to the most appropriate model:")
    print("• 🚀 GPT-4o-mini: Fast responses for simple queries")
    print("• 🧠 GPT-4o: Detailed analysis for complex questions")
    print("\nEnter your question (or 'quit' to exit):")
    question = input("> ").strip()
    return question


def main():
    """Main execution function with error handling"""
    router = create_routing_chain()
    
    while True:
        try:
            question = get_user_question()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
                
            if not question:
                print("❌ Please enter a valid question.")
                continue
            
            print(f"\n🔄 Processing: '{question}'")
            print("-" * 60)
            
            # Route and process the query
            response = router.invoke({"query": question})
            
            # Display the response
            print("📋 Response:")
            print(response)
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error occurred: {str(e)}")
            print("Please try again with a different question.")


if __name__ == "__main__":
    # Example usage demonstrations
    print("🚀 LANGCHAIN ROUTING EXAMPLES")
    print("="*60)
    
    router = create_routing_chain()
    
    # Test different types of queries
    test_queries = [
        {
            "query": "Give me a quick explanation of Python",
            "expected": "Fast model (GPT-4o-mini)"
        },
        {
            "query": "Explain in detail how neural networks work and provide comprehensive examples",
            "expected": "Smart model (GPT-4o)"
        },
        {
            "query": "What is machine learning?",
            "expected": "Fast model (GPT-4o-mini)"
        }
    ]
    
    for test in test_queries:
        print(f"\n📝 Query: {test['query']}")
        print(f"🎯 Expected routing: {test['expected']}")
        
        try:
            response = router.invoke({"query": test["query"]})
            print(f"✅ Response: {response[:100]}...")  # Show first 100 chars
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print("-" * 40)
    
    # Start interactive mode
    print("\n🎮 Starting interactive mode...")
    main()