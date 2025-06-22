import asyncio
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Initialize the LLM - using environment variable for API key
llm = ChatOpenAI(
    model="gpt-3.5-turbo", 
    temperature=0,
    # OpenAI API key should be set in environment as OPENAI_API_KEY
)

# Sample article split into paragraphs for parallel processing
article = """Quantum physics, also known as quantum mechanics, is a fundamental theory in physics that describes the behavior of matter and energy at the atomic and subatomic scales. Unlike classical physics, quantum mechanics reveals that particles can exist in multiple states simultaneously through superposition.

The famous double-slit experiment demonstrates wave-particle duality, showing that particles like electrons can behave as both particles and waves. When individual electrons are fired at a screen with two slits, they create an interference pattern as if they were waves passing through both slits simultaneously.

Quantum entanglement is another bizarre phenomenon where two or more particles become correlated in such a way that the quantum state of each particle cannot be described independently. Einstein called this "spooky action at a distance" because measuring one particle instantly affects its entangled partner, regardless of the distance between them.

The Heisenberg Uncertainty Principle states that we cannot simultaneously know both the exact position and momentum of a particle with arbitrary precision. This is not due to limitations in measurement equipment, but rather a fundamental property of quantum systems.

Quantum tunneling occurs when particles pass through energy barriers that they classically shouldn't be able to overcome. This phenomenon is essential for nuclear fusion in stars and is utilized in various technologies like scanning tunneling microscopes."""

async def summarize_paragraph(paragraph: str) -> str:
    """Asynchronously summarize a single paragraph using langchain's async invoke."""
    try:
        messages = [HumanMessage(content=f"{paragraph}\n\nGive a one-sentence summary of this paragraph:")]
        response = await llm.ainvoke(messages)
        return response.content.strip()
    except Exception as e:
        return f"Error summarizing paragraph: {str(e)}"

async def main():
    """Main function demonstrating parallel processing with async langchain."""
    print("🚀 Starting parallel processing of article paragraphs...\n")
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY environment variable not set")
        print("   Set it with: export OPENAI_API_KEY='your-key-here'")
        return
    
    # Split article into paragraphs
    paragraphs = [p.strip() for p in article.split("\n\n") if p.strip()]
    print(f"📝 Processing {len(paragraphs)} paragraphs in parallel...\n")
    
    # Use asyncio.gather to run all summarizations concurrently
    summaries = await asyncio.gather(*[summarize_paragraph(p) for p in paragraphs])
    
    # Display results
    print("📋 Parallel Processing Results:")
    print("=" * 50)
    for i, summary in enumerate(summaries, 1):
        print(f"{i}. {summary}")
    
    print(f"\n✅ Successfully processed {len(summaries)} paragraphs concurrently!")

if __name__ == "__main__":
    asyncio.run(main())
