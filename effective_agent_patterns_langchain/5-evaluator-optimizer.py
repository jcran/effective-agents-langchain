from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
from typing import List, Optional
import asyncio

# Modern model configuration with current naming
llm = ChatOpenAI(
    model="gpt-4o",  # Updated to current model
    temperature=0.7,
    max_tokens=2000
)

class CritiqueResult(BaseModel):
    """Structured critique output"""
    overall_quality: int = Field(description="Quality score from 1-10", ge=1, le=10)
    strengths: List[str] = Field(description="What works well in the content")
    weaknesses: List[str] = Field(description="Areas that need improvement")
    specific_suggestions: List[str] = Field(description="Concrete improvement suggestions")
    needs_revision: bool = Field(description="Whether content needs further revision")

class ImprovedContent(BaseModel):
    """Structured improved content output"""
    revised_content: str = Field(description="The improved version of the content")
    changes_made: List[str] = Field(description="List of specific changes made")
    confidence_score: float = Field(description="Confidence in improvement (0-1)", ge=0, le=1)

# Modern prompt templates with ChatPromptTemplate
draft_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert science communicator. Create clear, accurate, and engaging explanations."),
    ("user", "Explain {topic} in approximately {word_count} words. Make it accessible but scientifically accurate.")
])

critique_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a rigorous content critic and editor. Evaluate the given content and provide structured feedback.
    
    Focus on:
    - Scientific accuracy
    - Clarity and accessibility 
    - Engagement and flow
    - Completeness of explanation
    
    Be constructive but thorough in your critique."""),
    ("user", "Please critique this content:\n\n{content}")
])

improvement_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert content improver. Take the original content and the critique, then create an improved version.
    
    Apply the specific suggestions while maintaining the core message and approximate length."""),
    ("user", """Original content:
{original_content}

Critique and suggestions:
{critique}

Please provide an improved version that addresses the critique.""")
])

# Create structured output chains using with_structured_output
critique_chain = critique_prompt | llm.with_structured_output(CritiqueResult)
improvement_chain = improvement_prompt | llm.with_structured_output(ImprovedContent)

# Basic content generation chain
draft_chain = draft_prompt | llm | StrOutputParser()

async def evaluator_optimizer_workflow(topic: str, word_count: int = 200, max_iterations: int = 3):
    """
    Modern evaluator-optimizer pattern with structured outputs and async support
    """
    print(f"🚀 Starting evaluator-optimizer workflow for: {topic}")
    
    # Generate initial draft
    print("\n📝 Generating initial draft...")
    current_content = await draft_chain.ainvoke({
        "topic": topic, 
        "word_count": word_count
    })
    print(f"Initial draft:\n{current_content}\n")
    
    iteration = 0
    improvement_history = []
    
    while iteration < max_iterations:
        iteration += 1
        print(f"🔍 Iteration {iteration}: Evaluating and optimizing...")
        
        # Get structured critique
        critique = await critique_chain.ainvoke({"content": current_content})
        
        print(f"\n📊 Quality Score: {critique.overall_quality}/10")
        print(f"🎯 Needs Revision: {critique.needs_revision}")
        
        if critique.overall_quality >= 8 and not critique.needs_revision:
            print("✅ Content quality is satisfactory!")
            break
            
        print(f"\n💪 Strengths: {', '.join(critique.strengths)}")
        print(f"⚠️  Areas for improvement: {', '.join(critique.weaknesses)}")
        print(f"💡 Suggestions: {', '.join(critique.specific_suggestions)}")
        
        # Generate improved version
        improvement = await improvement_chain.ainvoke({
            "original_content": current_content,
            "critique": f"""
            Quality Score: {critique.overall_quality}/10
            Weaknesses: {'; '.join(critique.weaknesses)}
            Suggestions: {'; '.join(critique.specific_suggestions)}
            """
        })
        
        # Update content
        current_content = improvement.revised_content
        improvement_history.append({
            "iteration": iteration,
            "quality_score": critique.overall_quality,
            "changes": improvement.changes_made,
            "confidence": improvement.confidence_score
        })
        
        print(f"\n🔄 Changes made: {', '.join(improvement.changes_made)}")
        print(f"📈 Improvement confidence: {improvement.confidence_score:.2f}")
        print(f"\nRevised content:\n{current_content}\n")
        
        # Break if we're not making meaningful improvements
        if improvement.confidence_score < 0.3:
            print("⚠️  Low confidence in further improvements. Stopping.")
            break
    
    return {
        "final_content": current_content,
        "iterations_completed": iteration,
        "improvement_history": improvement_history,
        "final_quality_score": critique.overall_quality if 'critique' in locals() else None
    }

# Advanced pattern: Multi-perspective evaluation
class MultiPerspectiveCritique(BaseModel):
    """Critique from multiple expert perspectives"""
    scientific_accuracy: CritiqueResult = Field(description="Scientific accuracy evaluation")
    readability: CritiqueResult = Field(description="Readability and accessibility evaluation") 
    engagement: CritiqueResult = Field(description="Engagement and interest evaluation")
    overall_recommendation: str = Field(description="Overall recommendation for next steps")

multi_perspective_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a panel of three experts evaluating content:
    1. A scientific accuracy expert
    2. A readability and accessibility expert  
    3. An engagement and communication expert
    
    Provide separate evaluations from each perspective."""),
    ("user", "Please evaluate this content from all three expert perspectives:\n\n{content}")
])

multi_perspective_chain = multi_perspective_prompt | llm.with_structured_output(MultiPerspectiveCritique)

async def advanced_evaluator_optimizer(topic: str, word_count: int = 200):
    """
    Advanced pattern with multi-perspective evaluation
    """
    print(f"🎯 Advanced evaluator-optimizer for: {topic}")
    
    # Generate initial content
    content = await draft_chain.ainvoke({"topic": topic, "word_count": word_count})
    print(f"\nInitial content:\n{content}\n")
    
    # Multi-perspective evaluation
    multi_critique = await multi_perspective_chain.ainvoke({"content": content})
    
    print("📊 Multi-Perspective Evaluation:")
    print(f"🔬 Scientific Accuracy: {multi_critique.scientific_accuracy.overall_quality}/10")
    print(f"📖 Readability: {multi_critique.readability.overall_quality}/10") 
    print(f"🎪 Engagement: {multi_critique.engagement.overall_quality}/10")
    print(f"\n💭 Overall Recommendation: {multi_critique.overall_recommendation}")
    
    # Determine which aspect needs most improvement
    scores = {
        "scientific_accuracy": multi_critique.scientific_accuracy.overall_quality,
        "readability": multi_critique.readability.overall_quality,
        "engagement": multi_critique.engagement.overall_quality
    }
    
    lowest_aspect = min(scores, key=scores.get)
    lowest_score = scores[lowest_aspect]
    
    if lowest_score < 7:
        print(f"\n🎯 Focusing improvement on: {lowest_aspect.replace('_', ' ').title()}")
        
        # Get the specific critique for the weakest area
        specific_critique = getattr(multi_critique, lowest_aspect)
        
        # Targeted improvement
        targeted_improvement = await improvement_chain.ainvoke({
            "original_content": content,
            "critique": f"""
            Focus on improving {lowest_aspect.replace('_', ' ')}.
            Current score: {specific_critique.overall_quality}/10
            Weaknesses: {'; '.join(specific_critique.weaknesses)}
            Suggestions: {'; '.join(specific_critique.specific_suggestions)}
            """
        })
        
        print(f"\n✨ Targeted improvements made:")
        for change in targeted_improvement.changes_made:
            print(f"  • {change}")
        
        print(f"\n📝 Final optimized content:\n{targeted_improvement.revised_content}")
        
        return targeted_improvement.revised_content
    else:
        print("✅ All aspects score well (7+). Content is ready!")
        return content

# Demonstration function
async def main():
    """Demonstrate both patterns"""
    
    print("=" * 60)
    print("🔄 BASIC EVALUATOR-OPTIMIZER PATTERN")
    print("=" * 60)
    
    result = await evaluator_optimizer_workflow(
        topic="special relativity",
        word_count=200,
        max_iterations=3
    )
    
    print(f"\n🏁 Workflow completed in {result['iterations_completed']} iterations")
    if result['final_quality_score']:
        print(f"📊 Final quality score: {result['final_quality_score']}/10")
    
    print("\n" + "=" * 60)
    print("🎯 ADVANCED MULTI-PERSPECTIVE PATTERN")
    print("=" * 60)
    
    final_content = await advanced_evaluator_optimizer(
        topic="quantum entanglement",
        word_count=250
    )
    
    print("\n🎉 Demonstration complete!")

if __name__ == "__main__":
    asyncio.run(main())