"""
Example 2: Creating Custom Agents
==================================

This example demonstrates how to create specialized agents for specific tasks.
Shows different return types and custom processing logic.

Prerequisites:
- Set OPENROUTER_API_KEY environment variable
- Install arshai package
"""

import os
import json
import asyncio
from typing import Dict, Any, List
from arshai.agents.base import BaseAgent
from arshai.core.interfaces.iagent import IAgentInput
from arshai.core.interfaces.illm import ILLMInput, ILLMConfig, ILLM
from arshai.llms.openrouter import OpenRouterClient


class SentimentAnalysisAgent(BaseAgent):
    """
    Custom agent specialized in sentiment analysis.
    
    Capabilities:
    - Analyzes emotional tone of text
    - Provides confidence scores
    - Identifies key emotional indicators
    
    Returns:
        Dict[str, Any]: Structured sentiment analysis
    """
    
    def __init__(self, llm_client: ILLM, **kwargs):
        """Initialize with specialized prompt for sentiment analysis."""
        system_prompt = """You are a sentiment analysis expert. 
        Analyze the emotional tone of messages and provide:
        1. Overall sentiment (positive/negative/neutral)
        2. Confidence score (0-100%)
        3. Key emotional indicators
        
        Always respond in this JSON format:
        {
            "sentiment": "positive/negative/neutral",
            "confidence": 0-100,
            "indicators": ["list", "of", "indicators"],
            "explanation": "brief explanation"
        }"""
        
        super().__init__(llm_client, system_prompt, **kwargs)
        self.analysis_history = []  # Track analyses for summary
    
    async def process(self, input: IAgentInput) -> Dict[str, Any]:
        """
        Analyze sentiment and return structured data.
        
        Returns dictionary instead of string for structured output.
        """
        # Prepare analysis request
        llm_input = ILLMInput(
            system_prompt=self.system_prompt,
            user_message=f"Analyze the sentiment of: {input.message}"
        )
        
        # Get analysis from LLM
        result = await self.llm_client.chat(llm_input)
        response_text = result.get('llm_response', '{}')
        
        # Parse JSON response
        try:
            analysis = json.loads(response_text)
        except json.JSONDecodeError:
            # Fallback if parsing fails
            analysis = {
                "sentiment": "unknown",
                "confidence": 0,
                "indicators": ["parsing_error"],
                "explanation": "Failed to parse LLM response"
            }
        
        # Store in history
        self.analysis_history.append({
            "input": input.message,
            "analysis": analysis
        })
        
        return analysis
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all analyses performed."""
        if not self.analysis_history:
            return {"message": "No analyses performed yet"}
        
        sentiments = [a["analysis"]["sentiment"] for a in self.analysis_history]
        avg_confidence = sum(a["analysis"]["confidence"] for a in self.analysis_history) / len(self.analysis_history)
        
        return {
            "total_analyses": len(self.analysis_history),
            "average_confidence": round(avg_confidence, 2),
            "sentiment_distribution": {
                "positive": sentiments.count("positive"),
                "negative": sentiments.count("negative"),
                "neutral": sentiments.count("neutral")
            }
        }


class TranslationAgent(BaseAgent):
    """
    Agent specialized in language translation.
    
    Capabilities:
    - Translates between multiple languages
    - Preserves tone and context
    - Provides alternative translations
    
    Returns:
        Dict[str, Any]: Translation with metadata
    """
    
    def __init__(self, llm_client: ILLM, target_language: str = "Spanish", **kwargs):
        """Initialize with translation capabilities."""
        self.target_language = target_language
        
        system_prompt = f"""You are a professional translator.
        Translate text to {target_language} while:
        1. Preserving the original meaning and tone
        2. Providing cultural context when needed
        3. Offering alternative translations when applicable
        
        Respond in JSON format:
        {{
            "translation": "main translation",
            "alternatives": ["alt1", "alt2"],
            "notes": "any cultural or context notes"
        }}"""
        
        super().__init__(llm_client, system_prompt, **kwargs)
    
    async def process(self, input: IAgentInput) -> Dict[str, Any]:
        """Translate text and return structured result."""
        # Check for language override in metadata
        target_lang = input.metadata.get("target_language", self.target_language) if input.metadata else self.target_language
        
        llm_input = ILLMInput(
            system_prompt=self.system_prompt.replace(self.target_language, target_lang),
            user_message=f"Translate to {target_lang}: {input.message}"
        )
        
        result = await self.llm_client.chat(llm_input)
        response_text = result.get('llm_response', '{}')
        
        try:
            translation_data = json.loads(response_text)
            translation_data["source_text"] = input.message
            translation_data["target_language"] = target_lang
        except json.JSONDecodeError:
            # Fallback to simple string if not JSON
            translation_data = {
                "translation": response_text,
                "source_text": input.message,
                "target_language": target_lang,
                "alternatives": [],
                "notes": ""
            }
        
        return translation_data


class CodeReviewAgent(BaseAgent):
    """
    Agent specialized in code review and analysis.
    
    Capabilities:
    - Reviews code for best practices
    - Identifies potential issues
    - Suggests improvements
    
    Returns:
        Dict[str, Any]: Code review with findings and suggestions
    """
    
    def __init__(self, llm_client: ILLM, language: str = "Python", **kwargs):
        """Initialize with code review expertise."""
        self.language = language
        
        system_prompt = f"""You are an expert {language} code reviewer.
        Review code for:
        1. Best practices and conventions
        2. Potential bugs or issues
        3. Performance considerations
        4. Security concerns
        5. Readability and maintainability
        
        Provide constructive feedback in JSON format:
        {{
            "overall_quality": "excellent/good/fair/needs_improvement",
            "issues": [
                {{"type": "bug/style/performance/security", "line": 0, "description": "...", "severity": "high/medium/low"}}
            ],
            "suggestions": ["list of improvement suggestions"],
            "positive_aspects": ["what was done well"]
        }}"""
        
        super().__init__(llm_client, system_prompt, **kwargs)
    
    async def process(self, input: IAgentInput) -> Dict[str, Any]:
        """Review code and return structured feedback."""
        llm_input = ILLMInput(
            system_prompt=self.system_prompt,
            user_message=f"Review this {self.language} code:\n\n{input.message}"
        )
        
        result = await self.llm_client.chat(llm_input)
        response_text = result.get('llm_response', '{}')
        
        try:
            review = json.loads(response_text)
            review["code_length"] = len(input.message.split('\n'))
            review["language"] = self.language
        except json.JSONDecodeError:
            review = {
                "overall_quality": "unable_to_review",
                "issues": [],
                "suggestions": ["Failed to parse review"],
                "positive_aspects": [],
                "code_length": len(input.message.split('\n')),
                "language": self.language
            }
        
        return review


async def main():
    """Demonstrate custom agent creation and usage."""
    
    # Check for API key
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("⚠️  Please set OPENROUTER_API_KEY environment variable")
        return
    
    # Initialize LLM client
    config = ILLMConfig(model="openai/gpt-4o-mini", temperature=0.3)
    llm_client = OpenRouterClient(config)
    
    # Example 1: Sentiment Analysis Agent
    print("=" * 60)
    print("SENTIMENT ANALYSIS AGENT")
    print("=" * 60)
    
    sentiment_agent = SentimentAnalysisAgent(llm_client)
    
    test_texts = [
        "I absolutely love this new feature! It's amazing and works perfectly!",
        "This is terrible. Nothing works as expected and support is unresponsive.",
        "The product is okay. It does what it's supposed to do, nothing special."
    ]
    
    for text in test_texts:
        print(f"\n📝 Text: {text}")
        analysis = await sentiment_agent.process(IAgentInput(message=text))
        print(f"📊 Analysis:")
        print(f"   Sentiment: {analysis['sentiment']}")
        print(f"   Confidence: {analysis['confidence']}%")
        print(f"   Indicators: {', '.join(analysis.get('indicators', []))}")
        print(f"   Explanation: {analysis.get('explanation', 'N/A')}")
    
    print(f"\n📈 Summary: {sentiment_agent.get_summary()}")
    
    # Example 2: Translation Agent
    print("\n" + "=" * 60)
    print("TRANSLATION AGENT")
    print("=" * 60)
    
    translator = TranslationAgent(llm_client, target_language="French")
    
    texts_to_translate = [
        "Hello, how are you today?",
        "The weather is beautiful this morning."
    ]
    
    for text in texts_to_translate:
        print(f"\n🌐 Original: {text}")
        translation = await translator.process(IAgentInput(message=text))
        print(f"🇫🇷 Translation: {translation['translation']}")
        if translation.get('alternatives'):
            print(f"   Alternatives: {', '.join(translation['alternatives'])}")
        if translation.get('notes'):
            print(f"   Notes: {translation['notes']}")
    
    # Translate to different language using metadata
    print("\n🔄 Using metadata to override target language...")
    result = await translator.process(IAgentInput(
        message="Good morning!",
        metadata={"target_language": "Japanese"}
    ))
    print(f"🇯🇵 Japanese: {result['translation']}")
    
    # Example 3: Code Review Agent
    print("\n" + "=" * 60)
    print("CODE REVIEW AGENT")
    print("=" * 60)
    
    code_reviewer = CodeReviewAgent(llm_client, language="Python")
    
    sample_code = """
def calculate_average(numbers):
    sum = 0
    for i in range(len(numbers)):
        sum = sum + numbers[i]
    average = sum / len(numbers)
    return average
"""
    
    print(f"📝 Code to review:\n{sample_code}")
    
    review = await code_reviewer.process(IAgentInput(message=sample_code))
    
    print("\n🔍 Code Review Results:")
    print(f"   Overall Quality: {review['overall_quality']}")
    print(f"   Issues Found: {len(review.get('issues', []))}")
    for issue in review.get('issues', []):
        print(f"      - [{issue['severity']}] {issue['type']}: {issue['description']}")
    print(f"   Suggestions:")
    for suggestion in review.get('suggestions', []):
        print(f"      - {suggestion}")
    print(f"   Positive Aspects:")
    for positive in review.get('positive_aspects', []):
        print(f"      ✓ {positive}")
    
    print("\n✅ Custom agents demonstration completed!")


if __name__ == "__main__":
    asyncio.run(main())