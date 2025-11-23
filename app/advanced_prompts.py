"""
Advanced Prompt Engineering Module
PATTERN: Chain-of-Thought and Few-Shot Learning Prompts
WHY: Improve response quality through structured reasoning

SPECIFICATION:
- Chain-of-thought prompts for complex reasoning
- Few-shot examples for consistent behavior
- Context-aware prompt selection
- Dynamic prompt assembly based on intent

ARCHITECTURE:
[User Input] -> [Intent Detection] -> [Prompt Selector]
                                            |
                                    [Template + Examples]
                                            |
                                    [Claude API] -> [Response]
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import random


class PromptStrategy(Enum):
    """Available prompt engineering strategies."""
    BASIC = "basic"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    FEW_SHOT = "few_shot"
    RETRIEVAL_AUGMENTED = "retrieval_augmented"
    SOCRATIC = "socratic"


@dataclass
class FewShotExample:
    """A single few-shot learning example."""
    user_input: str
    reasoning: Optional[str]  # For chain-of-thought
    assistant_response: str
    intent: str


# Few-shot examples for different conversation intents
FEW_SHOT_EXAMPLES: Dict[str, List[FewShotExample]] = {
    "reflection": [
        FewShotExample(
            user_input="I think I finally understand why functional programming matters",
            reasoning="User is having an insight moment. I should acknowledge the breakthrough and gently probe for deeper understanding.",
            assistant_response="That's a significant realization! What clicked for you? Understanding the 'why' often helps solidify new concepts.",
            intent="reflection"
        ),
        FewShotExample(
            user_input="I realized I've been approaching this problem wrong the whole time",
            reasoning="User discovered a misconception. This is valuable learning. I should validate this and help them articulate the shift.",
            assistant_response="Those moments of clarity are valuable. What was your old approach, and what changed your perspective?",
            intent="reflection"
        ),
    ],
    "question": [
        FewShotExample(
            user_input="How do neural networks actually learn?",
            reasoning="Technical question seeking explanation. Keep it concise but accurate, and check their current understanding level.",
            assistant_response="They adjust connection weights based on errors - like tuning knobs until the output matches expectations. What's your current mental model of this?",
            intent="question"
        ),
        FewShotExample(
            user_input="Why is testing so important?",
            reasoning="Conceptual question about software practices. Give a concrete reason and relate it to their experience.",
            assistant_response="Tests are your safety net - they catch mistakes before users do. Have you experienced a bug that tests would have caught?",
            intent="question"
        ),
    ],
    "listing": [
        FewShotExample(
            user_input="First, I want to learn Python. Then JavaScript. And maybe Rust.",
            reasoning="User is listing goals. I should acknowledge and help them think about prioritization or connections.",
            assistant_response="Solid progression! Python and JavaScript will cover most use cases. What draws you to Rust specifically?",
            intent="listing"
        ),
    ],
    "statement": [
        FewShotExample(
            user_input="Machine learning is fascinating",
            reasoning="General statement expressing interest. Probe to understand what specifically interests them.",
            assistant_response="It really is! What aspect fascinates you most - the math, the applications, or something else?",
            intent="statement"
        ),
    ],
    "end_conversation": [
        FewShotExample(
            user_input="Thanks, I think I'm good for now",
            reasoning="User wants to wrap up. Acknowledge gracefully and offer a brief summary if appropriate.",
            assistant_response="Great session! Feel free to come back when you want to explore more ideas.",
            intent="end_conversation"
        ),
    ]
}


class AdvancedPromptEngine:
    """
    Advanced prompt engineering for improved conversation quality.

    CONCEPT: Better prompts = better responses
    WHY: Structured reasoning and examples improve consistency
    PATTERN: Strategy pattern for prompt selection
    """

    def __init__(self):
        self.strategy = PromptStrategy.FEW_SHOT
        self._examples = FEW_SHOT_EXAMPLES

    def set_strategy(self, strategy: PromptStrategy):
        """Change the active prompt strategy."""
        self.strategy = strategy

    def build_system_prompt(
        self,
        strategy: Optional[PromptStrategy] = None,
        include_reasoning: bool = True
    ) -> str:
        """
        Build the system prompt based on selected strategy.

        PATTERN: Template method for prompt construction
        WHY: Different strategies need different base prompts
        """
        active_strategy = strategy or self.strategy

        base_prompt = """You are a personal learning companion helping capture and develop ideas.

Core Behaviors:
- Ask ONE clarifying question when responses are vague or under 10 words
- Connect new ideas to previously mentioned topics
- Keep responses under 3 sentences unless asked for more
- Be curious and encouraging, mirror the user's energy

Special Abilities:
- Recognize learning moments and acknowledge breakthroughs
- Help users articulate vague thoughts into clear concepts
- Notice patterns and connections across conversations"""

        if active_strategy == PromptStrategy.CHAIN_OF_THOUGHT:
            return base_prompt + """

IMPORTANT: Before responding, briefly think through:
1. What is the user trying to express or learn?
2. What would be most helpful right now?
3. How can I help them go deeper without overwhelming?

Show your reasoning in <thinking> tags, then give your response."""

        elif active_strategy == PromptStrategy.SOCRATIC:
            return base_prompt + """

SOCRATIC METHOD: Guide through questions rather than statements.
- When user makes a claim, ask "What makes you think that?"
- When they're stuck, ask "What do you already know about this?"
- Help them discover insights rather than telling them answers
- Use "How might..." and "What if..." questions generously"""

        elif active_strategy == PromptStrategy.RETRIEVAL_AUGMENTED:
            return base_prompt + """

CONTEXT AWARENESS: You have access to the user's previous conversations.
- Reference relevant past discussions when appropriate
- Notice patterns in their learning journey
- Connect current topics to previous insights
- Help them see their progress over time"""

        return base_prompt

    def format_few_shot_context(
        self,
        intent: str,
        num_examples: int = 2,
        include_reasoning: bool = False
    ) -> str:
        """
        Format few-shot examples for the given intent.

        CONCEPT: Show examples of desired behavior
        WHY: Claude learns from examples in context
        """
        examples = self._examples.get(intent, self._examples.get("statement", []))

        if not examples:
            return ""

        # Select examples (random if more than needed)
        selected = random.sample(examples, min(num_examples, len(examples)))

        formatted_examples = []
        for ex in selected:
            example_text = f"User: {ex.user_input}\n"
            if include_reasoning and ex.reasoning:
                example_text += f"<thinking>{ex.reasoning}</thinking>\n"
            example_text += f"Assistant: {ex.assistant_response}"
            formatted_examples.append(example_text)

        return "Here are examples of good responses:\n\n" + "\n\n---\n\n".join(formatted_examples)

    def build_user_message(
        self,
        user_text: str,
        context: List[Dict],
        intent: str,
        similar_conversations: Optional[List[Dict]] = None,
        strategy: Optional[PromptStrategy] = None
    ) -> str:
        """
        Build the complete user message with context and examples.

        ARCHITECTURE:
        [Context] + [Few-shot Examples] + [Similar Conversations] + [Current Input]
        """
        active_strategy = strategy or self.strategy
        parts = []

        # Add conversation history
        if context:
            history = self._format_conversation_history(context)
            parts.append(f"Conversation so far:\n{history}")

        # Add few-shot examples for consistency
        if active_strategy in [PromptStrategy.FEW_SHOT, PromptStrategy.CHAIN_OF_THOUGHT]:
            examples = self.format_few_shot_context(
                intent,
                include_reasoning=(active_strategy == PromptStrategy.CHAIN_OF_THOUGHT)
            )
            if examples:
                parts.append(examples)

        # Add semantically similar past conversations (RAG)
        if active_strategy == PromptStrategy.RETRIEVAL_AUGMENTED and similar_conversations:
            similar_context = self._format_similar_conversations(similar_conversations)
            if similar_context:
                parts.append(f"Relevant past discussions:\n{similar_context}")

        # Add the current input
        parts.append(f"User just said: {user_text}")

        # Add response instruction
        if active_strategy == PromptStrategy.CHAIN_OF_THOUGHT:
            parts.append("Think through your response in <thinking> tags first, then respond.")
        else:
            parts.append("Respond naturally and help them capture their learning effectively.")

        return "\n\n".join(parts)

    def _format_conversation_history(self, context: List[Dict]) -> str:
        """Format conversation history for prompt."""
        lines = []
        for exchange in context[-5:]:  # Last 5 exchanges
            lines.append(f"User: {exchange.get('user', '')}")
            lines.append(f"You: {exchange.get('agent', '')}")
        return "\n".join(lines)

    def _format_similar_conversations(
        self,
        similar: List[Dict],
        max_examples: int = 3
    ) -> str:
        """Format similar past conversations for RAG context."""
        if not similar:
            return ""

        formatted = []
        for conv in similar[:max_examples]:
            formatted.append(
                f"- User said: \"{conv.get('user_text', '')[:100]}...\"\n"
                f"  You replied: \"{conv.get('agent_text', '')[:100]}...\""
            )

        return "\n".join(formatted)

    def extract_response(self, raw_response: str) -> Tuple[str, Optional[str]]:
        """
        Extract the actual response and reasoning from Claude's output.

        Returns: (response_text, reasoning_text or None)
        """
        import re

        # Check for thinking tags
        thinking_pattern = r'<thinking>(.*?)</thinking>'
        thinking_match = re.search(thinking_pattern, raw_response, re.DOTALL)

        reasoning = None
        response = raw_response

        if thinking_match:
            reasoning = thinking_match.group(1).strip()
            response = re.sub(thinking_pattern, '', raw_response, flags=re.DOTALL).strip()

        return response, reasoning

    def select_strategy_for_context(
        self,
        user_text: str,
        intent: str,
        context_length: int,
        has_similar_conversations: bool
    ) -> PromptStrategy:
        """
        Dynamically select the best prompt strategy.

        CONCEPT: Adaptive prompt engineering
        WHY: Different situations benefit from different approaches
        """
        # Complex questions benefit from chain-of-thought
        complex_indicators = ['why', 'how does', 'explain', 'what if', 'compare']
        if any(indicator in user_text.lower() for indicator in complex_indicators):
            return PromptStrategy.CHAIN_OF_THOUGHT

        # Reflection moments benefit from Socratic method
        if intent == 'reflection':
            return PromptStrategy.SOCRATIC

        # If we have relevant past conversations, use RAG
        if has_similar_conversations:
            return PromptStrategy.RETRIEVAL_AUGMENTED

        # Default to few-shot for consistency
        return PromptStrategy.FEW_SHOT

    def add_custom_example(
        self,
        intent: str,
        user_input: str,
        assistant_response: str,
        reasoning: Optional[str] = None
    ):
        """
        Add a custom few-shot example.

        CONCEPT: Learn from successful interactions
        WHY: Improve over time based on actual conversations
        """
        example = FewShotExample(
            user_input=user_input,
            reasoning=reasoning,
            assistant_response=assistant_response,
            intent=intent
        )

        if intent not in self._examples:
            self._examples[intent] = []

        self._examples[intent].append(example)

        # Keep only the most recent examples
        if len(self._examples[intent]) > 10:
            self._examples[intent] = self._examples[intent][-10:]


# Global advanced prompt engine instance
prompt_engine = AdvancedPromptEngine()
