"""
Claude Conversation Handler - SPARC Implementation

SPECIFICATION:
- Input: User text + conversation context
- Output: Intelligent response with follow-up questions
- Constraints: < 900ms response time, < 3 sentences
- Intelligence: Ask clarifying questions, connect ideas

PSEUDOCODE:
1. Get last 5 exchanges from context
2. Detect intent and select prompt strategy
3. Build advanced prompt with few-shot examples
4. Optionally retrieve similar past conversations (RAG)
5. Call Claude API
6. Extract response and reasoning
7. Return response

ARCHITECTURE:
- Single responsibility: Only handles Claude interaction
- Dependency injection: Pass in state manager
- Error boundaries: Graceful fallback responses
- Week 3: Advanced prompt engineering integration

REFINEMENT:
- Cache frequently used prompts
- Batch API calls when possible
- Use streaming for lower latency perception
- Adaptive prompt strategy selection

CODE:
"""
from typing import List, Dict, Optional, Tuple
import anthropic
from app.config import settings
from app.advanced_prompts import prompt_engine, PromptStrategy
import re
import logging

logger = logging.getLogger(__name__)


class ConversationHandler:
    def __init__(self):
        self.client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._prompt_engine = prompt_engine
        self._vector_store = None  # Lazy loaded
        self._use_advanced_prompts = settings.use_chain_of_thought or settings.use_few_shot

    @property
    def system_prompt(self) -> str:
        """Property for test compatibility - returns basic system prompt."""
        return self._create_basic_system_prompt()

    def _create_system_prompt(self) -> str:
        """Alias for _create_basic_system_prompt for test compatibility."""
        return self._create_basic_system_prompt()

    async def _get_vector_store(self):
        """Lazy load vector store for RAG."""
        if self._vector_store is None and settings.enable_vector_search:
            try:
                from app.vector_store import vector_store
                await vector_store.initialize()
                self._vector_store = vector_store
            except Exception as e:
                logger.warning(f"Vector store not available: {e}")
        return self._vector_store

    def _get_system_prompt(self, strategy: PromptStrategy) -> str:
        """
        Get system prompt based on active strategy.

        CONCEPT: Dynamic prompt selection
        WHY: Different situations benefit from different prompting approaches
        """
        if self._use_advanced_prompts:
            return self._prompt_engine.build_system_prompt(strategy)
        return self._create_basic_system_prompt()

    def _create_basic_system_prompt(self) -> str:
        """
        CONCEPT: Carefully crafted system prompt
        WHY: The prompt IS the intelligence - no complex logic needed
        PATTERN: Behavioral specification through examples
        """
        return """You are a personal learning companion helping capture and develop ideas.

Your role:
- Ask ONE clarifying question when responses are vague or under 10 words
- Connect new ideas to previously mentioned topics in the conversation
- When user lists items, ask "What else?" once, then summarize
- Keep responses under 3 sentences
- Never lecture or provide long explanations unless explicitly asked

Response style:
- Be curious and encouraging
- Use natural, conversational language
- Mirror the user's energy level
- Focus on helping them articulate their thoughts

Special behaviors:
- If user says something profound, acknowledge it briefly
- If they're stuck, offer a gentle prompt like "What made you think of that?"
- If they say goodbye or want to end, confirm and summarize key points"""

    def _format_context(self, exchanges: List[Dict]) -> str:
        """
        PATTERN: Context window formatting
        WHY: Claude needs structured context for coherence
        """
        if not exchanges:
            return "This is the start of our conversation."
        
        context_lines = []
        for exchange in exchanges:
            context_lines.append(f"User: {exchange.get('user', '')}")
            context_lines.append(f"You: {exchange.get('agent', '')}")
        
        return "Previous conversation:\n" + "\n".join(context_lines)
    
    def _should_add_followup(self, user_text: str, response: str) -> bool:
        """
        CONCEPT: Intelligent fallback detection
        WHY: Ensure conversation continues even with short responses
        """
        # Check if response already contains a question
        has_question = "?" in response
        
        # Check if user input is very short
        is_short_input = len(user_text.split()) <= 3
        
        # Check if response is conversational ender
        enders = ["goodbye", "bye", "see you", "thank you", "thanks"]
        is_ending = any(end in response.lower() for end in enders)
        
        return is_short_input and not has_question and not is_ending
    
    async def generate_response(
        self,
        user_text: str,
        context: List[Dict],
        session_metadata: Optional[Dict] = None
    ) -> str:
        """
        PATTERN: Main conversation logic with error handling
        WHY: Centralized intelligence with graceful degradation

        Week 3 Enhancement: Now uses advanced prompt engineering
        with chain-of-thought, few-shot, and RAG capabilities.
        """
        try:
            # Detect intent for prompt strategy selection
            intent = self.detect_intent(user_text)

            # Get similar past conversations for RAG (if enabled)
            similar_conversations = []
            if settings.enable_vector_search:
                vs = await self._get_vector_store()
                if vs:
                    try:
                        similar_conversations = await vs.semantic_search(
                            query=user_text,
                            limit=3,
                            similarity_threshold=settings.semantic_search_threshold
                        )
                    except Exception as e:
                        logger.warning(f"RAG search failed: {e}")

            # Select prompt strategy based on context
            strategy = self._prompt_engine.select_strategy_for_context(
                user_text=user_text,
                intent=intent,
                context_length=len(context),
                has_similar_conversations=len(similar_conversations) > 0
            )

            # Get system prompt for strategy
            system_prompt = self._get_system_prompt(strategy)

            # Build user message with advanced prompting
            if self._use_advanced_prompts:
                user_message = self._prompt_engine.build_user_message(
                    user_text=user_text,
                    context=context,
                    intent=intent,
                    similar_conversations=similar_conversations,
                    strategy=strategy
                )
            else:
                # Fallback to basic prompt building
                context_str = self._format_context(context)
                user_message = f"{context_str}\n\nUser just said: {user_text}\n\nRespond naturally and help them capture their learning effectively."

            # Call Claude API
            message = await self.client.messages.create(
                model=settings.claude_model,
                max_tokens=settings.claude_max_tokens,
                temperature=settings.claude_temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )

            raw_response = message.content[0].text.strip()

            # Extract response (removing thinking tags if present)
            if self._use_advanced_prompts:
                response, reasoning = self._prompt_engine.extract_response(raw_response)
                if reasoning:
                    logger.debug(f"Chain-of-thought reasoning: {reasoning[:100]}...")
            else:
                response = raw_response

            # Add follow-up if needed
            if self._should_add_followup(user_text, response):
                followups = [
                    "Tell me more about that.",
                    "What made you think of that?",
                    "Can you elaborate?",
                    "What's the connection there?",
                    "How does that relate to what you're learning?"
                ]
                import random
                response += f" {random.choice(followups)}"

            return response

        except anthropic.RateLimitError:
            return "I need a moment to catch up. Could you repeat that?"
        except anthropic.APIError as e:
            logger.error(f"Claude API error: {e}")
            return "I'm having trouble connecting right now. Let's try again - what were you saying?"
        except Exception as e:
            logger.error(f"Unexpected error in conversation handler: {e}")
            return "Something went wrong on my end. Could you say that again?"
    
    def detect_intent(self, text: str) -> str:
        """
        CONCEPT: Simple intent detection without ML
        WHY: Fast, deterministic, good enough for our needs
        """
        text_lower = text.lower().strip()
        
        # Ending intents
        if any(word in text_lower for word in ["goodbye", "bye", "see you later", "talk to you later"]):
            return "end_conversation"
        
        # Question intents
        if text_lower.startswith(("what", "why", "how", "when", "where", "who")):
            return "question"
        
        # List intents  
        if any(word in text_lower for word in ["first", "second", "next", "then", "also"]):
            return "listing"
        
        # Reflection intents
        if any(word in text_lower for word in ["realize", "think", "feel", "believe", "understand"]):
            return "reflection"
        
        return "statement"
    
    def create_summary(self, exchanges: List[Dict]) -> str:
        """
        PATTERN: Context-aware summarization
        WHY: Help users see patterns in their learning
        """
        if not exchanges:
            return "We didn't get to explore any topics today."
        
        # Extract key topics (simple noun phrase extraction)
        all_text = " ".join([e.get("user", "") for e in exchanges])
        
        # Simple topic extraction (could be enhanced with NLP)
        words = all_text.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 4:  # Focus on substantial words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        top_topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:3]
        
        if top_topics:
            topics_str = ", ".join([topic[0] for topic in top_topics])
            return f"We explored: {topics_str}. Great conversation!"
        else:
            return "Thanks for sharing your thoughts today!"

# Global conversation handler instance
conversation_handler = ConversationHandler()