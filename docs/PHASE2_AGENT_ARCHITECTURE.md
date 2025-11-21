# Phase 2: Multi-Agent Architecture Design

**Version:** 1.0
**Date:** 2025-11-21
**Status:** Design Phase
**Target:** Production-ready LangGraph multi-agent system

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture Overview](#system-architecture-overview)
3. [Agent Specifications](#agent-specifications)
4. [Communication Protocol](#communication-protocol)
5. [State Management](#state-management)
6. [Orchestration Strategy](#orchestration-strategy)
7. [Integration with v1.0](#integration-with-v10)
8. [Implementation Guide](#implementation-guide)
9. [Testing Strategy](#testing-strategy)
10. [Performance Requirements](#performance-requirements)
11. [Migration Path](#migration-path)

---

## Executive Summary

### Objectives

Transform the learning_voice_agent from a single-agent system into a **production-ready multi-agent orchestration platform** using LangGraph principles, enabling:

- **Specialized expertise** through domain-specific agents
- **Parallel processing** for faster response times
- **Complex reasoning** via multi-agent collaboration
- **Scalable architecture** for future agent additions
- **Backward compatibility** with existing v1.0 system

### Key Design Principles

1. **Agent Autonomy**: Each agent has clear responsibilities and decision-making capability
2. **Loose Coupling**: Agents communicate via standardized message protocol
3. **State Isolation**: Each agent maintains its own state while sharing common context
4. **Graceful Degradation**: System operates even if individual agents fail
5. **Observable**: All agent interactions are logged and traceable

### Success Criteria

- ✅ 5 specialized agents operational (Conversation, Analysis, Research, Vision, Synthesis)
- ✅ Sub-1.5s response time for 95% of requests
- ✅ Zero data loss during migration from v1.0
- ✅ Agent coordination working with < 100ms overhead
- ✅ Backward compatible with all v1.0 APIs

---

## System Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLIENT APPLICATIONS                          │
├────────────┬────────────┬────────────┬─────────────────────────┤
│  Web UI    │  Mobile    │   Phone    │    API Clients          │
│  (Vue 3)   │   (PWA)    │  (Twilio)  │    (REST/WS)            │
└─────┬──────┴──────┬─────┴──────┬─────┴──────┬──────────────────┘
      │             │            │            │
      └─────────────┴────────────┴────────────┘
                     │
      ┌──────────────────────────────────────────────────────────┐
      │           FASTAPI ORCHESTRATION LAYER                    │
      │                                                          │
      │  ┌────────────────────────────────────────────────┐     │
      │  │      LangGraph Agent Orchestrator              │     │
      │  │  ┌──────────────────────────────────────────┐  │     │
      │  │  │  State Graph (Agent Workflow)            │  │     │
      │  │  │                                          │  │     │
      │  │  │   START → Route → Agent(s) → Merge      │  │     │
      │  │  │             │                   │        │  │     │
      │  │  │             └─→ [Parallel] ─────┘        │  │     │
      │  │  └──────────────────────────────────────────┘  │     │
      │  └────────────────────────────────────────────────┘     │
      └──────────────────────────┬───────────────────────────────┘
                                 │
      ┌──────────────────────────────────────────────────────────┐
      │                 AGENT LAYER (5 Agents)                   │
      ├────────────┬────────────┬────────────┬────────────┬──────┤
      │Conversation│  Analysis  │  Research  │   Vision   │Synth │
      │   Agent    │   Agent    │   Agent    │   Agent    │Agent │
      │            │            │            │            │      │
      │ Claude 3.5 │   Claude   │  Tools +   │  GPT-4V    │Claude│
      │  Sonnet    │   Haiku    │  Claude    │            │Sonnet│
      │            │            │            │            │      │
      │ • Dialog   │ • Extract  │ • Web      │ • Images   │•Synth│
      │ • Context  │ • Concepts │ • Docs     │ • OCR      │•Conn │
      │ • Q&A      │ • Entities │ • Search   │ • Diagrams │•Summ │
      └─────┬──────┴──────┬─────┴──────┬─────┴──────┬─────┴──┬───┘
            │             │            │            │        │
      ┌─────────────────────────────────────────────────────────┐
      │                 SHARED STATE MANAGER                    │
      │                                                         │
      │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
      │  │Conversation  │  │   Agent      │  │   Result     │  │
      │  │   Context    │  │   States     │  │    Cache     │  │
      │  │   (Redis)    │  │  (Memory)    │  │   (Redis)    │  │
      │  └──────────────┘  └──────────────┘  └──────────────┘  │
      └─────────────────────────────────────────────────────────┘
                                 │
      ┌─────────────────────────────────────────────────────────┐
      │                   DATA LAYER                            │
      ├─────────────┬─────────────┬─────────────┬───────────────┤
      │   SQLite    │   Redis     │  ChromaDB   │    Neo4j      │
      │  (Archive)  │  (Cache)    │  (Future)   │   (Future)    │
      └─────────────┴─────────────┴─────────────┴───────────────┘
```

### Agent Communication Flow

```
User Input
    │
    ▼
┌───────────────────────────────────────────────────────────┐
│ 1. INPUT PROCESSING & ROUTING                             │
│                                                           │
│  Input → Classifier → Route Decision                     │
│           │                                              │
│           ├─ Simple query? → ConversationAgent only     │
│           ├─ Complex query? → Multiple agents           │
│           ├─ Image/doc?    → VisionAgent + others       │
│           └─ Deep insight?  → All agents                │
└───────────────────┬───────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────────┐
│ 2. PARALLEL AGENT EXECUTION                               │
│                                                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │Conversation │  │  Analysis   │  │  Research   │       │
│  │   Agent     │  │   Agent     │  │   Agent     │       │
│  └─────┬───────┘  └─────┬───────┘  └─────┬───────┘       │
│        │                │                │               │
│        │   Shared State Manager (Context Exchange)      │
│        │                │                │               │
│  ┌─────▼───────┐  ┌─────▼───────┐  ┌─────▼───────┐       │
│  │  Response   │  │  Concepts   │  │  Facts      │       │
│  │   Draft     │  │  Extracted  │  │  Retrieved  │       │
│  └─────────────┘  └─────────────┘  └─────────────┘       │
└───────────────────┬───────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────────┐
│ 3. SYNTHESIS & RESPONSE GENERATION                        │
│                                                           │
│  SynthesisAgent:                                         │
│   • Combines all agent outputs                           │
│   • Resolves conflicts                                   │
│   • Generates coherent response                          │
│   • Adds follow-up questions                             │
└───────────────────┬───────────────────────────────────────┘
                    │
                    ▼
              Final Response
```

### Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                  REQUEST LIFECYCLE                          │
└─────────────────────────────────────────────────────────────┘

  Time →
  │
  0ms   ├─ HTTP Request arrives at FastAPI
  │     │
  5ms   ├─ Orchestrator initializes shared state
  │     │  • Creates conversation context
  │     │  • Loads recent history from Redis
  │     │  • Initializes agent states
  │     │
  10ms  ├─ Input Classification
  │     │  • Detect intent (query, statement, request)
  │     │  • Identify required agents
  │     │  • Set execution mode (sequential/parallel)
  │     │
  15ms  ├─ Agent Dispatch (Parallel)
  │     │
  │     ├─┬─ ConversationAgent starts
  │     │ │   • Formats conversation context
  │     │ │   • Calls Claude 3.5 Sonnet API
  │     │ │
  │     │ ├─ AnalysisAgent starts
  │     │ │   • Extracts entities & concepts
  │     │ │   • Calls Claude Haiku for NER
  │     │ │
  │     │ └─ ResearchAgent starts (if needed)
  │     │     • Determines search queries
  │     │     • Executes tool calls
  │     │
  900ms ├─ Agents complete (parallel)
  │     │  • ConversationAgent: Response draft (800ms)
  │     │  • AnalysisAgent: Concepts extracted (300ms)
  │     │  • ResearchAgent: Facts retrieved (600ms)
  │     │
  920ms ├─ SynthesisAgent merges results
  │     │  • Combines response + concepts + facts
  │     │  • Adds follow-up question
  │     │  • Formats final output
  │     │
  950ms ├─ State Update (Background)
  │     │  • Update Redis context
  │     │  • Save to SQLite
  │     │  • Update metrics
  │     │
  960ms ├─ Response sent to client
  │
  │     Total: ~960ms (well under 1.5s target)
```

---

## Agent Specifications

### 1. ConversationAgent

**Purpose:** Primary user interaction and natural dialog management.

**Responsibilities:**
- Maintain natural conversational flow
- Ask clarifying questions when needed
- Connect to previous conversation context
- Mirror user's communication style
- Generate engaging follow-up questions

**Model:** Claude 3.5 Sonnet (upgraded from Haiku for better reasoning)

**Interface:**
```python
class ConversationAgent:
    """
    Primary conversational agent for natural dialog.

    Capabilities:
    - Context-aware responses
    - Intelligent follow-up questions
    - Tone matching
    - Conversation steering
    """

    async def respond(
        self,
        user_input: str,
        context: ConversationContext,
        agent_insights: Optional[Dict[str, Any]] = None
    ) -> ConversationResponse:
        """
        Generate conversational response.

        Args:
            user_input: User's message
            context: Conversation history and metadata
            agent_insights: Optional insights from other agents
                {
                    'concepts': [...],      # From AnalysisAgent
                    'facts': [...],         # From ResearchAgent
                    'vision_analysis': {...} # From VisionAgent
                }

        Returns:
            ConversationResponse with text, intent, follow_ups
        """
        pass
```

**System Prompt:**
```python
CONVERSATION_AGENT_PROMPT = """You are the Conversation Agent in a multi-agent learning system.

Your role:
- Engage users in natural, flowing conversation about their learning
- Ask ONE clarifying question when responses are vague
- Connect new topics to previous conversation themes
- Keep responses conversational and under 3 sentences
- Use insights from other agents to enrich your responses

When you receive agent_insights:
- Use 'concepts' to connect ideas across the conversation
- Reference 'facts' naturally when relevant
- Incorporate 'vision_analysis' if discussing images

Response style:
- Curious and encouraging
- Natural language (avoid being robotic)
- Mirror user's energy and formality level
- Focus on helping them articulate thoughts

Special instructions:
- If user shares a profound insight, acknowledge briefly
- If stuck, prompt gently: "What made you think of that?"
- For goodbyes, create brief summary of key topics discussed
"""
```

**Example Usage:**
```python
# Simple query (single agent)
response = await conversation_agent.respond(
    user_input="I'm learning about neural networks",
    context=ConversationContext(
        session_id="abc-123",
        history=[],
        metadata={}
    )
)
# Output: "That's exciting! What aspect of neural networks are you focusing on?"

# Complex query (with other agent insights)
response = await conversation_agent.respond(
    user_input="I'm confused about backpropagation",
    context=context,
    agent_insights={
        'concepts': ['gradient descent', 'chain rule', 'loss function'],
        'facts': [
            "Backpropagation is the algorithm for calculating gradients",
            "It uses the chain rule from calculus"
        ]
    }
)
# Output: "Backpropagation can be tricky! It's essentially using the chain
#          rule to calculate how to adjust weights. Which part is most
#          confusing - the math or the intuition?"
```

**Performance Targets:**
- Response time: < 900ms (P95)
- Context window: 8K tokens
- Failure rate: < 0.5%

---

### 2. AnalysisAgent

**Purpose:** Extract concepts, entities, patterns, and learning themes from conversations.

**Responsibilities:**
- Named entity recognition (topics, people, concepts)
- Concept extraction and categorization
- Pattern detection across conversations
- Topic clustering
- Knowledge graph node identification

**Model:** Claude Haiku (fast, cost-effective for structured extraction)

**Interface:**
```python
class AnalysisAgent:
    """
    Analyzes conversation for concepts, entities, and patterns.

    Capabilities:
    - Entity extraction (topics, people, concepts)
    - Theme detection
    - Pattern recognition
    - Concept relationship mapping
    """

    async def analyze(
        self,
        user_input: str,
        context: ConversationContext
    ) -> AnalysisResult:
        """
        Extract structured insights from conversation.

        Args:
            user_input: Current user message
            context: Conversation history

        Returns:
            AnalysisResult containing:
            - entities: List[Entity]
            - concepts: List[Concept]
            - themes: List[str]
            - relationships: List[Relationship]
            - confidence_scores: Dict[str, float]
        """
        pass
```

**Output Schema:**
```python
@dataclass
class AnalysisResult:
    """Structured analysis output"""
    entities: List[Entity]              # Named entities found
    concepts: List[Concept]             # Abstract concepts
    themes: List[str]                   # Conversation themes
    relationships: List[Relationship]    # Concept connections
    sentiment: str                       # Overall sentiment
    confidence: float                    # Analysis confidence

@dataclass
class Entity:
    text: str                # Entity text
    type: str                # Type: TOPIC, PERSON, ORG, CONCEPT
    confidence: float        # Confidence score
    context: str             # Surrounding context

@dataclass
class Concept:
    name: str                # Concept name
    category: str            # Category (tech, science, etc)
    importance: float        # Importance score (0-1)
    related_to: List[str]    # Related concepts

@dataclass
class Relationship:
    source: str              # Source concept
    target: str              # Target concept
    type: str                # Relationship type
    strength: float          # Relationship strength
```

**System Prompt:**
```python
ANALYSIS_AGENT_PROMPT = """You are the Analysis Agent in a multi-agent system.

Your role:
- Extract entities (topics, concepts, people) from user input
- Identify themes and patterns in the conversation
- Detect relationships between concepts
- Categorize learning topics

Output format (JSON):
{
  "entities": [
    {"text": "neural networks", "type": "TOPIC", "confidence": 0.95},
    {"text": "backpropagation", "type": "CONCEPT", "confidence": 0.9}
  ],
  "concepts": [
    {"name": "deep learning", "category": "AI/ML", "importance": 0.8}
  ],
  "themes": ["machine learning fundamentals", "neural network training"],
  "relationships": [
    {"source": "backpropagation", "target": "gradient descent",
     "type": "uses", "strength": 0.9}
  ],
  "sentiment": "curious",
  "confidence": 0.85
}

Guidelines:
- Focus on extracting learning-relevant entities
- Categorize concepts by domain (tech, science, arts, etc)
- Identify connections between concepts mentioned
- Rate importance based on conversation focus
"""
```

**Example Usage:**
```python
result = await analysis_agent.analyze(
    user_input="I'm studying transformers in NLP and how attention mechanisms work",
    context=context
)

# Output:
# AnalysisResult(
#   entities=[
#     Entity(text="transformers", type="TOPIC", confidence=0.95),
#     Entity(text="NLP", type="TOPIC", confidence=0.98),
#     Entity(text="attention mechanisms", type="CONCEPT", confidence=0.92)
#   ],
#   concepts=[
#     Concept(name="deep learning", category="AI/ML", importance=0.9),
#     Concept(name="sequence modeling", category="AI/ML", importance=0.8)
#   ],
#   themes=["natural language processing", "neural architectures"],
#   relationships=[
#     Relationship(source="transformers", target="attention mechanisms",
#                  type="uses", strength=0.95)
#   ],
#   sentiment="curious",
#   confidence=0.88
# )
```

**Performance Targets:**
- Response time: < 300ms (P95)
- Precision: > 85% for entity extraction
- Recall: > 80% for concept identification

---

### 3. ResearchAgent

**Purpose:** Retrieve external knowledge via tools, web search, and document retrieval.

**Responsibilities:**
- Web search for factual information
- Document retrieval from knowledge bases
- Tool execution (calculator, code runner, etc)
- Fact verification
- Citation management

**Model:** Claude 3.5 Sonnet with tool use

**Interface:**
```python
class ResearchAgent:
    """
    Retrieves external knowledge via tools and search.

    Capabilities:
    - Web search
    - Document retrieval
    - Tool execution
    - Fact checking
    """

    def __init__(self):
        self.tools = [
            WebSearchTool(),
            WikipediaTool(),
            ArxivTool(),
            CalculatorTool(),
            CodeExecutorTool()
        ]

    async def research(
        self,
        query: str,
        context: ConversationContext,
        required_tools: Optional[List[str]] = None
    ) -> ResearchResult:
        """
        Conduct research using available tools.

        Args:
            query: Research query
            context: Conversation context for relevance
            required_tools: Optional list of specific tools to use

        Returns:
            ResearchResult with facts, sources, and confidence
        """
        pass
```

**Available Tools:**
```python
class WebSearchTool:
    """Search the web for current information"""
    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        pass

class WikipediaTool:
    """Retrieve Wikipedia articles"""
    async def get_summary(self, topic: str) -> str:
        pass

class ArxivTool:
    """Search academic papers on arXiv"""
    async def search(self, query: str) -> List[Paper]:
        pass

class CalculatorTool:
    """Perform mathematical calculations"""
    async def calculate(self, expression: str) -> float:
        pass

class CodeExecutorTool:
    """Execute code in sandboxed environment"""
    async def execute(self, code: str, language: str) -> ExecutionResult:
        pass
```

**Output Schema:**
```python
@dataclass
class ResearchResult:
    """Research output with sources"""
    facts: List[Fact]               # Retrieved facts
    sources: List[Source]           # Source citations
    tool_calls: List[ToolCall]      # Tools executed
    confidence: float               # Overall confidence

@dataclass
class Fact:
    statement: str                  # Factual statement
    source_id: str                  # Reference to source
    confidence: float               # Fact confidence
    relevance: float                # Relevance to query

@dataclass
class Source:
    id: str                         # Unique ID
    title: str                      # Source title
    url: Optional[str]              # Source URL
    type: str                       # web, paper, wiki, etc
    credibility: float              # Source credibility score
```

**System Prompt:**
```python
RESEARCH_AGENT_PROMPT = """You are the Research Agent in a multi-agent system.

Your role:
- Use tools to retrieve factual information
- Search the web for current, accurate data
- Find academic papers when needed
- Execute calculations or code when helpful
- Always cite sources

Available tools:
- web_search: Search the internet
- wikipedia: Get Wikipedia summaries
- arxiv: Search academic papers
- calculator: Perform calculations
- code_executor: Run code snippets

Tool selection guidelines:
- Use web_search for current events and general facts
- Use wikipedia for established concepts and definitions
- Use arxiv for academic/research topics
- Use calculator for math problems
- Use code_executor for demonstrating programming concepts

Output format (JSON):
{
  "facts": [
    {
      "statement": "Transformers were introduced in 'Attention is All You Need'",
      "source_id": "arxiv_001",
      "confidence": 0.95,
      "relevance": 0.9
    }
  ],
  "sources": [
    {
      "id": "arxiv_001",
      "title": "Attention is All You Need",
      "url": "https://arxiv.org/abs/1706.03762",
      "type": "paper",
      "credibility": 0.98
    }
  ],
  "tool_calls": ["arxiv_search", "web_search"],
  "confidence": 0.92
}
"""
```

**Example Usage:**
```python
result = await research_agent.research(
    query="What is the transformer architecture?",
    context=context
)

# Output:
# ResearchResult(
#   facts=[
#     Fact(
#       statement="The Transformer architecture was introduced in the 2017
#                  paper 'Attention is All You Need' by Vaswani et al.",
#       source_id="arxiv_001",
#       confidence=0.98,
#       relevance=0.95
#     ),
#     Fact(
#       statement="Transformers use self-attention mechanisms instead of
#                  recurrence or convolution",
#       source_id="wiki_001",
#       confidence=0.95,
#       relevance=0.9
#     )
#   ],
#   sources=[
#     Source(id="arxiv_001", title="Attention is All You Need", ...),
#     Source(id="wiki_001", title="Transformer (machine learning)", ...)
#   ],
#   tool_calls=["arxiv_search", "wikipedia_get"],
#   confidence=0.94
# )
```

**Performance Targets:**
- Response time: < 800ms (P95)
- Fact accuracy: > 90%
- Source credibility: > 85%

---

### 4. VisionAgent

**Purpose:** Process images, documents, diagrams, and visual content.

**Responsibilities:**
- Image understanding and description
- OCR for text extraction
- Diagram and chart analysis
- Document processing
- Visual concept extraction

**Model:** GPT-4V (Vision)

**Interface:**
```python
class VisionAgent:
    """
    Processes visual content (images, documents, diagrams).

    Capabilities:
    - Image description and analysis
    - OCR text extraction
    - Diagram interpretation
    - Document processing
    """

    async def analyze_visual(
        self,
        image_data: bytes,
        image_type: str,
        context: ConversationContext,
        analysis_type: Optional[str] = "general"
    ) -> VisionResult:
        """
        Analyze visual content.

        Args:
            image_data: Raw image bytes
            image_type: Image MIME type
            context: Conversation context
            analysis_type: Type of analysis
                - "general": Overall description
                - "ocr": Text extraction
                - "diagram": Diagram analysis
                - "document": Document processing

        Returns:
            VisionResult with description, extracted text, concepts
        """
        pass
```

**Output Schema:**
```python
@dataclass
class VisionResult:
    """Visual analysis output"""
    description: str                # Natural language description
    extracted_text: Optional[str]   # OCR text if applicable
    concepts: List[str]             # Visual concepts identified
    objects: List[DetectedObject]   # Objects detected
    analysis_type: str              # Type of analysis performed
    confidence: float               # Overall confidence

@dataclass
class DetectedObject:
    label: str                      # Object label
    confidence: float               # Detection confidence
    location: Optional[BoundingBox] # Bounding box (if available)
```

**System Prompt:**
```python
VISION_AGENT_PROMPT = """You are the Vision Agent in a multi-agent system.

Your role:
- Analyze images, diagrams, and documents
- Extract text via OCR when present
- Identify visual concepts and objects
- Describe visual content in context of learning

Analysis guidelines:
- For diagrams: Explain structure, flow, and relationships
- For documents: Extract key text and structure
- For images: Describe relevant learning content
- For charts: Interpret data and trends

Output format (JSON):
{
  "description": "A neural network architecture diagram showing...",
  "extracted_text": "Input Layer -> Hidden Layers -> Output Layer",
  "concepts": ["neural network", "deep learning", "architecture"],
  "objects": [
    {"label": "network diagram", "confidence": 0.92}
  ],
  "analysis_type": "diagram",
  "confidence": 0.88
}

Context awareness:
- Connect visual content to conversation topics
- Highlight learning-relevant aspects
- Provide clear, educational descriptions
"""
```

**Example Usage:**
```python
# Analyze a neural network diagram
result = await vision_agent.analyze_visual(
    image_data=diagram_bytes,
    image_type="image/png",
    context=context,
    analysis_type="diagram"
)

# Output:
# VisionResult(
#   description="A diagram of a feedforward neural network with an input
#                layer (3 nodes), two hidden layers (5 and 4 nodes), and
#                an output layer (2 nodes). Arrows show connections between
#                all nodes across consecutive layers.",
#   extracted_text="Input -> Hidden1 -> Hidden2 -> Output",
#   concepts=["neural network", "feedforward", "multilayer perceptron"],
#   objects=[
#     DetectedObject(label="network diagram", confidence=0.95)
#   ],
#   analysis_type="diagram",
#   confidence=0.91
# )
```

**Performance Targets:**
- Response time: < 1200ms (P95)
- OCR accuracy: > 95%
- Description quality: > 90% user satisfaction

---

### 5. SynthesisAgent

**Purpose:** Generate insights, summaries, and connections across all agent outputs.

**Responsibilities:**
- Combine outputs from all agents into coherent response
- Generate meta-insights about learning patterns
- Create summaries of conversations
- Recommend next learning steps
- Identify knowledge gaps

**Model:** Claude 3.5 Sonnet

**Interface:**
```python
class SynthesisAgent:
    """
    Synthesizes insights from all agents and generates recommendations.

    Capabilities:
    - Multi-agent output integration
    - Insight generation
    - Pattern synthesis
    - Learning recommendations
    """

    async def synthesize(
        self,
        conversation_response: ConversationResponse,
        analysis: AnalysisResult,
        research: Optional[ResearchResult],
        vision: Optional[VisionResult],
        context: ConversationContext
    ) -> SynthesisResult:
        """
        Synthesize insights from all agent outputs.

        Args:
            conversation_response: From ConversationAgent
            analysis: From AnalysisAgent
            research: Optional from ResearchAgent
            vision: Optional from VisionAgent
            context: Conversation context

        Returns:
            SynthesisResult with final response and insights
        """
        pass

    async def generate_insights(
        self,
        context: ConversationContext,
        timeframe: str = "session"
    ) -> List[Insight]:
        """
        Generate meta-insights about learning patterns.

        Args:
            context: Full conversation context
            timeframe: "session", "week", "month"

        Returns:
            List of insights about learning patterns
        """
        pass
```

**Output Schema:**
```python
@dataclass
class SynthesisResult:
    """Final synthesized output"""
    final_response: str             # Coherent final response
    insights: List[Insight]         # Generated insights
    recommendations: List[str]      # Learning recommendations
    knowledge_gaps: List[str]       # Identified gaps
    connections: List[Connection]   # Cross-topic connections
    confidence: float               # Synthesis confidence

@dataclass
class Insight:
    type: str                       # Type: pattern, trend, connection
    description: str                # Insight description
    evidence: List[str]             # Supporting evidence
    importance: float               # Importance score

@dataclass
class Connection:
    concept_a: str                  # First concept
    concept_b: str                  # Second concept
    relationship: str               # Relationship description
    strength: float                 # Connection strength
```

**System Prompt:**
```python
SYNTHESIS_AGENT_PROMPT = """You are the Synthesis Agent in a multi-agent system.

Your role:
- Integrate outputs from all agents into a coherent response
- Generate meta-insights about learning patterns
- Identify connections across topics
- Recommend next learning steps
- Highlight knowledge gaps

Inputs you receive:
- conversation_response: Natural language response from ConversationAgent
- analysis: Extracted concepts and entities
- research: Retrieved facts and sources (if applicable)
- vision: Visual content analysis (if applicable)

Synthesis guidelines:
- Combine agent outputs without redundancy
- Preserve conversational tone from ConversationAgent
- Naturally integrate facts from ResearchAgent
- Reference visual content when applicable
- Add 1-2 insights if patterns emerge

Output format (JSON):
{
  "final_response": "Coherent response text with all insights integrated",
  "insights": [
    {
      "type": "pattern",
      "description": "You've been exploring ML architectures consistently",
      "evidence": ["transformers", "neural networks", "attention mechanisms"],
      "importance": 0.8
    }
  ],
  "recommendations": [
    "Dive deeper into attention mechanisms",
    "Practice implementing a simple transformer"
  ],
  "knowledge_gaps": ["Understanding of positional encoding"],
  "connections": [
    {
      "concept_a": "transformers",
      "concept_b": "attention mechanisms",
      "relationship": "Transformers are built on attention mechanisms",
      "strength": 0.95
    }
  ],
  "confidence": 0.9
}

Quality criteria:
- Response should feel natural, not mechanical
- Insights should be actionable
- Recommendations should be specific
- Connections should be meaningful
"""
```

**Example Usage:**
```python
result = await synthesis_agent.synthesize(
    conversation_response=conv_response,
    analysis=analysis_result,
    research=research_result,
    vision=None,
    context=context
)

# Output:
# SynthesisResult(
#   final_response="Transformers are a powerful architecture that uses
#                  attention mechanisms instead of recurrence. They were
#                  introduced in the 2017 paper 'Attention is All You Need'.
#                  What aspect would you like to explore - the architecture
#                  itself or how to implement one?",
#   insights=[
#     Insight(
#       type="pattern",
#       description="You're building a strong foundation in modern NLP",
#       evidence=["transformers", "attention", "NLP", "sequence modeling"],
#       importance=0.85
#     )
#   ],
#   recommendations=[
#     "Try implementing a simple attention mechanism",
#     "Explore the transformer architecture visually"
#   ],
#   knowledge_gaps=["Positional encoding", "Multi-head attention details"],
#   connections=[...],
#   confidence=0.92
# )
```

**Performance Targets:**
- Response time: < 200ms (P95)
- Synthesis quality: > 90% coherence score
- Insight relevance: > 85%

---

## Communication Protocol

### Message Format

All inter-agent communication uses a standardized JSON message format:

```python
@dataclass
class AgentMessage:
    """Standard message format for agent communication"""

    # Message metadata
    message_id: str                 # Unique message ID
    timestamp: datetime             # Message timestamp
    sender: str                     # Sending agent name
    recipient: Optional[str]        # Target agent (None = broadcast)
    message_type: str               # Type: request, response, event

    # Message content
    payload: Dict[str, Any]         # Message data
    context: Dict[str, Any]         # Shared context

    # Tracking
    correlation_id: str             # Request correlation ID
    parent_id: Optional[str]        # Parent message (for threading)

    # Metadata
    priority: int                   # Priority (0-10, default 5)
    ttl: int                        # Time-to-live in seconds

    def to_json(self) -> str:
        """Serialize to JSON"""
        pass

    @classmethod
    def from_json(cls, json_str: str) -> 'AgentMessage':
        """Deserialize from JSON"""
        pass
```

### Message Types

**1. Request Messages**
```python
# ConversationAgent requesting analysis
{
    "message_id": "msg_abc123",
    "timestamp": "2025-11-21T10:30:00Z",
    "sender": "conversation_agent",
    "recipient": "analysis_agent",
    "message_type": "request",
    "payload": {
        "action": "analyze",
        "user_input": "I'm learning about neural networks",
        "context": {...}
    },
    "context": {
        "session_id": "sess_xyz",
        "user_id": "user_123"
    },
    "correlation_id": "req_001",
    "priority": 7
}
```

**2. Response Messages**
```python
# AnalysisAgent responding
{
    "message_id": "msg_def456",
    "timestamp": "2025-11-21T10:30:00.300Z",
    "sender": "analysis_agent",
    "recipient": "conversation_agent",
    "message_type": "response",
    "payload": {
        "status": "success",
        "result": {
            "entities": [...],
            "concepts": [...],
            "themes": [...]
        }
    },
    "context": {...},
    "correlation_id": "req_001",
    "parent_id": "msg_abc123"
}
```

**3. Event Messages (Broadcast)**
```python
# Agent broadcasting an event
{
    "message_id": "msg_ghi789",
    "timestamp": "2025-11-21T10:30:01Z",
    "sender": "analysis_agent",
    "recipient": null,  # Broadcast to all
    "message_type": "event",
    "payload": {
        "event": "concept_detected",
        "data": {
            "concept": "neural networks",
            "confidence": 0.95
        }
    },
    "context": {...},
    "correlation_id": "req_001"
}
```

### Communication Patterns

**1. Sequential Pattern**
```
User Input
    │
    ▼
ConversationAgent
    │
    ├─→ (analysis_request) ─→ AnalysisAgent
    │                              │
    │                              ▼
    │   ← (analysis_response) ←───┘
    │
    ├─→ (research_request) ─→ ResearchAgent
    │                              │
    │                              ▼
    │   ← (research_response) ←───┘
    │
    ▼
Response
```

**2. Parallel Pattern (Default for complex queries)**
```
User Input
    │
    ▼
Orchestrator
    │
    ├─→ ConversationAgent ─────┐
    │                          │
    ├─→ AnalysisAgent ─────────┼─→ (parallel execution)
    │                          │
    ├─→ ResearchAgent ─────────┘
    │
    └─→ (wait for all) ─→ SynthesisAgent ─→ Response
```

**3. Event-Driven Pattern**
```
AnalysisAgent
    │
    └─→ (broadcast: concept_detected)
            │
            ├─→ ConversationAgent (updates context)
            ├─→ ResearchAgent (triggers research)
            └─→ SynthesisAgent (records for insights)
```

### Error Handling

```python
class AgentError(Exception):
    """Base agent error"""
    def __init__(
        self,
        agent_name: str,
        error_type: str,
        message: str,
        recoverable: bool = True
    ):
        self.agent_name = agent_name
        self.error_type = error_type
        self.message = message
        self.recoverable = recoverable

# Error types
ERROR_TYPES = [
    "timeout",          # Agent timeout
    "api_error",        # Upstream API error
    "validation",       # Input validation error
    "resource",         # Resource exhaustion
    "internal"          # Internal agent error
]

# Error response message
{
    "message_type": "error",
    "payload": {
        "error_type": "timeout",
        "error_message": "Agent exceeded 5s timeout",
        "recoverable": True,
        "fallback_available": True
    }
}
```

**Error Recovery Strategy:**

1. **Timeout Errors**: Return partial results or fallback response
2. **API Errors**: Retry with exponential backoff (max 3 attempts)
3. **Validation Errors**: Request clarification from user
4. **Resource Errors**: Queue request for later processing
5. **Internal Errors**: Log and return graceful degradation

---

## State Management

### Shared State Architecture

```python
class SharedStateManager:
    """
    Manages shared state across all agents.

    State layers:
    1. Conversation context (Redis)
    2. Agent-specific state (in-memory)
    3. Result cache (Redis)
    4. Persistent storage (SQLite)
    """

    def __init__(self):
        self.redis_client = redis.AsyncRedis()
        self.agent_states: Dict[str, AgentState] = {}

    async def get_conversation_context(
        self,
        session_id: str
    ) -> ConversationContext:
        """Retrieve conversation context from Redis"""
        pass

    async def update_conversation_context(
        self,
        session_id: str,
        update: ContextUpdate
    ) -> None:
        """Update conversation context"""
        pass

    async def get_agent_state(
        self,
        agent_name: str,
        session_id: str
    ) -> AgentState:
        """Get agent-specific state"""
        pass

    async def update_agent_state(
        self,
        agent_name: str,
        session_id: str,
        state: AgentState
    ) -> None:
        """Update agent state"""
        pass

    async def cache_result(
        self,
        key: str,
        result: Any,
        ttl: int = 300
    ) -> None:
        """Cache agent result"""
        pass

    async def get_cached_result(
        self,
        key: str
    ) -> Optional[Any]:
        """Retrieve cached result"""
        pass
```

### State Schemas

**1. Conversation Context**
```python
@dataclass
class ConversationContext:
    """Shared conversation context"""
    session_id: str
    user_id: Optional[str]

    # Conversation history
    messages: List[Message]          # Recent messages (last 10)
    exchanges: List[Exchange]        # User-agent exchanges (last 5)

    # Extracted knowledge
    concepts: List[str]              # All mentioned concepts
    entities: List[str]              # All mentioned entities
    themes: List[str]                # Conversation themes

    # Metadata
    created_at: datetime
    last_activity: datetime
    exchange_count: int

    # Agent insights
    agent_data: Dict[str, Any]       # Agent-specific context data
```

**2. Agent State**
```python
@dataclass
class AgentState:
    """Agent-specific state"""
    agent_name: str
    session_id: str

    # Execution state
    status: str                      # idle, processing, waiting
    current_task: Optional[str]      # Current task ID
    last_update: datetime

    # Agent memory
    memory: Dict[str, Any]           # Agent-specific memory
    cache: Dict[str, Any]            # Cached computations

    # Performance metrics
    invocation_count: int
    avg_response_time: float
    error_count: int
```

### State Synchronization

**Redis Keys Structure:**
```
# Conversation context
session:{session_id}:context             → ConversationContext JSON
session:{session_id}:messages            → List of recent messages
session:{session_id}:concepts            → Set of concepts

# Agent states
agent:{agent_name}:{session_id}:state    → AgentState JSON
agent:{agent_name}:{session_id}:cache    → Agent cache

# Result cache
result:{agent_name}:{hash}               → Cached result
```

**Update Strategy:**
```python
async def update_context_transactional(
    session_id: str,
    updates: List[ContextUpdate]
) -> None:
    """
    Update context with transaction semantics.

    Ensures:
    - Atomic updates
    - Conflict resolution
    - State consistency
    """
    async with redis_client.pipeline() as pipe:
        # Load current context
        context = await get_conversation_context(session_id)

        # Apply updates
        for update in updates:
            context = apply_update(context, update)

        # Write back
        await pipe.set(
            f"session:{session_id}:context",
            context.to_json(),
            ex=1800  # 30 min TTL
        )
        await pipe.execute()
```

### State Isolation vs Sharing

**Isolated State (Agent-Specific):**
- Tool execution results
- Intermediate computations
- Agent-specific cache
- Internal processing state

**Shared State (Cross-Agent):**
- Conversation history
- Extracted concepts/entities
- User preferences
- Session metadata

**Principle:** *Minimize shared state, maximize message passing*

---

## Orchestration Strategy

### LangGraph State Machine

```python
from langgraph.graph import StateGraph, END

class AgentOrchestrator:
    """
    LangGraph-based agent orchestration.

    Implements state graph for agent coordination:
    - Input classification
    - Agent routing
    - Parallel execution
    - Result synthesis
    """

    def __init__(self):
        self.agents = {
            'conversation': ConversationAgent(),
            'analysis': AnalysisAgent(),
            'research': ResearchAgent(),
            'vision': VisionAgent(),
            'synthesis': SynthesisAgent()
        }

        # Build LangGraph state machine
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        Build agent orchestration graph.

        Graph structure:
        START → classify → route → [agents] → synthesize → END
        """
        workflow = StateGraph()

        # Add nodes
        workflow.add_node("classify", self._classify_input)
        workflow.add_node("route", self._route_to_agents)
        workflow.add_node("execute_agents", self._execute_agents)
        workflow.add_node("synthesize", self._synthesize_results)

        # Add edges
        workflow.set_entry_point("classify")
        workflow.add_edge("classify", "route")
        workflow.add_edge("route", "execute_agents")
        workflow.add_edge("execute_agents", "synthesize")
        workflow.add_edge("synthesize", END)

        return workflow.compile()

    async def process_request(
        self,
        user_input: str,
        session_id: str,
        modality: str = "text"
    ) -> OrchestrationResult:
        """
        Process user request through agent graph.

        Args:
            user_input: User message or input data
            session_id: Session identifier
            modality: Input modality (text, audio, image)

        Returns:
            OrchestrationResult with final response
        """
        # Initialize state
        initial_state = {
            "user_input": user_input,
            "session_id": session_id,
            "modality": modality,
            "timestamp": datetime.now(),
            "agent_results": {},
            "errors": []
        }

        # Execute graph
        final_state = await self.graph.ainvoke(initial_state)

        return OrchestrationResult(
            response=final_state["final_response"],
            agent_results=final_state["agent_results"],
            metadata=final_state.get("metadata", {}),
            errors=final_state["errors"]
        )
```

### Classification Logic

```python
async def _classify_input(self, state: Dict) -> Dict:
    """
    Classify input to determine agent routing.

    Classifications:
    - simple_query: Conversation only
    - complex_query: Conversation + Analysis + Research
    - visual_query: Vision + Conversation + Analysis
    - insight_request: All agents for deep analysis
    """
    user_input = state["user_input"]
    modality = state["modality"]

    # Simple heuristics (can be enhanced with ML)
    if modality == "image":
        classification = "visual_query"
    elif len(user_input.split()) < 10:
        classification = "simple_query"
    elif any(word in user_input.lower() for word in
             ["explain", "research", "what is", "how does"]):
        classification = "complex_query"
    elif "insight" in user_input.lower() or "summary" in user_input.lower():
        classification = "insight_request"
    else:
        classification = "simple_query"

    state["classification"] = classification
    return state
```

### Routing Strategy

```python
async def _route_to_agents(self, state: Dict) -> Dict:
    """
    Route to appropriate agents based on classification.

    Routing rules:
    - simple_query: [conversation]
    - complex_query: [conversation, analysis, research]
    - visual_query: [vision, conversation, analysis]
    - insight_request: [all agents]
    """
    classification = state["classification"]

    routing_map = {
        "simple_query": ["conversation"],
        "complex_query": ["conversation", "analysis", "research"],
        "visual_query": ["vision", "conversation", "analysis"],
        "insight_request": ["conversation", "analysis", "research", "synthesis"]
    }

    state["selected_agents"] = routing_map.get(
        classification,
        ["conversation"]  # Default fallback
    )
    state["execution_mode"] = "parallel" if len(state["selected_agents"]) > 1 else "single"

    return state
```

### Parallel Execution

```python
async def _execute_agents(self, state: Dict) -> Dict:
    """
    Execute selected agents in parallel.

    Uses asyncio.gather for concurrent execution.
    Handles timeouts and errors gracefully.
    """
    selected_agents = state["selected_agents"]
    execution_mode = state["execution_mode"]

    # Get conversation context
    context = await shared_state.get_conversation_context(
        state["session_id"]
    )

    if execution_mode == "single":
        # Single agent execution
        agent = self.agents[selected_agents[0]]
        result = await agent.process(state["user_input"], context)
        state["agent_results"][selected_agents[0]] = result

    else:
        # Parallel execution with timeout
        tasks = []
        for agent_name in selected_agents:
            if agent_name == "synthesis":
                continue  # Skip synthesis, run after others

            agent = self.agents[agent_name]
            task = asyncio.create_task(
                self._execute_agent_with_timeout(
                    agent, state["user_input"], context, agent_name
                )
            )
            tasks.append((agent_name, task))

        # Wait for all agents (with timeout)
        results = await asyncio.gather(
            *[task for _, task in tasks],
            return_exceptions=True
        )

        # Collect results
        for (agent_name, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                state["errors"].append({
                    "agent": agent_name,
                    "error": str(result)
                })
                # Use fallback
                state["agent_results"][agent_name] = None
            else:
                state["agent_results"][agent_name] = result

    return state

async def _execute_agent_with_timeout(
    self,
    agent: Any,
    user_input: str,
    context: ConversationContext,
    agent_name: str,
    timeout: float = 5.0
) -> Any:
    """Execute agent with timeout"""
    try:
        return await asyncio.wait_for(
            agent.process(user_input, context),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        logger.warning(f"Agent {agent_name} timed out after {timeout}s")
        raise
```

### When to Use Single vs Multiple Agents

**Single Agent (ConversationAgent only):**
- Simple greetings: "Hello", "Hi there"
- Short statements: "I'm learning Python"
- Follow-up acknowledgments: "Got it", "Thanks"
- Session control: "Goodbye", "End conversation"

**Multiple Agents (Parallel):**
- Complex questions: "How do transformers work in NLP?"
- Learning requests: "Explain backpropagation in neural networks"
- Visual input: Any image or document upload
- Deep analysis: "What patterns do you see in my learning?"

**Decision Algorithm:**
```python
def should_use_multi_agent(
    user_input: str,
    context: ConversationContext,
    modality: str
) -> bool:
    """Determine if multi-agent processing is needed"""

    # Always use multi-agent for visual input
    if modality in ["image", "document"]:
        return True

    # Short inputs typically don't need it
    if len(user_input.split()) < 5:
        return False

    # Check for complexity indicators
    complexity_indicators = [
        "how", "why", "explain", "what is", "help me understand",
        "research", "find", "show me", "compare", "analyze"
    ]

    has_complexity = any(
        indicator in user_input.lower()
        for indicator in complexity_indicators
    )

    # Check conversation depth
    is_deep_conversation = context.exchange_count > 5

    return has_complexity or is_deep_conversation
```

---

## Integration with v1.0

### Backward Compatibility Strategy

**API Versioning:**
```
v1 endpoints (existing):
/api/conversation          → Single ConversationHandler (v1.0)
/api/search               → SQLite FTS5
/ws/{session_id}          → v1.0 WebSocket

v2 endpoints (new):
/v2/conversation          → Multi-agent orchestration
/v2/search                → Semantic search (future)
/v2/ws/{session_id}       → v2.0 WebSocket with agent metadata
```

**Migration Approach:**
```python
# app/main.py - Hybrid routing

@app.post("/api/conversation")
async def conversation_v1(request: ConversationRequest):
    """
    v1.0 endpoint - maintains existing behavior
    Uses single ConversationHandler for backward compatibility
    """
    return await conversation_handler_v1.process(request)

@app.post("/v2/conversation")
async def conversation_v2(request: ConversationRequestV2):
    """
    v2.0 endpoint - multi-agent orchestration
    Requires explicit opt-in from clients
    """
    return await agent_orchestrator.process(request)

@app.post("/api/conversation/smart")
async def conversation_smart(request: ConversationRequest):
    """
    Smart endpoint - automatically routes to v1 or v2
    Based on feature flags and request complexity
    """
    if should_use_v2(request):
        return await agent_orchestrator.process(request)
    else:
        return await conversation_handler_v1.process(request)
```

### Feature Flags

```python
# app/config.py

class FeatureFlags:
    """Feature flags for gradual rollout"""

    # Agent system
    ENABLE_MULTI_AGENT: bool = False
    ENABLE_ANALYSIS_AGENT: bool = False
    ENABLE_RESEARCH_AGENT: bool = False
    ENABLE_VISION_AGENT: bool = False
    ENABLE_SYNTHESIS_AGENT: bool = False

    # Routing
    V2_TRAFFIC_PERCENTAGE: int = 0    # 0-100
    V2_USER_WHITELIST: List[str] = [] # User IDs for beta

    # Fallback
    V2_FALLBACK_TO_V1: bool = True    # Fallback on errors

    @classmethod
    def should_use_v2(
        cls,
        request: ConversationRequest,
        user_id: Optional[str] = None
    ) -> bool:
        """Determine if request should use v2"""

        # Check if multi-agent is enabled
        if not cls.ENABLE_MULTI_AGENT:
            return False

        # Check whitelist
        if user_id in cls.V2_USER_WHITELIST:
            return True

        # Check traffic percentage (random sampling)
        if random.randint(0, 100) < cls.V2_TRAFFIC_PERCENTAGE:
            return True

        return False
```

### Data Migration Path

**Phase 1: Dual Write (Weeks 1-2)**
```python
# Write to both v1 and v2 storage simultaneously
async def save_exchange(exchange: Exchange):
    # v1: SQLite
    await db_v1.save_exchange(exchange)

    # v2: PostgreSQL (if enabled)
    if ENABLE_V2_STORAGE:
        await db_v2.save_exchange(exchange)
```

**Phase 2: Dual Read (Weeks 3-4)**
```python
# Read from v2 with fallback to v1
async def get_conversation_history(session_id: str):
    if ENABLE_V2_STORAGE:
        try:
            return await db_v2.get_history(session_id)
        except Exception:
            logger.warning("v2 read failed, falling back to v1")
            return await db_v1.get_history(session_id)
    else:
        return await db_v1.get_history(session_id)
```

**Phase 3: Full Migration (Weeks 5-6)**
```python
# Migrate all historical data from v1 to v2
async def migrate_all_data():
    async for batch in db_v1.iter_batches(size=100):
        await db_v2.batch_insert(batch)
        await verify_batch(batch)
```

### Gradual Rollout Plan

```
Week 1-2: Development & Testing
├─ Build multi-agent system
├─ Feature flags: All OFF
└─ v2 endpoints available for testing

Week 3-4: Internal Beta (5%)
├─ V2_TRAFFIC_PERCENTAGE = 5
├─ V2_USER_WHITELIST = [internal team]
└─ Monitor metrics closely

Week 5-6: Public Beta (25%)
├─ V2_TRAFFIC_PERCENTAGE = 25
├─ Collect user feedback
└─ Fix identified issues

Week 7-8: Gradual Increase (50%)
├─ V2_TRAFFIC_PERCENTAGE = 50
└─ Performance optimization

Week 9-10: Majority Traffic (75%)
├─ V2_TRAFFIC_PERCENTAGE = 75
└─ Prepare for full cutover

Week 11-12: Full Cutover (100%)
├─ V2_TRAFFIC_PERCENTAGE = 100
├─ Deprecate v1 endpoints (6-month notice)
└─ Begin v1 decommissioning
```

---

## Implementation Guide

### Directory Structure

```
learning_voice_agent/
├── app/
│   ├── agents/                    # NEW: Multi-agent system
│   │   ├── __init__.py
│   │   ├── base.py               # Base agent interface
│   │   ├── conversation.py       # ConversationAgent
│   │   ├── analysis.py           # AnalysisAgent
│   │   ├── research.py           # ResearchAgent
│   │   ├── vision.py             # VisionAgent
│   │   ├── synthesis.py          # SynthesisAgent
│   │   └── tools/                # Agent tools
│   │       ├── web_search.py
│   │       ├── wikipedia.py
│   │       └── calculator.py
│   │
│   ├── orchestration/            # NEW: Agent orchestration
│   │   ├── __init__.py
│   │   ├── orchestrator.py       # LangGraph orchestrator
│   │   ├── state_manager.py      # Shared state management
│   │   ├── message_bus.py        # Agent messaging
│   │   └── routing.py            # Agent routing logic
│   │
│   ├── schemas/                  # NEW: Agent schemas
│   │   ├── __init__.py
│   │   ├── messages.py           # Message schemas
│   │   ├── agent_results.py      # Result schemas
│   │   └── state.py              # State schemas
│   │
│   ├── main.py                   # FastAPI app (updated)
│   ├── conversation_handler.py   # v1.0 handler (preserved)
│   ├── audio_pipeline.py         # Existing
│   ├── database.py               # Existing
│   ├── state_manager.py          # Existing (enhanced)
│   └── config.py                 # Config (with feature flags)
│
├── tests/
│   ├── agents/                   # NEW: Agent tests
│   │   ├── test_conversation_agent.py
│   │   ├── test_analysis_agent.py
│   │   ├── test_research_agent.py
│   │   ├── test_vision_agent.py
│   │   └── test_synthesis_agent.py
│   │
│   └── orchestration/            # NEW: Orchestration tests
│       ├── test_orchestrator.py
│       ├── test_state_manager.py
│       └── test_routing.py
│
└── docs/
    ├── PHASE2_AGENT_ARCHITECTURE.md  # This document
    └── AGENT_API_REFERENCE.md        # Agent API docs
```

### Base Agent Interface

```python
# app/agents/base.py

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass
import time

@dataclass
class AgentMetrics:
    """Agent performance metrics"""
    invocations: int = 0
    total_time: float = 0.0
    errors: int = 0
    avg_latency: float = 0.0

class BaseAgent(ABC):
    """
    Base class for all agents.

    Provides:
    - Common interface
    - Metrics tracking
    - Error handling
    - Logging
    """

    def __init__(self, name: str):
        self.name = name
        self.metrics = AgentMetrics()
        self.logger = logging.getLogger(f"agent.{name}")

    @abstractmethod
    async def process(
        self,
        user_input: str,
        context: ConversationContext,
        **kwargs
    ) -> Any:
        """
        Process input and return result.

        Must be implemented by subclasses.
        """
        pass

    async def execute_with_metrics(
        self,
        user_input: str,
        context: ConversationContext,
        **kwargs
    ) -> Any:
        """
        Execute agent with metrics tracking.
        """
        start_time = time.time()

        try:
            result = await self.process(user_input, context, **kwargs)

            # Update metrics
            elapsed = time.time() - start_time
            self.metrics.invocations += 1
            self.metrics.total_time += elapsed
            self.metrics.avg_latency = (
                self.metrics.total_time / self.metrics.invocations
            )

            self.logger.info(
                f"{self.name} completed",
                extra={
                    "latency_ms": elapsed * 1000,
                    "invocation": self.metrics.invocations
                }
            )

            return result

        except Exception as e:
            self.metrics.errors += 1
            self.logger.error(
                f"{self.name} error",
                extra={"error": str(e)},
                exc_info=True
            )
            raise

    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics"""
        return {
            "name": self.name,
            "invocations": self.metrics.invocations,
            "avg_latency_ms": self.metrics.avg_latency * 1000,
            "errors": self.metrics.errors,
            "error_rate": (
                self.metrics.errors / max(self.metrics.invocations, 1)
            )
        }
```

### Implementation Steps

**Week 1: Foundation**
```bash
# Day 1-2: Setup
- Create agent directory structure
- Implement BaseAgent interface
- Set up LangGraph dependencies
- Create message schemas

# Day 3-4: ConversationAgent
- Port existing conversation_handler to ConversationAgent
- Upgrade to Claude 3.5 Sonnet
- Add agent message handling
- Write unit tests

# Day 5: Integration
- Update main.py with /v2/conversation endpoint
- Add feature flags
- Test end-to-end flow
```

**Week 2: Additional Agents**
```bash
# Day 1-2: AnalysisAgent
- Implement entity extraction
- Add concept detection
- Create analysis schemas
- Write tests

# Day 3-4: ResearchAgent
- Implement web search tool
- Add Wikipedia integration
- Create tool execution framework
- Write tests

# Day 5: Testing
- Integration tests for multi-agent
- Performance testing
- Bug fixes
```

### Code Example: Complete Flow

```python
# Example: Processing a complex query

# 1. User sends request
request = {
    "text": "How do transformers work in NLP?",
    "session_id": "abc-123"
}

# 2. Orchestrator processes
orchestrator = AgentOrchestrator()
result = await orchestrator.process_request(
    user_input=request["text"],
    session_id=request["session_id"]
)

# 3. Behind the scenes:
# - Classifier determines: "complex_query"
# - Router selects: [conversation, analysis, research]
# - Agents execute in parallel:
#   - ConversationAgent: Drafts response (800ms)
#   - AnalysisAgent: Extracts concepts (250ms)
#   - ResearchAgent: Searches for facts (600ms)
# - SynthesisAgent: Merges results (150ms)

# 4. Final response
{
    "final_response": "Transformers are neural network architectures...",
    "insights": [
        {
            "type": "pattern",
            "description": "You're exploring modern NLP architectures",
            "importance": 0.85
        }
    ],
    "concepts": ["transformers", "attention", "NLP"],
    "sources": [
        {
            "title": "Attention is All You Need",
            "url": "https://arxiv.org/abs/1706.03762"
        }
    ],
    "metadata": {
        "agents_used": ["conversation", "analysis", "research", "synthesis"],
        "total_time_ms": 950,
        "agent_times_ms": {
            "conversation": 800,
            "analysis": 250,
            "research": 600,
            "synthesis": 150
        }
    }
}
```

---

## Testing Strategy

### Test Pyramid

```
        /\
       /E2E\          5% - End-to-end multi-agent flows
      /────\
     /Integr\         25% - Agent integration, orchestration
    /────────\
   /   Unit   \       70% - Individual agent logic, tools
  /────────────\
```

### Unit Tests

```python
# tests/agents/test_analysis_agent.py

import pytest
from app.agents.analysis import AnalysisAgent
from app.schemas.state import ConversationContext

@pytest.fixture
def analysis_agent():
    return AnalysisAgent()

@pytest.fixture
def sample_context():
    return ConversationContext(
        session_id="test_session",
        messages=[],
        concepts=[],
        entities=[]
    )

@pytest.mark.asyncio
async def test_extract_entities(analysis_agent, sample_context):
    """Test entity extraction"""
    result = await analysis_agent.process(
        user_input="I'm learning about neural networks and deep learning",
        context=sample_context
    )

    # Verify entities extracted
    assert len(result.entities) >= 2
    assert any(e.text == "neural networks" for e in result.entities)
    assert any(e.text == "deep learning" for e in result.entities)

    # Verify entity types
    assert all(e.type in ["TOPIC", "CONCEPT"] for e in result.entities)

    # Verify confidence scores
    assert all(e.confidence > 0.5 for e in result.entities)

@pytest.mark.asyncio
async def test_extract_concepts(analysis_agent, sample_context):
    """Test concept extraction"""
    result = await analysis_agent.process(
        user_input="Transformers use attention mechanisms for sequence processing",
        context=sample_context
    )

    # Verify concepts
    assert len(result.concepts) >= 1
    concept_names = [c.name for c in result.concepts]
    assert any("attention" in name.lower() for name in concept_names)

    # Verify categories
    assert all(c.category in ["AI/ML", "NLP", "tech"] for c in result.concepts)

@pytest.mark.asyncio
async def test_performance(analysis_agent, sample_context):
    """Test analysis performance"""
    import time

    start = time.time()
    result = await analysis_agent.process(
        user_input="I'm studying machine learning algorithms",
        context=sample_context
    )
    elapsed = time.time() - start

    # Should complete in under 500ms
    assert elapsed < 0.5, f"Analysis took {elapsed*1000}ms"
```

### Integration Tests

```python
# tests/orchestration/test_orchestrator.py

@pytest.mark.asyncio
async def test_multi_agent_orchestration():
    """Test full multi-agent orchestration"""
    orchestrator = AgentOrchestrator()

    result = await orchestrator.process_request(
        user_input="Explain how neural networks learn",
        session_id="test_session"
    )

    # Verify agents were called
    assert "conversation" in result.metadata["agents_used"]
    assert "analysis" in result.metadata["agents_used"]

    # Verify response quality
    assert len(result.response) > 50
    assert "neural network" in result.response.lower()

    # Verify performance
    assert result.metadata["total_time_ms"] < 1500

@pytest.mark.asyncio
async def test_parallel_execution():
    """Test agents execute in parallel"""
    orchestrator = AgentOrchestrator()

    import time
    start = time.time()

    result = await orchestrator.process_request(
        user_input="Research transformers in NLP",
        session_id="test_session"
    )

    elapsed = time.time() - start

    # Parallel execution should be faster than sequential
    # (sequential would be sum of all agent times)
    agent_times = result.metadata["agent_times_ms"]
    sequential_time = sum(agent_times.values())
    parallel_time = elapsed * 1000

    # Parallel should be significantly faster
    assert parallel_time < sequential_time * 0.7

@pytest.mark.asyncio
async def test_graceful_degradation():
    """Test system works even if agent fails"""
    orchestrator = AgentOrchestrator()

    # Mock research agent to fail
    orchestrator.agents["research"] = MockFailingAgent()

    result = await orchestrator.process_request(
        user_input="Complex query requiring research",
        session_id="test_session"
    )

    # Should still get response despite research failure
    assert result.response
    assert len(result.errors) > 0
    assert result.errors[0]["agent"] == "research"
```

### End-to-End Tests

```python
# tests/e2e/test_complete_flow.py

@pytest.mark.asyncio
async def test_complete_conversation_flow(test_client):
    """Test complete conversation flow through API"""

    # 1. Start conversation
    response = await test_client.post("/v2/conversation", json={
        "text": "I'm learning about deep learning",
        "session_id": "e2e_test"
    })
    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert "response" in data
    assert "concepts" in data
    assert "metadata" in data

    # 2. Follow-up with complex query
    response = await test_client.post("/v2/conversation", json={
        "text": "How do transformers work?",
        "session_id": "e2e_test"
    })
    assert response.status_code == 200
    data = response.json()

    # Should use multiple agents
    assert len(data["metadata"]["agents_used"]) > 1

    # Should have research results
    if "sources" in data:
        assert len(data["sources"]) > 0

    # 3. Request insights
    response = await test_client.post("/v2/conversation", json={
        "text": "What patterns do you see in my learning?",
        "session_id": "e2e_test"
    })
    assert response.status_code == 200
    data = response.json()

    # Should have insights
    assert "insights" in data
    assert len(data["insights"]) > 0
```

### Performance Tests

```python
# tests/performance/test_latency.py

import pytest
import asyncio
from locust import HttpUser, task, between

class ConversationUser(HttpUser):
    """Load test user for conversation endpoint"""
    wait_time = between(1, 3)

    @task
    def send_simple_message(self):
        """Test simple message performance"""
        self.client.post("/v2/conversation", json={
            "text": "Hello, how are you?",
            "session_id": self.generate_session_id()
        })

    @task(2)
    def send_complex_message(self):
        """Test complex message performance"""
        self.client.post("/v2/conversation", json={
            "text": "Explain how neural networks learn from data",
            "session_id": self.generate_session_id()
        })

    def generate_session_id(self):
        import uuid
        return f"load_test_{uuid.uuid4()}"

# Run with: locust -f test_latency.py --users 50 --spawn-rate 10
```

---

## Performance Requirements

### Response Time Targets

| Scenario | Target (P95) | Max (P99) |
|----------|--------------|-----------|
| Simple query (single agent) | 800ms | 1000ms |
| Complex query (multi-agent) | 1200ms | 1500ms |
| Visual query (with image) | 1500ms | 2000ms |
| Insight generation | 2000ms | 3000ms |

### Component Latency Budgets

```
Total Request: 1200ms (P95)
│
├─ Input Processing: 50ms
│   ├─ Classification: 20ms
│   └─ Routing: 30ms
│
├─ Agent Execution: 950ms (parallel)
│   ├─ ConversationAgent: 800ms
│   ├─ AnalysisAgent: 250ms
│   └─ ResearchAgent: 600ms
│
├─ Synthesis: 150ms
│
└─ Response Formatting: 50ms
```

### Throughput Requirements

- **Concurrent sessions:** Support 50+ concurrent conversations
- **Requests per second:** Handle 20+ req/s sustained
- **Peak throughput:** Handle 50 req/s for 1 minute bursts

### Resource Limits

```python
# Agent-specific timeouts
AGENT_TIMEOUTS = {
    "conversation": 5.0,  # 5 seconds
    "analysis": 2.0,      # 2 seconds
    "research": 5.0,      # 5 seconds (tools may be slow)
    "vision": 8.0,        # 8 seconds (GPT-4V can be slow)
    "synthesis": 2.0      # 2 seconds
}

# Resource limits
MAX_CONCURRENT_AGENTS = 5       # Max parallel agents per request
MAX_MESSAGE_SIZE = 10_000       # Max message size (chars)
MAX_CONTEXT_SIZE = 50_000       # Max context size (chars)
MAX_AGENT_RETRIES = 2           # Max retries per agent
```

### Monitoring Metrics

```python
# Metrics to track
METRICS = [
    # Latency
    "request_latency_ms",
    "agent_latency_ms_by_agent",
    "synthesis_latency_ms",

    # Throughput
    "requests_per_second",
    "agents_invoked_per_second",

    # Errors
    "error_rate_by_agent",
    "timeout_rate_by_agent",
    "fallback_rate",

    # Resource usage
    "concurrent_requests",
    "redis_operations_per_second",
    "api_calls_per_second",

    # Quality
    "multi_agent_percentage",
    "average_agents_per_request",
    "synthesis_quality_score"
]
```

---

## Migration Path

### Phase 1: Setup (Week 1)

**Objectives:**
- Set up agent infrastructure
- Create base classes
- Implement feature flags

**Tasks:**
1. Create directory structure
2. Implement `BaseAgent` class
3. Set up LangGraph dependencies
4. Create message schemas
5. Add feature flags to config
6. Create /v2/conversation endpoint (returns 501 Not Implemented)

**Validation:**
- [ ] Directory structure created
- [ ] BaseAgent tests passing
- [ ] Feature flags functional
- [ ] v2 endpoint returns proper error

### Phase 2: ConversationAgent (Week 1-2)

**Objectives:**
- Port existing logic to ConversationAgent
- Upgrade to Claude 3.5 Sonnet
- Implement agent interface

**Tasks:**
1. Create ConversationAgent class
2. Port conversation_handler logic
3. Upgrade to Claude 3.5 Sonnet
4. Add agent message handling
5. Write comprehensive tests
6. Performance benchmarking

**Validation:**
- [ ] ConversationAgent functional
- [ ] All v1 features preserved
- [ ] Latency < 900ms (P95)
- [ ] Tests coverage > 80%

### Phase 3: AnalysisAgent (Week 2)

**Objectives:**
- Build entity and concept extraction
- Integrate with ConversationAgent

**Tasks:**
1. Create AnalysisAgent class
2. Implement entity extraction
3. Add concept detection
4. Create analysis schemas
5. Write tests
6. Integration with orchestrator

**Validation:**
- [ ] Entity extraction > 85% precision
- [ ] Concept detection working
- [ ] Latency < 300ms (P95)
- [ ] Integration tests passing

### Phase 4: Orchestration (Week 3)

**Objectives:**
- Implement LangGraph orchestration
- Enable parallel agent execution

**Tasks:**
1. Create AgentOrchestrator class
2. Build state graph
3. Implement routing logic
4. Add parallel execution
5. Create SharedStateManager
6. Write orchestration tests

**Validation:**
- [ ] Orchestrator routing correctly
- [ ] Parallel execution working
- [ ] State management functional
- [ ] End-to-end tests passing

### Phase 5: Beta Release (Week 3-4)

**Objectives:**
- Enable v2 for 5% traffic
- Monitor and fix issues

**Tasks:**
1. Enable V2_TRAFFIC_PERCENTAGE = 5
2. Set up monitoring dashboards
3. Monitor error rates
4. Fix identified issues
5. Collect performance metrics
6. Gather user feedback

**Validation:**
- [ ] v2 handling 5% traffic
- [ ] Error rate < 1%
- [ ] Latency targets met
- [ ] No critical bugs

### Weeks 5-12: Complete Remaining Agents

See [REBUILD_STRATEGY.md](REBUILD_STRATEGY.md) for full timeline.

---

## Conclusion

This architecture document provides a comprehensive blueprint for implementing a production-ready multi-agent system for the learning_voice_agent v2.0. Key highlights:

**Strengths:**
- **Modular design** with clear agent responsibilities
- **Scalable architecture** using LangGraph patterns
- **Backward compatible** with v1.0 system
- **Observable** with comprehensive metrics
- **Testable** with clear testing strategy

**Next Steps:**
1. Review and approve this architecture
2. Begin Phase 1 implementation
3. Set up monitoring infrastructure
4. Create detailed implementation tickets

**Success Metrics:**
- ✅ 5 agents operational
- ✅ Sub-1.5s latency (P95)
- ✅ Zero data loss
- ✅ Backward compatible
- ✅ Production-ready code

---

**Document Version:** 1.0
**Last Updated:** 2025-11-21
**Status:** Design Complete - Ready for Implementation
**Approvers:** [To be filled]

