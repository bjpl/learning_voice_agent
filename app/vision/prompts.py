"""
Vision Analysis Prompt Templates

SPECIFICATION:
- Task-specific vision prompts
- Context-aware prompt building
- Structured output formats
- Best practices for vision analysis

ARCHITECTURE:
- Template-based prompts
- Dynamic prompt composition
- JSON output formatting
- Domain-specific templates

WHY:
- Consistent high-quality results
- Reusable prompt patterns
- Easy to maintain and extend
"""
from typing import Dict, List, Optional
from enum import Enum


class VisionTask(str, Enum):
    """Vision analysis task types"""
    DESCRIBE = "describe"
    OCR = "ocr"
    DIAGRAM = "diagram"
    COMPARE = "compare"
    ANALYZE = "analyze"
    EXTRACT_DATA = "extract_data"
    CLASSIFY = "classify"
    DETECT_OBJECTS = "detect_objects"
    IDENTIFY_TEXT = "identify_text"
    EXPLAIN = "explain"


class VisionPromptTemplates:
    """
    Vision prompt templates for different tasks

    PATTERN: Template method pattern with composition
    WHY: Consistent prompts with customization
    """

    # ========== BASIC ANALYSIS PROMPTS ==========

    DESCRIBE_IMAGE = """Analyze this image in detail and provide a comprehensive description.

Include:
1. **Main Subject**: What is the primary focus of the image?
2. **Visual Elements**: Colors, shapes, composition, style
3. **Context**: Setting, environment, atmosphere
4. **Notable Details**: Interesting or significant features
5. **Quality Assessment**: Technical quality, lighting, clarity

Format your response as structured JSON:
{
  "main_subject": "Brief description",
  "visual_elements": {
    "colors": ["list of dominant colors"],
    "composition": "Description of layout and arrangement",
    "style": "Visual style or artistic approach"
  },
  "context": "Environmental and contextual details",
  "notable_details": ["List of interesting features"],
  "quality": {
    "overall": "Assessment of image quality",
    "lighting": "Lighting assessment",
    "clarity": "Clarity and sharpness assessment"
  }
}"""

    QUICK_DESCRIBE = """Provide a concise but informative description of this image in 2-3 sentences.
Focus on the main subject and most important visual elements."""

    # ========== OCR AND TEXT EXTRACTION ==========

    EXTRACT_TEXT = """Extract ALL visible text from this image with high accuracy.

Requirements:
1. Preserve exact text content (spelling, capitalization, punctuation)
2. Maintain text structure (headings, paragraphs, lists)
3. Indicate text position (top, middle, bottom, left, right)
4. Note text style (bold, italic, size variations)
5. Identify any unreadable or unclear text

Format as JSON:
{
  "text_blocks": [
    {
      "content": "Exact text content",
      "position": "Location in image",
      "style": "Text formatting",
      "confidence": "high|medium|low"
    }
  ],
  "full_text": "All text concatenated in reading order",
  "metadata": {
    "total_blocks": 0,
    "language": "Detected language",
    "readability": "Overall text clarity"
  }
}"""

    IDENTIFY_TEXT = """Identify and list all text visible in this image.
Focus on accuracy and completeness. Include even partial or small text."""

    # ========== DIAGRAM AND TECHNICAL ANALYSIS ==========

    ANALYZE_DIAGRAM = """Analyze this technical diagram or chart in detail.

Provide:
1. **Diagram Type**: Flow chart, UML, architecture, data flow, etc.
2. **Components**: List all elements (boxes, nodes, shapes)
3. **Connections**: Relationships and data flows
4. **Labels**: All text labels and annotations
5. **Purpose**: What the diagram represents or explains
6. **Technical Details**: Specific technical information conveyed

Format as JSON:
{
  "diagram_type": "Type of diagram",
  "purpose": "What it represents",
  "components": [
    {
      "type": "Component type",
      "label": "Component label",
      "description": "What it represents"
    }
  ],
  "connections": [
    {
      "from": "Source component",
      "to": "Target component",
      "type": "Connection type",
      "label": "Connection description"
    }
  ],
  "technical_details": "Detailed technical analysis",
  "interpretation": "High-level explanation"
}"""

    EXPLAIN_DIAGRAM = """Explain what this diagram shows to someone unfamiliar with the topic.
Focus on clarity and educational value."""

    # ========== IMAGE COMPARISON ==========

    COMPARE_IMAGES = """Compare these images and identify similarities and differences.

Analyze:
1. **Visual Similarities**: Shared colors, composition, style, subjects
2. **Visual Differences**: What varies between images
3. **Content Similarities**: Similar objects, themes, or concepts
4. **Content Differences**: Different elements or messages
5. **Quality Comparison**: Technical quality differences
6. **Overall Assessment**: Which is better for what purpose

Format as JSON:
{
  "similarities": {
    "visual": ["List of visual similarities"],
    "content": ["List of content similarities"]
  },
  "differences": {
    "visual": ["List of visual differences"],
    "content": ["List of content differences"]
  },
  "quality_comparison": "Quality assessment",
  "recommendation": "Which image for what use case"
}"""

    # ========== DATA EXTRACTION ==========

    EXTRACT_STRUCTURED_DATA = """Extract structured data from this image.

Look for:
1. Tables and grids
2. Forms and fields
3. Lists and enumerations
4. Key-value pairs
5. Numerical data
6. Dates and timestamps

Return data in JSON format with appropriate structure."""

    # ========== CLASSIFICATION ==========

    CLASSIFY_IMAGE = """Classify this image into appropriate categories.

Provide:
1. **Primary Category**: Main classification
2. **Secondary Categories**: Additional applicable categories
3. **Tags**: Descriptive keywords
4. **Content Type**: Photo, illustration, diagram, screenshot, etc.
5. **Subject Matter**: What the image depicts
6. **Use Cases**: Potential applications

Format as JSON:
{
  "primary_category": "Main category",
  "secondary_categories": ["Additional categories"],
  "tags": ["Descriptive keywords"],
  "content_type": "Type of image",
  "subject_matter": "What it shows",
  "use_cases": ["Potential uses"]
}"""

    # ========== OBJECT DETECTION ==========

    DETECT_OBJECTS = """Identify and list all objects visible in this image.

For each object provide:
1. Object name/type
2. Approximate location
3. Size (relative: large, medium, small)
4. Count (if multiple)
5. Notable characteristics

Format as JSON:
{
  "objects": [
    {
      "name": "Object name",
      "location": "Position in image",
      "size": "Relative size",
      "count": 1,
      "characteristics": "Notable features"
    }
  ],
  "total_objects": 0,
  "dominant_objects": ["Most prominent objects"]
}"""

    # ========== CONTEXTUAL ANALYSIS ==========

    ANALYZE_CONTEXT = """Analyze the context and meaning of this image.

Consider:
1. **Setting**: Where and when this might be
2. **Purpose**: Why this image was created
3. **Audience**: Who this is intended for
4. **Message**: What it communicates
5. **Emotional Tone**: Mood and feeling
6. **Cultural Context**: Any cultural elements

Format as JSON."""

    EDUCATIONAL_ANALYSIS = """Analyze this image from an educational perspective.

Identify:
1. Learning concepts shown
2. Educational value
3. Complexity level
4. Subject areas
5. Teaching opportunities
6. Age appropriateness

Format as JSON."""


class VisionPromptBuilder:
    """
    Build custom vision prompts dynamically

    PATTERN: Builder pattern for complex prompt construction
    WHY: Flexible prompt creation with validation
    """

    def __init__(self):
        self.task: Optional[VisionTask] = None
        self.context: Optional[str] = None
        self.constraints: List[str] = []
        self.output_format: str = "json"
        self.additional_instructions: List[str] = []

    def set_task(self, task: VisionTask) -> 'VisionPromptBuilder':
        """Set the vision task type"""
        self.task = task
        return self

    def add_context(self, context: str) -> 'VisionPromptBuilder':
        """Add contextual information"""
        self.context = context
        return self

    def add_constraint(self, constraint: str) -> 'VisionPromptBuilder':
        """Add a constraint or requirement"""
        self.constraints.append(constraint)
        return self

    def set_output_format(self, format: str) -> 'VisionPromptBuilder':
        """Set desired output format"""
        self.output_format = format
        return self

    def add_instruction(self, instruction: str) -> 'VisionPromptBuilder':
        """Add additional instruction"""
        self.additional_instructions.append(instruction)
        return self

    def build(self) -> str:
        """
        Build the final prompt

        Returns:
            Complete prompt string
        """
        if not self.task:
            raise ValueError("Task must be set before building prompt")

        # Get base template
        template = self._get_base_template()

        # Add context if provided
        if self.context:
            template = f"Context: {self.context}\n\n{template}"

        # Add constraints
        if self.constraints:
            constraints_text = "\n".join(f"- {c}" for c in self.constraints)
            template += f"\n\nConstraints:\n{constraints_text}"

        # Add additional instructions
        if self.additional_instructions:
            instructions_text = "\n".join(f"- {i}" for i in self.additional_instructions)
            template += f"\n\nAdditional Requirements:\n{instructions_text}"

        # Add output format instruction
        if self.output_format == "json":
            template += "\n\nProvide response in valid JSON format."

        return template

    def _get_base_template(self) -> str:
        """Get base template for task"""
        templates = {
            VisionTask.DESCRIBE: VisionPromptTemplates.DESCRIBE_IMAGE,
            VisionTask.OCR: VisionPromptTemplates.EXTRACT_TEXT,
            VisionTask.DIAGRAM: VisionPromptTemplates.ANALYZE_DIAGRAM,
            VisionTask.COMPARE: VisionPromptTemplates.COMPARE_IMAGES,
            VisionTask.EXTRACT_DATA: VisionPromptTemplates.EXTRACT_STRUCTURED_DATA,
            VisionTask.CLASSIFY: VisionPromptTemplates.CLASSIFY_IMAGE,
            VisionTask.DETECT_OBJECTS: VisionPromptTemplates.DETECT_OBJECTS,
        }
        return templates.get(self.task, VisionPromptTemplates.DESCRIBE_IMAGE)


# ========== HELPER FUNCTIONS ==========

def get_prompt_for_task(task: VisionTask, **kwargs) -> str:
    """
    Get appropriate prompt for vision task

    Args:
        task: Vision task type
        **kwargs: Additional parameters (context, constraints, etc.)

    Returns:
        Formatted prompt string
    """
    builder = VisionPromptBuilder().set_task(task)

    if "context" in kwargs:
        builder.add_context(kwargs["context"])

    if "constraints" in kwargs:
        for constraint in kwargs["constraints"]:
            builder.add_constraint(constraint)

    if "instructions" in kwargs:
        for instruction in kwargs["instructions"]:
            builder.add_instruction(instruction)

    return builder.build()


def create_custom_prompt(
    base_instruction: str,
    requirements: Optional[List[str]] = None,
    output_format: str = "json"
) -> str:
    """
    Create a custom vision prompt

    Args:
        base_instruction: Main instruction
        requirements: List of specific requirements
        output_format: Desired output format

    Returns:
        Complete prompt string
    """
    prompt = base_instruction

    if requirements:
        req_text = "\n".join(f"{i+1}. {req}" for i, req in enumerate(requirements))
        prompt += f"\n\nRequirements:\n{req_text}"

    if output_format == "json":
        prompt += "\n\nProvide response in valid JSON format."

    return prompt


def get_educational_prompt(subject_area: str, grade_level: str) -> str:
    """
    Get educational analysis prompt

    Args:
        subject_area: Subject (math, science, history, etc.)
        grade_level: Grade level or age group

    Returns:
        Educational analysis prompt
    """
    return f"""Analyze this image from an educational perspective for {grade_level} students studying {subject_area}.

Identify:
1. Key concepts related to {subject_area}
2. Learning opportunities
3. Discussion points for {grade_level} level
4. Connections to curriculum standards
5. Potential misconceptions to address
6. Extension activities

Format as JSON with structured educational insights."""


def get_accessibility_prompt() -> str:
    """Get prompt for accessibility description"""
    return """Create an accessibility-friendly description of this image for visually impaired users.

Include:
1. Clear, concise overview
2. Important details in logical order
3. Text content (if any)
4. Relevant context
5. Emotional tone or atmosphere

Keep description natural and informative, suitable for screen readers."""
