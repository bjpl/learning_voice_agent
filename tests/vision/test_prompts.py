"""
Tests for Vision Prompts

SPECIFICATION:
- Test prompt templates
- Test prompt builder
- Test task-specific prompts
- Test custom prompt generation

ARCHITECTURE:
- pytest-based tests
- Template validation
- Prompt composition testing

WHY:
- Ensure prompts are well-formed
- Validate prompt builder logic
- Test educational prompt generation
"""
import pytest

from app.vision.prompts import (
    VisionTask,
    VisionPromptTemplates,
    VisionPromptBuilder,
    get_prompt_for_task,
    create_custom_prompt,
    get_educational_prompt,
    get_accessibility_prompt,
)


class TestVisionTask:
    """Test VisionTask enum"""

    def test_vision_task_values(self):
        """Test all vision task types"""
        assert VisionTask.DESCRIBE == "describe"
        assert VisionTask.OCR == "ocr"
        assert VisionTask.DIAGRAM == "diagram"
        assert VisionTask.COMPARE == "compare"
        assert VisionTask.ANALYZE == "analyze"


class TestVisionPromptTemplates:
    """Test prompt templates"""

    def test_describe_image_template(self):
        """Test describe image template"""
        template = VisionPromptTemplates.DESCRIBE_IMAGE

        assert "main subject" in template.lower()
        assert "visual elements" in template.lower()
        assert "json" in template.lower()

    def test_extract_text_template(self):
        """Test OCR template"""
        template = VisionPromptTemplates.EXTRACT_TEXT

        assert "extract" in template.lower()
        assert "text" in template.lower()
        assert "json" in template.lower()

    def test_analyze_diagram_template(self):
        """Test diagram analysis template"""
        template = VisionPromptTemplates.ANALYZE_DIAGRAM

        assert "diagram" in template.lower()
        assert "components" in template.lower()
        assert "connections" in template.lower()

    def test_compare_images_template(self):
        """Test image comparison template"""
        template = VisionPromptTemplates.COMPARE_IMAGES

        assert "compare" in template.lower()
        assert "similarities" in template.lower()
        assert "differences" in template.lower()

    def test_all_templates_have_content(self):
        """Test all templates are non-empty"""
        templates = [
            VisionPromptTemplates.DESCRIBE_IMAGE,
            VisionPromptTemplates.QUICK_DESCRIBE,
            VisionPromptTemplates.EXTRACT_TEXT,
            VisionPromptTemplates.IDENTIFY_TEXT,
            VisionPromptTemplates.ANALYZE_DIAGRAM,
            VisionPromptTemplates.EXPLAIN_DIAGRAM,
            VisionPromptTemplates.COMPARE_IMAGES,
            VisionPromptTemplates.EXTRACT_STRUCTURED_DATA,
            VisionPromptTemplates.CLASSIFY_IMAGE,
            VisionPromptTemplates.DETECT_OBJECTS,
        ]

        for template in templates:
            assert len(template) > 0
            assert isinstance(template, str)


class TestVisionPromptBuilder:
    """Test prompt builder"""

    def test_builder_basic(self):
        """Test basic prompt building"""
        builder = VisionPromptBuilder()
        builder.set_task(VisionTask.DESCRIBE)

        prompt = builder.build()

        assert len(prompt) > 0
        assert isinstance(prompt, str)

    def test_builder_with_context(self):
        """Test builder with context"""
        builder = VisionPromptBuilder()
        builder.set_task(VisionTask.DESCRIBE)
        builder.add_context("Educational material for students")

        prompt = builder.build()

        assert "Educational material" in prompt

    def test_builder_with_constraints(self):
        """Test builder with constraints"""
        builder = VisionPromptBuilder()
        builder.set_task(VisionTask.ANALYZE)
        builder.add_constraint("Focus on educational value")
        builder.add_constraint("Identify learning concepts")

        prompt = builder.build()

        assert "Constraints:" in prompt
        assert "educational value" in prompt
        assert "learning concepts" in prompt

    def test_builder_with_instructions(self):
        """Test builder with additional instructions"""
        builder = VisionPromptBuilder()
        builder.set_task(VisionTask.OCR)
        builder.add_instruction("Preserve formatting")
        builder.add_instruction("Note unclear text")

        prompt = builder.build()

        assert "Additional Requirements:" in prompt
        assert "Preserve formatting" in prompt

    def test_builder_output_format(self):
        """Test output format specification"""
        builder = VisionPromptBuilder()
        builder.set_task(VisionTask.CLASSIFY)
        builder.set_output_format("json")

        prompt = builder.build()

        assert "JSON" in prompt

    def test_builder_chaining(self):
        """Test method chaining"""
        prompt = (
            VisionPromptBuilder()
            .set_task(VisionTask.DIAGRAM)
            .add_context("Flowchart")
            .add_constraint("Identify all nodes")
            .add_instruction("Explain connections")
            .build()
        )

        assert "Flowchart" in prompt
        assert "Identify all nodes" in prompt
        assert "Explain connections" in prompt

    def test_builder_without_task_raises_error(self):
        """Test builder raises error without task"""
        builder = VisionPromptBuilder()

        with pytest.raises(ValueError, match="Task must be set"):
            builder.build()


class TestGetPromptForTask:
    """Test get_prompt_for_task helper"""

    def test_get_describe_prompt(self):
        """Test getting describe prompt"""
        prompt = get_prompt_for_task(VisionTask.DESCRIBE)

        assert len(prompt) > 0
        assert "describe" in prompt.lower() or "main subject" in prompt.lower()

    def test_get_ocr_prompt(self):
        """Test getting OCR prompt"""
        prompt = get_prompt_for_task(VisionTask.OCR)

        assert "extract" in prompt.lower() or "text" in prompt.lower()

    def test_get_prompt_with_context(self):
        """Test getting prompt with context"""
        prompt = get_prompt_for_task(
            VisionTask.ANALYZE,
            context="Biology diagram"
        )

        assert "Biology diagram" in prompt

    def test_get_prompt_with_constraints(self):
        """Test getting prompt with constraints"""
        prompt = get_prompt_for_task(
            VisionTask.CLASSIFY,
            constraints=["Focus on subject matter", "Identify complexity"]
        )

        assert "Focus on subject matter" in prompt
        assert "Identify complexity" in prompt


class TestCreateCustomPrompt:
    """Test custom prompt creation"""

    def test_create_basic_custom_prompt(self):
        """Test creating basic custom prompt"""
        prompt = create_custom_prompt("Analyze this educational image")

        assert "Analyze this educational image" in prompt

    def test_create_custom_prompt_with_requirements(self):
        """Test custom prompt with requirements"""
        prompt = create_custom_prompt(
            "Identify learning concepts",
            requirements=[
                "List main topics",
                "Identify difficulty level",
                "Suggest teaching approach"
            ]
        )

        assert "Identify learning concepts" in prompt
        assert "List main topics" in prompt
        assert "Identify difficulty level" in prompt
        assert "Suggest teaching approach" in prompt

    def test_create_custom_prompt_json_format(self):
        """Test custom prompt with JSON format"""
        prompt = create_custom_prompt(
            "Analyze image",
            output_format="json"
        )

        assert "JSON" in prompt


class TestEducationalPrompt:
    """Test educational prompt generation"""

    def test_educational_prompt_basic(self):
        """Test basic educational prompt"""
        prompt = get_educational_prompt("mathematics", "middle school")

        assert "mathematics" in prompt.lower()
        assert "middle school" in prompt.lower()
        assert "educational" in prompt.lower()

    def test_educational_prompt_different_subjects(self):
        """Test educational prompts for different subjects"""
        subjects = ["science", "history", "literature", "physics"]

        for subject in subjects:
            prompt = get_educational_prompt(subject, "high school")
            assert subject in prompt.lower()

    def test_educational_prompt_structure(self):
        """Test educational prompt has required elements"""
        prompt = get_educational_prompt("biology", "college")

        assert "concepts" in prompt.lower()
        assert "learning" in prompt.lower()
        assert "json" in prompt.lower()


class TestAccessibilityPrompt:
    """Test accessibility prompt generation"""

    def test_accessibility_prompt(self):
        """Test accessibility-friendly prompt"""
        prompt = get_accessibility_prompt()

        assert "accessibility" in prompt.lower()
        assert "description" in prompt.lower()
        assert "visually impaired" in prompt.lower() or "screen reader" in prompt.lower()

    def test_accessibility_prompt_structure(self):
        """Test accessibility prompt structure"""
        prompt = get_accessibility_prompt()

        # Should include key elements for good alt text
        assert "clear" in prompt.lower() or "concise" in prompt.lower()
        assert "context" in prompt.lower()


class TestPromptQuality:
    """Test prompt quality and characteristics"""

    def test_prompts_are_detailed(self):
        """Test that prompts provide sufficient detail"""
        tasks = [
            VisionTask.DESCRIBE,
            VisionTask.OCR,
            VisionTask.DIAGRAM,
            VisionTask.COMPARE,
        ]

        for task in tasks:
            prompt = get_prompt_for_task(task)
            # Prompts should be reasonably detailed
            assert len(prompt) > 100

    def test_prompts_request_json(self):
        """Test that structured prompts request JSON"""
        structured_tasks = [
            VisionTask.DESCRIBE,
            VisionTask.OCR,
            VisionTask.DIAGRAM,
            VisionTask.CLASSIFY,
        ]

        for task in structured_tasks:
            prompt = get_prompt_for_task(task)
            assert "json" in prompt.lower()

    def test_prompts_have_clear_instructions(self):
        """Test that prompts have clear instructions"""
        tasks = [
            VisionTask.DESCRIBE,
            VisionTask.OCR,
            VisionTask.DIAGRAM,
        ]

        for task in tasks:
            prompt = get_prompt_for_task(task)
            # Should contain numbered lists or clear sections
            assert any(marker in prompt for marker in ["1.", "2.", "Include:", "Provide:"])


class TestPromptEdgeCases:
    """Test edge cases in prompt generation"""

    def test_empty_context(self):
        """Test prompt with empty context"""
        prompt = get_prompt_for_task(VisionTask.DESCRIBE, context="")

        # Should still work
        assert len(prompt) > 0

    def test_empty_constraints(self):
        """Test prompt with empty constraints"""
        prompt = get_prompt_for_task(VisionTask.ANALYZE, constraints=[])

        # Should still work
        assert len(prompt) > 0

    def test_custom_prompt_empty_requirements(self):
        """Test custom prompt with empty requirements"""
        prompt = create_custom_prompt("Analyze", requirements=[])

        assert "Analyze" in prompt

    def test_builder_multiple_contexts(self):
        """Test builder behavior with context set multiple times"""
        builder = VisionPromptBuilder()
        builder.set_task(VisionTask.DESCRIBE)
        builder.add_context("First context")
        builder.add_context("Second context")  # This replaces first

        prompt = builder.build()

        # Only last context should be present
        assert "Second context" in prompt
