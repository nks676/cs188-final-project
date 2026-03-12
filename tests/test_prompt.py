"""Unit tests for taskb/prompt.py."""

from taskb import prompt


class TestPromptConstants:
    def test_system_message_requires_scene_lookup(self):
        assert "MUST call" in prompt.SYSTEM_MESSAGE
        assert "get_scene_state()" in prompt.SYSTEM_MESSAGE

    def test_api_reference_lists_core_functions(self):
        for name in (
            "get_scene_state",
            "get_workspace_bounds",
            "pick_and_place",
            "get_corner_pos",
            "make_circle_positions",
            "say",
        ):
            assert name in prompt.API_REFERENCE

    def test_few_shot_examples_cover_all_categories(self):
        examples = prompt._FEW_SHOT_EXAMPLES
        assert len(examples) == 12
        instructions = [instruction for instruction, _ in examples]
        expected = (
            "Put the red block in the top right corner.",
            "Stack the green block on the red block.",
            "Line up all the blocks along the right side.",
            "If the red block is to the left of the blue block, swap them.",
            "Put all the large blocks on the left side, then stack the small blocks on the right.",
            "Cut the block in half.",
        )
        for item in expected:
            assert item in instructions


class TestBuildPrompt:
    def test_build_prompt_contains_sections(self):
        built = prompt.build_prompt("Move the red block to the left side.")
        assert prompt.SYSTEM_MESSAGE in built
        assert prompt.API_REFERENCE in built
        assert "# === FEW-SHOT EXAMPLES ===" in built

    def test_build_prompt_appends_final_instruction(self):
        instruction = "Move the red block to the left side."
        built = prompt.build_prompt(instruction)
        assert built.endswith(f"# INSTRUCTION: {instruction}\n")

    def test_build_prompt_wraps_examples_as_python_blocks(self):
        built = prompt.build_prompt("Move the blue block to the corner.")
        assert built.count("```python") == len(prompt._FEW_SHOT_EXAMPLES)
        assert built.count("# INSTRUCTION: ") == len(prompt._FEW_SHOT_EXAMPLES) + 1
