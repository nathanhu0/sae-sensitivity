"""
Standalone CLI to generate prompt CSVs from activations, replacing gen_prompts.py.

This script provides prompt generation utilities for SAE feature analysis.
"""

from typing import Dict, List, Optional, Any


SYSTEM_PROMPT_TEMPLATE = (
    "You are a meticulous AI researcher conducting an important investigation into a "
    "specific {entity_type} inside a language model that activates in response to text inputs. "
    "Your overall task is to generate additional text samples that cause the {entity_type} to strongly activate.\n\n"
    "You will receive a list of text examples on which the {entity_type} activates. "
    "Specific tokens causing activation will appear between delimiters like {{{{this}}}}. "
    "Consecutive activating tokens will also be accordingly delimited {{{{just like this}}}}. "
    "If no tokens are highlighted with {{}}, then the {entity_type} does not activate on any tokens in the input.\n\n"
    "Note: {entity_type_plural} activate on a word-by-word basis. "
    "Also, {entity_type} activations can only depend on words before the word it activates on."
)


def get_system_prompt(entity_type: str = "feature") -> str:
    """Return the system prompt for the specified entity type."""
    entity_type_plural = entity_type + "s"
    return SYSTEM_PROMPT_TEMPLATE.format(
        entity_type=entity_type, entity_type_plural=entity_type_plural
    )


CORE_PROMPT_HIGHLY_RELEVANT = (
    "Your task is to generate text samples that strongly activate this feature. "
    "Study the examples carefully to identify both their shared and varying traits. "
    "Your generated samples should:\n"
    "- Preserve any consistent traits, patterns, or constraints present across all examples\n"
    "- Match the diversity level shown in the examples — neither more diverse nor more uniform\n"
    "- Vary along the same dimensions that the examples vary (e.g., if examples differ in tone but share a topic, maintain that pattern)\n"
    "- Avoid introducing new types of variation not present in the example set\n"
    "- Avoid collapsing into repetitive or overly similar outputs\n"
)


def build_core_prompt(
    sample_type: str,
    num_per_call: int,
    feature_description: Optional[str] = None,
    separator: Optional[str] = None,
) -> str:
    """Construct the core instruction prompt used for generation."""
    if separator is None:
        separator = "<SAMPLE_SEPARATOR/>"

    # Only the "highly_relevant" case is used in this simplified flow.
    core_prompt = CORE_PROMPT_HIGHLY_RELEVANT

    sample_word_single = "sample"
    sample_word_plural = sample_word_single + "s"
    sample_word = sample_word_single if num_per_call == 1 else sample_word_plural
    # Add +1 to compensate for occasional missing examples from LLMs
    num_examples = num_per_call if num_per_call == 1 else num_per_call + 1

    core_prompt += (
        f" Generate exactly {num_examples} new {sample_word} separated by {separator}. "
        "Note that the feature may involve semantic content, grammatical structures, abstract concepts, "
        "specific named entities (e.g., people, organizations, locations), or formatting elements like newlines, "
        "punctuation, citations, or special characters. If present, these should be preserved as the activating signal."
    )
    if num_per_call > 1:
        core_prompt += (
            f" Present each {sample_word_single} without numbering or bullets. "
            f"Important: place {separator} between generated {sample_word_plural}."
        )
    core_prompt += "."

    if feature_description:
        core_prompt += f" The feature may be described as: {feature_description}."

    return core_prompt


USER_PROMPT_TEMPLATE = (
    "Consider the feature that activates when the given examples below are present. {core_prompt}\n\n"
    "See the following {num_examples} examples that activate the feature, separated by {separator}:\n"
    "{example_str}"
)


def build_user_prompt(core_prompt: str, examples: List[Dict[str, str]], separator: str) -> str:
    """Construct user prompt from core instructions and examples."""
    # Extract just the activation texts
    example_texts = []
    for ex in examples:
        if isinstance(ex, dict):
            # Handle both 'activation_text' and 'Sequence' keys
            text = ex.get("activation_text") or ex.get("Sequence", "")
        else:
            text = str(ex)
        example_texts.append(text)
    
    example_str = f"\n{separator}\n".join(example_texts)
    
    return USER_PROMPT_TEMPLATE.format(
        core_prompt=core_prompt,
        num_examples=len(examples),
        separator=separator,
        example_str=example_str
    )