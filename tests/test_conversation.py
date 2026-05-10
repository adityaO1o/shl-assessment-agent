"""Tests for the ConversationAnalyzer.

Creates several example user messages and prints the structured analysis
result for each case: vague, comparison, refinement, and legal/off-topic.
"""
from pathlib import Path
import sys

# Ensure workspace root is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.conversation import ConversationAnalyzer


def print_result(title: str, result) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    # Use Pydantic model's JSON for readable structured output
    # Use `model_dump_json` for Pydantic v2 compatibility
    try:
        print(result.model_dump_json(indent=2))
    except Exception:
        # Fallback for older Pydantic versions
        print(result.json())


def main() -> None:
    analyzer = ConversationAnalyzer(verbose=False)

    # 1) Vague query
    vague_msgs = [{"role": "user", "content": "We need assessments"}]
    res_vague = analyzer.analyze(vague_msgs)
    print_result("Vague query: 'We need assessments'", res_vague)

    # 2) Comparison
    comp_msgs = [{"role": "user", "content": "What is the difference between OPQ and DSI?"}]
    res_comp = analyzer.analyze(comp_msgs)
    print_result("Comparison query: 'What is the difference between OPQ and DSI?'", res_comp)

    # 3) Refinement
    ref_msgs = [{"role": "user", "content": "Add personality tests"}]
    res_ref = analyzer.analyze(ref_msgs)
    print_result("Refinement query: 'Add personality tests'", res_ref)

    # 4) Legal / off-topic
    legal_msgs = [{"role": "user", "content": "Does this satisfy HIPAA legal requirements?"}]
    res_legal = analyzer.analyze(legal_msgs)
    print_result("Legal/off-topic query: 'Does this satisfy HIPAA legal requirements?'", res_legal)


if __name__ == "__main__":
    main()
