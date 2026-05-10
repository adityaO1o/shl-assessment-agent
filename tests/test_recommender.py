"""Smoke test for `SHLRecommender`.

Uses a single conversation message, generates a response, and prints the
`reply`, `recommendations`, and `end_of_conversation` fields.
"""

from pathlib import Path
import sys

# Ensure workspace root is importable when the script is run directly.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.recommender import SHLRecommender


def main() -> None:
    conversation = [
        {
            "role": "user",
            "content": "Hiring a senior backend Java engineer with Spring and AWS experience",
        }
    ]

    recommender = SHLRecommender()
    response = recommender.generate_reply(conversation)

    print("Reply:")
    print(response["reply"])
    print()
    print("Recommendations:")
    print(response["recommendations"])
    print()
    print("End of conversation:")
    print(response["end_of_conversation"])

    assert "reply" in response
    assert "recommendations" in response
    assert "end_of_conversation" in response
    print("Success: SHLRecommender returned the expected response structure")


if __name__ == "__main__":
    main()
