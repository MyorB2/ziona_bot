import ast
import logging
import os
import pandas as pd

from business_logic.chatbot.react_agent import ReActAgent
from business_logic.chatbot.react_evaluator import ResponseEvaluator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("faiss").setLevel(logging.WARNING)
logging.getLogger("faiss.loader").setLevel(logging.WARNING)

HF_TOKEN = "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"

# Suppress Windows symlink warning for Hugging Face cache
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def main():
    """Main function to demonstrate usage"""

    # Configuration
    KNOWLEDGE_BASE_PATH = r"C:\Users\myor1\PycharmProjects\ziona_bot\resources\knowledge_base.csv"

    try:
        # Categories explanation
        # 1: Calling for annihilation, theory conspiracies, identify Jewish people as evil.
        # 2: Economic stereotypes, biological racism, humiliating external imaging.
        # 3: Demonize Israel, compare Israel and Zionism to Nazis, anti-Zionism, denial of the right to exist.
        # 4: Be cynical about the Holocaust, Holocaust denial.
        # 5: Mention of Jewish or Israeli public figures along with antisemitic connotation, Implying of Jewish people.
        label_map = {1: "antisemitic ideology", 2: "stereotypes and dehumanisation",
                     3: "antisemitism against Israel or Zionism", 4: "Holocaust or Zionism denial",
                     5: "indirect antisemitism or secondary objective"}

        # Load knowledge base
        knowledge_base = pd.read_csv(KNOWLEDGE_BASE_PATH)
        knowledge_base = knowledge_base[['source', 'url', 'content', 'categories']]
        knowledge_base = knowledge_base.dropna(subset=['source', 'url', 'content'])
        knowledge_base = knowledge_base[knowledge_base['url'].apply(lambda x: x.startswith("http"))]
        knowledge_base.reset_index(drop=True, inplace=True)
        # category_id is list of integers
        knowledge_base["categories"] = knowledge_base["categories"].apply(lambda x: ast.literal_eval(x))
        logger.info(f"Loaded knowledge base with {len(knowledge_base)} entries")

        # Initialize agent
        agent = ReActAgent(knowledge_base)

        # Example usage
        test_cases = [
            {
                "comment": "They always call you antisemitic just because you criticize Israel. It's just a way to silence people.",
                "category_id": 5,
                "category_name": label_map[5]
            },
            {
                "comment": "Jews control all the banks and media",
                "category_id": 2,
                "category_name": label_map[2]
            }
        ]

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{'=' * 50}")
            print(f"Test Case {i}")
            print(f"{'=' * 50}")

            result = agent.generate_response(
                test_case["comment"],
                test_case["category_id"],
                test_case["category_name"]
            )

            print(f"Comment: {test_case['comment']}")
            print(f"Category: {test_case['category_name']}")
            print(f"Thought 1: {result['thought_1']}")
            print(f"Action 1: {result['action_1']}")
            print(f"Observation 1: {result['observation_1']}")
            print(f"Thought 2: {result['thought_2']}")
            print(f"Action 2: {result['action_2']}")
            print(f"Observation 2: {result['observation_2']}")
            print(f"Thought 3: {result['thought_3']}")
            print(f"Action 3: {result['action_3']}")
            print(f"Observation 3: {result['observation_3']}")
            print(f"Final Response: {result['final_response']}")
            print(f"Source: {result['source']}")
            print(f"URL: {result['url']}")

            evaluator = ResponseEvaluator()
            results = evaluator.evaluate_agent_response(test_case["comment"], test_case["category_name"], result['final_response'])
            print("\nEvaluation Results:")
            for key, value in results.items():
                print(f"{key}: {value}")
            print()

    except Exception as e:
        logger.error(f"Error in main execution: {e}")


if __name__ == "__main__":
    main()
