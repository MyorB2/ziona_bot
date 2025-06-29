import ast
import json
import logging
from pathlib import Path

import numpy as np
import os
import pandas as pd
from collections import Counter

from business_logic.chatbot.react_agent import ReActAgent
from business_logic.chatbot.react_evaluator import ResponseEvaluator
from business_logic.classification.classification_wrapper import ClassificationModel
from src.utils import normalize_categories

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("faiss").setLevel(logging.WARNING)
logging.getLogger("faiss.loader").setLevel(logging.WARNING)

HF_TOKEN = ""
ROOT_PATH = Path(__file__).parent.parent
RESOURCE_PATH = ROOT_PATH / "resources"
KNOWLEDGE_BASE_PATH = RESOURCE_PATH / "knowledge_base_categorized.csv"

# Suppress Windows symlink warning for Hugging Face cache
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def main():
    """Main function to demonstrate usage"""
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
        logger.info("Reading knowledge base...")
        knowledge_base = pd.read_csv(KNOWLEDGE_BASE_PATH)
        knowledge_base = knowledge_base[['source', 'url', 'paragraph', 'primary_categories']]
        # category_id is list of integers
        knowledge_base["primary_categories"] = knowledge_base["primary_categories"].apply(lambda x: ast.literal_eval(x))
        knowledge_base["primary_categories"] = knowledge_base["primary_categories"].apply(lambda x: normalize_categories(x))
        knowledge_base = knowledge_base.dropna(subset=['primary_categories'])
        logger.info(f"Loaded knowledge base with {len(knowledge_base)} entries")

        # Initialize agent
        agent = ReActAgent(knowledge_base)

        #######################################
        # Initialize classification model
        model_paths = {
            "deberta": "/content/drive/MyDrive/Project/Models/Multi_model/1805_1/final_multilabel_model/saving",
            "hatebert": "Hate-speech-CNERG/dehatebert-mono-english",
            "twitter": "cardiffnlp/twitter-roberta-base",
        }
        classification_model = ClassificationModel(model_paths)

        test_comments = [
            "They always call you antisemitic just because you criticize Israel. It's just a way to silence people.",
            "do all israelis kills babies for fun???",
            "why do those penguins always have to not fit to the society, they interfere the marathon runners, "
            "they should respect them even if it is on their neighborhood and on Sabbath"
        ]
        test_cases = []
        for comment in test_comments:
            pred = classification_model.predict(comment)
            category_id = pred["predicted_labels"][0]
            test_cases.append({"comment": comment, "category_id": category_id, "category_name": label_map[category_id]})

        # # Example usage
        # test_cases = [
        #     {
        #         "comment": "They always call you antisemitic just because you criticize Israel. It's just a way to silence people.",
        #         "category_id": 5,
        #         "category_name": label_map[5]
        #     },
        #     {
        #         "comment": "do all israelis kills babies for fun???",
        #         "category_id": 3,
        #         "category_name": label_map[3]
        #     },
        #     {
        #         "comment": "why do those penguins always have to not fit to the society, they interfere the marathon runners, "
        #                    "they should respect them even if it is on their neighborhood and on Sabbath",
        #         "category_id": 2,
        #         "category_name": label_map[2]
        #     },
        #     {
        #         "comment": "think facebook care? the last name of the ceo is Zuckerberg, is there more greedy person then a jew?",
        #         "category_id": 1,
        #         "category_name": label_map[1]
        #     },
        #     {
        #         "comment": "the jews gave us jesus as our god and saviour they dont worship him and most of our taxes go to them they run the banks on the higher level",
        #         "category_id": 1,
        #         "category_name": label_map[1]
        #     },
        #     {
        #         "comment": "what did you expect??? they claim the Germans genocide(!) them "
        #                    "of course they insist hamas try to eliminate them!",
        #         "category_id": 4,
        #         "category_name": label_map[4]
        #     }
        # ]
        #######################################

        # Save all scores for calculating total evaluation score
        eval_scores = {}

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{'=' * 10}")
            print(f"Test Case {i}")
            print(f"{'=' * 10}")

            result = agent.generate_response(
                test_case["comment"],
                test_case["category_id"],
                test_case["category_name"]
            )

            # Export detailed JSON
            json_result = {
                "Comment": test_case['comment'],
                "Category": test_case['category_name'],
                "Thought 1": result['thought_1'],
                "Action 1": result['action_1'],
                "Observation 1": result['observation_1'],
                "Thought 2": result['thought_2'],
                "Action 2": result['action_2'],
                "Observation 2": result['observation_2'],
                "Thought 3": result['thought_3'],
                "Action 3": result['action_3'],
                "Observation 3": result['observation_3'],
                "Final Response": result['final_response'].split("\n").join(" "),
                "Source": result['source'],
                "URL": result['url']
            }
            json_result = json.dumps(json_result, indent=2, ensure_ascii=False)
            with open(str(RESOURCE_PATH / f"response_{i}.json"), "w", encoding="utf-8") as f:
                f.write(json_result)

            logger.info("Start evaluating...")
            evaluator = ResponseEvaluator()
            evals = evaluator.evaluate_agent_response(test_case["comment"], test_case["category_name"], result)
            print("\nEvaluation Results:")
            for key, value in evals.items():
                print(f"{key}: {value}")
                if key in eval_scores:
                    if isinstance(value, bool):
                        value = int(value)
                    eval_scores[key].append(value)
                else:
                    eval_scores[key] = [value]
            print()

        logger.info(f"Final evaluation results:")
        for key, val in eval_scores.items():
            if isinstance(eval_scores[key], str):
                eval_scores[key] = Counter(val).most_common(1)[0][0]
            else:
                eval_scores[key] = np.mean(val)
            print(f"{key}: {eval_scores[key]}")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")


if __name__ == "__main__":
    main()
