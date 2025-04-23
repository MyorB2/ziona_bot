import ollama

from business_logic.llm_evaluation import evaluate_response, LLMEvaluator
from business_logic.multilabel_classifier import AntisemitismClassifier


def ollama_chat(model, prompt):
    print(f"User prompt:\n{prompt}\n\n")
    client = ollama.Client()
    response = client.generate(model=model, prompt=prompt)
    print(f"{model} response:")
    print(response.response)
    return response


if __name__ == "__main__":
    comment = "why do those penguins always have to not fit to the society, they interfere the marathon runners, they should respect them even if it is on their neighborhood and on sabat"

    # classifier = AntisemitismClassifier()
    # label2 = classifier.predict(comment)
    label = "Foreigner/Alien"

    prompt = (f"I have come across an antithemic comment on social media that says: "
              f"\n{comment}. "
              f"\nThis response is antisemistic of the type {label}. As a Jewish person, I found  it very offensive, "
              f"\nI would like to combat antisemitism by respond to that person in a way that explains "
              f"what is wrong with the comment and educates about {label} antisemitism. "
              f"\ncan you please suggest at least two references "
              f"such as articles or papers or essay, or Hasbara videos that confronts "
              f"\nthe antisemitism type of {label}? Please summarize each reference. "
              f"\nFurther to this, please help me to phrase a polite comment "
              f"that explains the connection between the comment and {label} antisemitism, "
              f"\nin the response, include the same references you suggested about {label} antisemitism "
              f"along with their website links if there are.")

    model = 'llama3.2'

    print(f"starting a conversation with {model}\n")
    response = ollama_chat(model, prompt)

    # Evaluate a single response
    evaluator = LLMEvaluator(evaluator_model="llama3.2")
    evaluation_results = evaluator.evaluate_response(comment, label, response.response)

    print("\n---- LLM Evaluation Results ----")
    if "scores" in evaluation_results:
        print("\nScores:")
        for metric, score in evaluation_results["scores"].items():
            print(f"  - {metric}: {score:.2f}")

