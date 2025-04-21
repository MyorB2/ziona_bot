import ollama

from business_logic.multilabel_classifier import AntisemitismClassifier


def ollama_chat(model, prompt):
    print(f"User prompt:\n{prompt}\n\n")
    client = ollama.Client()
    response = client.generate(model=model, prompt=prompt)
    print(f"{model} response:")
    print(response.response)


if __name__ == "__main__":
    comment = "why do those penguins always have to not fit to the society, they interfere the marathon runners, they should respect them even if it is on their neighborhood and on sabat"

    # classifier = AntisemitismClassifier()
    # label2 = classifier.predict(comment)
    label = "Foreigner/Alien"

    prompt = (f"I have come across an antithemic comment on social media that says: \n{comment}. "
              f"\nThis response is antisemistic of the type {label}. As a Jewish person, I found  it very offensive, "
              f"\nI would like to combat antisemitism by respond to that person in a way that explains "
              f"what is wrong with the comment and educates about {label} antisemitism. "
              f"\ncan you please suggest at least two references "
              f"such as articles or papers or essay, or Hasbara videos that confronts "
              f"\nthe antisemitism type of {label}? Please summarize each reference. "
              f"\nFurther to this, please help me to phrase a polite comment "
              f"that explains the connection between the comment and {label} antisemitism, "
              f"\nuse the references you suggested about {label} antisemitism and include their website links.")

    model = 'llama3.2'

    print(f"starting a conversation with {model}\n")
    ollama_chat(model, prompt)
