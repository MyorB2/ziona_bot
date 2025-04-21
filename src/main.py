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

    classifier = AntisemitismClassifier()
    label2 = classifier.predict(comment)
    label = "Foreigner/Alien"

    prompt = (f"I have come across an antithemic comment on social media that says: \n{comment}. "
              f"\nThis response is antisemistic of the type {label}. As a Jewish person, I found  it very offensive, "
              f"\nI would like to respond to that person in a way that explains what is wrong with the comment "
              f"and educates about about {label} antisemitism. \ncan you please suggest at least one article "
              f"or paper or essay, or Hasbara video that confronts the antisemitism  type of {label}? "
              f"\nPlease summarize each reference. Further to this, please help me to phrase a comment "
              f"that uses the references you suggested, including links.")

    model = 'llama3.2'

    print(f"starting a conversation with {model}\n")
    ollama_chat(model, prompt)
