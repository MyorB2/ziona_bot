import ollama
from duckduckgo_search import DDGS


def ollama_chat(model, prompt):
    client = ollama.Client()
    response = client.generate(model=model, prompt=prompt)
    print("Response from Ollama:")
    print(response.response)


if __name__ == "__main__":
    comment = "why do those penguins always have to not fit to the society, they interfere the marathon runners, they should respect them even if it is on their neighborhood and on sabat."
    label = "Foreigner/Alien"
    prompt = (f"I have came across an antithemic comment on social media that says: {comment}. "
              f"This response is antisemistic of the type {label}. As a jewish person i found it very offensive, "
              f"I would like to response that person in a way that explains what is wrong with the comment "
              f"and educate about {label} antisemitism. can you please suggest at list one article "
              f"or paper or assay or Hasbara video that confront the antisimitism type of {label}? "
              f"please summarize each reference. Further to this, please help me to phrase a comment "
              f"that uses the references you suggested")
    model = 'llama3.2'

    print("start a conversation")
    ollama_chat(model, prompt)
