# Final_Project
Introduction: 
Since the outbreak of the Israel-Gaza War, many civilians joined the Hasbara forces.
Fighting against online antisemitism requires time and human resources.
We developed an Advisory Bot to analyze and respond to antisemitic expressions on social media automatically, using NLP technologies.
Most data is from “Decoding Antisemitism” - an EU project studying antisemitism in Europe. The rest of the data was sourced from Twitter.

Solution:
We have developed a system that classifies comments from across the web and returns a polite, educated, and well-founded response tailored to the type of antisemitism in the comment.

Categories:
0. Not Antisemitic
1. Ideology - Calling for annihilation, conspiracy theories, identifying Jewish people as evil.
2. Stereotypes and Dehumanization - Economic stereotypes, biological racism, humiliating external imaging.
3. Against Israel/ Zionism - Demonize Israel, compare Israel and Zionism to Nazis, anti-Zionism, denial of the right to exist.
4. Holocaust Denial/ Distortion - Be cynical about the Holocaust, Holocaust denial.
5. Indirect Antisemitic - Mention of Jewish or Israeli public figures along with antisemitic connotation, implying Jewish people.

Example:
comment = "they always call you antisemitic just because you criticize Israel. it’s just a way to silence people!"
label = "Foreigner/Alien"

Reasoning towards final response:
Thought: Retrieving factual explanation based on category 5 using BM25 + semantic relevance. | Found contradiction using factual verification. Paragraph: In 2005, the European Monitoring Centre on Racism and Xenophobia(EUMC, now the Fundamental Rights Agency), an agency of the European Union, developed a more detailed working definition, which stated: "Antisemitism is a certain perception of Jews, which may be expressed as hatred toward Jews. Rhetorical and physical manifestations of antisemitism are directed toward Jewish or non-Jewish individuals...
Source: Wikipedia
URL: https://en.wikipedia.org/wiki/Antisemitism![image](https://github.com/user-attachments/assets/bdb07d38-1949-4d80-96d5-8f829e4378cc)

Bot's response:
"Hey, I understand where you're coming from, but I want to clarify something. The statement 'They always call you antisemitic just because you criticize Israel' is misleading. According to the European Monitoring Centre on Racism and Xenophobia (EUMC), which is a well-established organization that tracks hate speech and antisemitism, criticism of Israel is not considered antisemitic. In fact, the EUMC defines antisemitism as 'a certain perception of Jews, which may be expressed as hatred toward Jews.' Criticism of Israel, like any other country, does not necessarily imply hatred towards its Jewish population. This is important because it means that criticism of Israel cannot be used as a way to silence people who have legitimate concerns or questions about the Israeli government's actions. Instead, we should focus on engaging in respectful and informed discussions about these issues.I'd love to hear more about what you're concerned about regarding Israel, and I'm happy to share my perspective if you're open to it."


System Architecture:
![image](https://github.com/user-attachments/assets/9fe348c3-89c4-4e4c-ab9c-f3197fd32e53)

A close-up for the retrieval model:
![image](https://github.com/user-attachments/assets/fcf0b11c-ec79-4d37-8852-a85ba56cc7ed)

