# Final_Project
Development of an Advisory Bot for Digital Advocacy - an intelligent system based on Natural Language Processing (NLP) technologies, designed to detect antisemitic expressions on social media automatically. 
The system provides responses grounded in reliable sources, aiming to support advocacy efforts and strengthen the fight against online antisemitism.

Categories explanation
1: Calling for annihilation, conspiracy theories, identifying Jewish people as evil.
2: Economic stereotypes, biological racism, humiliating external imaging.
3: Demonize Israel, compare Israel and Zionism to Nazis, anti-Zionism, denial of the right to exist.
4: Be cynical about the Holocaust, Holocaust denial.
5: Mention of Jewish or Israeli public figures along with antisemitic connotation, implying Jewish people.
label_map = {1: "antisemitic ideology", 2: "stereotypes and dehumanisation",
             3: "antisemitism against Israel or Zionism", 4: "Holocaust or Zionism denial",
             5: "indirect antisemitism or secondary objective"}

Example:
comment = "they always call you antisemitic just because you criticize Israel. it’s just a way to silence people!"
label = "Foreigner/Alien"

Reasoning towards final response:
Thought: Retrieving factual explanation based on category 5 using BM25 + semantic relevance. | Found contradiction using factual verification. Paragraph: In 2005, the European Monitoring Centre on Racism and Xenophobia(EUMC, now the Fundamental Rights Agency), an agency of the European Union, developed a more detailed working definition, which stated: "Antisemitism is a certain perception of Jews, which may be expressed as hatred toward Jews. Rhetorical and physical manifestations of antisemitism are directed toward Jewish or non-Jewish individuals...
Source: Wikipedia
URL: https://en.wikipedia.org/wiki/Antisemitism![image](https://github.com/user-attachments/assets/bdb07d38-1949-4d80-96d5-8f829e4378cc)

Bot's response:
"Hey, I understand where you're coming from, but I want to clarify something. The statement 'They always call you antisemitic just because you criticize Israel' is misleading. According to the European Monitoring Centre on Racism and Xenophobia (EUMC), which is a well-established organization that tracks hate speech and antisemitism, criticism of Israel is not considered antisemitic. In fact, the EUMC defines antisemitism as 'a certain perception of Jews, which may be expressed as hatred toward Jews.' Criticism of Israel, like any other country, does not necessarily imply hatred towards its Jewish population. This is important because it means that criticism of Israel cannot be used as a way to silence people who have legitimate concerns or questions about the Israeli government's actions. Instead, we should focus on engaging in respectful and informed discussions about these issues.I'd love to hear more about what you're concerned about regarding Israel, and I'm happy to share my perspective if you're open to it."

