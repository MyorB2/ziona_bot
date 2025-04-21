import nltk
from nltk.corpus import stopwords

from src.global_parameters import TYPE_KEYWORDS, EDUCATIONAL_KEYWORDS, REFERENCE_KEYWORDS, IMPOLITE_KEYWORDS, \
    POLITE_KEYWORDS

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)


def evaluate_response(original_comment, label, generated_response):
    """
    Evaluates the quality of a generated response combating antisemitism.

    Parameters:
    - original_comment (str): The original antisemitic comment
    - label (str): The type of antisemitism identified
    - generated_response (str): The LLM-generated response

    Returns:
    - dict: Dictionary containing evaluation scores and feedback
    """
    scores = {}
    feedback = {}

    # Check if type-specific keywords are mentioned
    if label in TYPE_KEYWORDS:
        keywords = TYPE_KEYWORDS[label]
        keywords_found = [kw for kw in keywords if kw.lower() in generated_response.lower()]
        type_score = len(keywords_found) / len(keywords) if keywords else 0
        scores["type_relevance"] = min(type_score * 2, 1.0)

        if scores["type_relevance"] < 0.5:
            feedback[
                "type_relevance"] = f"Low coverage of {label} antisemitism concepts."
        else:
            feedback["type_relevance"] = f"Good coverage of {label} antisemitism concepts."
    else:
        scores["type_relevance"] = 0.0
        feedback["type_relevance"] = f"Unknown antisemitism class: {label}"

    # Check if the response contains educational elements
    edu_count = sum(1 for marker in EDUCATIONAL_KEYWORDS if marker.lower() in generated_response.lower())
    scores["educational_value"] = min(edu_count / 5, 1.0)  # At least 5 markers for full score

    if scores["educational_value"] < 0.4:
        feedback["educational_value"] = "Missing educational content"
    else:
        feedback["educational_value"] = "Good educational content"

    # Check for reference inclusion
    has_references = any(marker.lower() in generated_response.lower() for marker in REFERENCE_KEYWORDS)
    scores["references"] = 1.0 if has_references else 0.0

    if not has_references:
        feedback["references"] = "No inclusion of references."
    else:
        feedback["references"] = "Good inclusion of references."

    # Politeness and tone assessment
    impolite_count = sum(1 for phrase in IMPOLITE_KEYWORDS if phrase.lower() in generated_response.lower())
    polite_count = sum(1 for phrase in POLITE_KEYWORDS if phrase.lower() in generated_response.lower())

    politeness_score = min((polite_count - impolite_count + 1) / 2, 1.0)
    scores["politeness"] = max(politeness_score, 0.0)

    if scores["politeness"] < 0.6:
        feedback["politeness"] = "Bad tone and politeness level."
    else:
        feedback["politeness"] = "Good tone and politeness level."

    # 5. Length appropriateness
    word_count = len(generated_response.split())
    if word_count < 100:
        scores["length"] = 0.5
        feedback["length"] = "Response is too short."
    elif word_count > 500:
        scores["length"] = 0.7
        feedback["length"] = "Response is too long."
    else:
        scores["length"] = 1.0
        feedback["length"] = "Response has appropriate length."

    # 6. Direct addressing of the original comment
    original_words = set(original_comment.lower().split())
    response_words = set(generated_response.lower().split())
    common_words = original_words.intersection(response_words)

    meaningful_common_words = [w for w in common_words if w not in stopwords and len(w) > 3]

    connection_score = min(len(meaningful_common_words) / 3, 1.0)  # At least 3 meaningful words
    scores["connection_to_original"] = connection_score

    if scores["connection_to_original"] < 0.5:
        feedback["connection_to_original"] = "Bad connection to the original comment."
    else:
        feedback["connection_to_original"] = "Good connection to the original comment."

    # Calculate overall score (weighted average)
    weights = {
        "type_relevance": 0.25,
        "educational_value": 0.25,
        "references": 0.15,
        "politeness": 0.15,
        "length": 0.10,
        "connection_to_original": 0.10
    }

    overall_score = sum(scores[metric] * weights[metric] for metric in weights)
    scores["overall"] = overall_score
    feedback["overall"] = f"Response score is {overall_score}."

    return {
        "scores": scores,
        "feedback": feedback
    }
