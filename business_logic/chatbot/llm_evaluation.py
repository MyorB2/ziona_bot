import ollama
import json


class LLMEvaluator:
    """
    Class to evaluate responses to antisemitic comments using another LLM as a judge.
    """

    def __init__(self, evaluator_model="llama3.2"):
        """
        Initialize the evaluator with a specific model.

        Parameters:
        - evaluator_model (str): The model to use for evaluation (default: llama3.2)
        """
        self.evaluator_model = evaluator_model
        self.client = ollama.Client()

    def evaluate_response(self, original_comment, antisemitism_type, generated_response):
        """
        Uses an LLM to evaluate the quality of a generated response combating antisemitism.

        Parameters:
        - original_comment (str): The original antisemitic comment
        - antisemitism_type (str): The type of antisemitism identified
        - generated_response (str): The LLM-generated response

        Returns:
        - dict: Dictionary containing evaluation scores and feedback
        """

        # Create an evaluation prompt that asks the LLM to analyze the response
        evaluation_prompt = f"""
        You are an expert evaluator specialized in assessing responses to antisemitic content. You will be given:
        1. An original antisemitic comment
        2. The identified type of antisemitism
        3. A generated response meant to educate and combat this antisemitism
        
        Your task is to evaluate the response on the following criteria, rating each from 0.0 to 1.0:
        
        1. TYPE_RELEVANCE: How well does the response address the specific type of antisemitism ({antisemitism_type})?
        2. EDUCATIONAL_VALUE: How informative and educational is the response about antisemitism?
        3. REFERENCES: Does the response include quality references and resources?
        4. TONE: Is the response polite, respectful, and likely to create a constructive dialogue?
        5. EFFECTIVENESS: How likely is this response to change the person's perspective or educate others reading it?
        6. CONNECTION: How well does the response directly address the specific content/wording of the original comment?
        
        For each criterion, provide:
        - A score between 0.0 and 1.0
        - A brief explanation for your rating
        - Specific suggestions for improvement
        
        ORIGINAL COMMENT:
        {original_comment}
        
        ANTISEMITISM TYPE:
        {antisemitism_type}
        
        GENERATED RESPONSE:
        {generated_response}
        
        Return your evaluation in valid JSON format with this structure:
        {{
          "scores": {{
            "type_relevance": [score],
            "educational_value": [score],
            "references": [score],
            "tone": [score],
            "effectiveness": [score],
            "connection": [score],
            "overall": [average_score]
          }},
          "feedback": {{
            "type_relevance": [feedback],
            "educational_value": [feedback],
            "references": [feedback],
            "tone": [feedback],
            "effectiveness": [feedback],
            "connection": [feedback],
            "overall": [overall_feedback]
          }},
          "improvement_suggestions": [list of specific improvements the response needs]
        }}
        
        Be objective and thorough in your assessment. Base your evaluation on evidence from the text.
        """

        # Get the evaluation from the LLM
        print(f"Evaluating response using {self.evaluator_model}...")
        response = self.client.generate(model=self.evaluator_model, prompt=evaluation_prompt)
        evaluation_text = response.response

        # Extract the JSON part from the response
        try:
            # Try to find JSON within the response
            json_start = evaluation_text.find('{')
            json_end = evaluation_text.rfind('}') + 1
            if json_start >= 0 and json_end > 0:
                json_str = evaluation_text[json_start:json_end]
                evaluation_results = json.loads(json_str)
            else:
                # If no JSON formatting found, use the whole text
                evaluation_results = json.loads(evaluation_text)

            return evaluation_results

        except json.JSONDecodeError:
            # If JSON parsing fails, return an error message and the raw response
            print("Failed to parse LLM response as JSON. Raw response:")
            print(evaluation_text)
            return {
                "scores": {"overall": 0.0},
                "feedback": {
                    "overall": "Error parsing evaluation. Please check the LLM's response format."
                },
                "raw_response": evaluation_text
            }

    def batch_evaluate(self, test_cases):
        """
        Evaluate multiple test cases and return aggregate results.

        Parameters:
        - test_cases: List of dictionaries containing:
            - original_comment: The original antisemitic comment
            - antisemitism_type: The type of antisemitism identified
            - generated_response: The LLM-generated response

        Returns:
        - dict: Aggregate evaluation results and individual case results
        """
        results = []

        for i, case in enumerate(test_cases):
            print(f"Evaluating case {i + 1}/{len(test_cases)}...")
            result = self.evaluate_response(
                case["original_comment"],
                case["antisemitism_type"],
                case["generated_response"]
            )
            results.append(result)

        # Calculate average scores across all test cases
        avg_scores = {}
        if results and "scores" in results[0]:
            metrics = results[0]["scores"].keys()
            for metric in metrics:
                scores = [r["scores"].get(metric, 0) for r in results if "scores" in r]
                avg_scores[metric] = sum(scores) / len(scores) if scores else 0

        return {
            "individual_results": results,
            "average_scores": avg_scores
        }
