import requests
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json


# Initialize the LLM
llm = ChatGroq(
    temperature=0.7,
    groq_api_key='groq_api_key',
    model_name="llama-3.3-70b-versatile"
)

# Supported languages (as per Piston API identifiers)
LANGUAGES = {
    "Python": "python3",
    "JavaScript": "javascript",
    "C": "c",
    "C++": "cpp",
    "Java": "java",
    "Go": "go",
    "Ruby": "ruby",
    "PHP": "php",
    "Rust": "rust"
}

def run_code(language, code, stdin_input=None):
    if language not in LANGUAGES:
        return f"❌ Language '{language}' is not supported."

    payload = {
        "language": LANGUAGES[language],
        "version": "*",
        "files": [{"name": "main", "content": code}]
    }

    # Only include stdin if provided
    if stdin_input is not None:
        payload["stdin"] = stdin_input

    try:
        response = requests.post("https://emkc.org/api/v2/piston/execute", json=payload)
        result = response.json()

        output = result.get("run", {}).get("output")
        if output is not None:
            return output.strip()
        else:
            return "⚠️ No output returned. Check for compilation/runtime errors."
    except Exception as e:
        return f"❌ Error occurred: {str(e)}"


def evaluate_code(problem: str, code: str, stdin_input: None):
    """
    Evaluates the given code against 9 quality parameters using Groq's LLaMA model.
    
    Parameters:
    - problem (str): Problem statement describing the code's purpose.
    - code (str): The source code to be evaluated.
    - stdin_input (str, optional): Example stdin input for the code.
    
    Returns:
    - dict: JSON evaluation containing score & explanation for each parameter.
    """

    # Define the evaluation prompt
    prompt_eval = PromptTemplate.from_template(
        """
        ### PROBLEM STATEMENT:
        {problem}

        ### CODE:
        {code}
        ```

        ### STDIN INPUT (If Any):
        {stdin_input}

        ### INSTRUCTION:
        You are an expert code reviewer and software architect.
        Evaluate the above code based on the following 9 parameters:
        1. Correctness
        2. Efficiency (Time & Space Complexity)
        3. Readability & Code Quality
        4. Scalability
        5. Robustness
        6. Maintainability
        7. Security
        8. Test Coverage
        9. Adherence to Standards

        Provide a detailed score and explanation for each of the 9 points in JSON format as shown:

        ### OUTPUT FORMAT:
        {{
          "Correctness": {{
            "score": 0-10,
            "explanation": "..."
          }},
          "Efficiency": {{
            "score": 0-10,
            "explanation": "..."
          }},
          "Readability & Code Quality": {{
            "score": 0-10,
            "explanation": "..."
          }},
          "Scalability": {{
            "score": 0-10,
            "explanation": "..."
          }},
          "Robustness": {{
            "score": 0-10,
            "explanation": "..."
          }},
          "Maintainability": {{
            "score": 0-10,
            "explanation": "..."
          }},
          "Security": {{
            "score": 0-10,
            "explanation": "..."
          }},
          "Test Coverage": {{
            "score": 0-10,
            "explanation": "..."
          }},
          "Adherence to Standards": {{
            "score": 0-10,
            "explanation": "..."
          }}
        }}
        """
    )

    # Chain the prompt with the model
    chain_eval = prompt_eval | llm

    # Invoke the model
    response = chain_eval.invoke({
        "problem": problem,
        "code": code,
        "stdin_input": stdin_input
    })

    # Parse the response as JSON
    json_parser = JsonOutputParser()
    evaluation = json_parser.parse(response.content)

    return evaluation


def generate_test_cases(problem_statement: str, code: str, language : str, stdin_input= None):
    """
    Generates test cases for a given programming problem using LLM.

    Parameters:
    - problem_statement (str): The problem description.
    - code (str): The code to be tested.
    - stdin_input (str): Example standard input for the code.

    Returns:
    - List[dict]: A list of test cases with 'input' and 'expected_output'.
    """

    # Prompt template
    prompt_testcase = PromptTemplate.from_template(
        """
        ### PROBLEM STATEMENT:
        {problem_statement}

        ### CODE:
        {code}

        ### TASK:
        You are an AI coding assistant. Generate at least 5 diverse test cases in JSON format to test the provided code.
        Each test case should have the following keys:
        - "input": The stdin input value (string)
        - "expected_output": The expected stdout output (string)
        Ensure that the test cases include edge cases and variations.
        Only return a valid JSON array. Do not include explanations.
        """
    )

    # Chain the prompt and LLM
    chain_testcase = prompt_testcase | llm

    # Get the response
    res = chain_testcase.invoke(input={
        'problem_statement': problem_statement,
        'code': code,
        'stdin_input': stdin_input
    })

    # Parse and return the result
    parser = JsonOutputParser()
    testcases = parser.parse(res.content)
    inputs = [item["input"] for item in testcases]
    expected_outputs = [item["expected_output"] for item in testcases]
    results = []  # ✅ store all results here
    for test_input, expected in zip(inputs, expected_outputs):
        output = run_code(language, code, stdin_input=test_input)
        status = "✅ Pass" if output == expected else f"❌ Fail (Got: {output})"
        evaluation = evaluate_code(problem_statement, code, stdin_input)
        results.append(f"Input: {test_input} | Expected: {expected} | {status}")
    return results,inputs,evaluation




# # Example usage
# if __name__ == "__main__":
#     problem = "Check if a number is a palindrome."
#     code = """
# def is_palindrome(number):
#     num_str = str(number)
#     return num_str == num_str[::-1]

# num = int(input("Enter a number: "))
# if is_palindrome(num):
#     print(f"{num} is a palindrome.")
# else:
#     print(f"{num} is not a palindrome.")
# """
#     stdin_input = "121"

    
#     pprint(results)

