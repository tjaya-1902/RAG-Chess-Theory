import requests
import os
from dotenv import load_dotenv

# API Link of the LLM
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

def generate_answer(question, context):
    """ This method sends a prompt to the LLM and returns an answer or an error message """

    prompt = f"""
                Please answer the following chess-related question strictly based on the provided context.

                Context:
                {context}

                If the context does not contain enough information to answer the question, reply exactly with:
                "The databank doesn't have the answer."

                Question: {question}
                Answer:"""

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.2
        }
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)

        # Show the raw response text if something goes wrong
        if response.status_code != 200:
            return f"⚠️ Failed with status {response.status_code}: {response.text}"

        result = response.json()

        # Check if output is a list and contains 'generated_text'
        if isinstance(result, list) and "generated_text" in result[0]:
            # Strip the answer after "Answer:"
            return result[0]["generated_text"].split("Answer:")[-1].strip()
        else:
            return "⚠️ Unexpected response format from Hugging Face API."

    except Exception as e:
        return f"⚠️ Failed to generate answer: {str(e)}"
