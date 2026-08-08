import os
import time
import logging

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if api_key is None:
    raise ValueError("OPENROUTER_API_KEY not found.")

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)


def call_llm(prompt, temperature=0.2, max_tokens=500):
    """
    Send a prompt to OpenRouter and return the model's response.
    Retries up to 3 times if an error occurs.
    """

    retries = 3

    for attempt in range(retries):

        try:

            response = client.chat.completions.create(
                model="openai/gpt-oss-20b:free",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )

            #return response.choices[0].message.content
            content = response.choices[0].message.content

            print("MODEL CONTENT:", repr(content))

            return content

        except Exception as e:

            logging.warning(
                f"Attempt {attempt+1}/{retries} failed: {e}"
            )

            if attempt < retries - 1:
                time.sleep(5)
            else:
                logging.error(
                    "LLM request failed after 3 attempts."
                )
                return None