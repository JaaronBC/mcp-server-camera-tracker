# Handles test behavior.
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv() 
openai_client = OpenAI(
    base_url="https://128.171.10.83:9443/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY")
)


print("Testing model connection...")

response = openai_client.chat.completions.create(
    model="openai/qwen-3.8-27b",
    messages=[
        {
            "role": "user",
            "content": "Reply with: Model test successful"
        }
    ],
)

print(response.choices[0].message.content)
