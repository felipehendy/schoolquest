from dotenv import load_dotenv
from openai import OpenAI
import os

# CARREGA O .env
load_dotenv()

# (opcional) debug
# print(os.getenv("OPENAI_API_KEY"))

client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-4.1-nano",
    messages=[
        {"role": "user", "content": "Responda apenas: API funcionando"}
    ]
)

print(resp.choices[0].message.content)
