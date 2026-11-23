import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(base_url=os.getenv("LLM_BASE_URL"), api_key=os.environ["LLM_API_KEY"])
r = client.chat.completions.create(
    model=os.getenv("MODEL_CHEAP"),
    messages=[{"role": "user", "content": "Say hello in 5 words."}],
)
print(r.choices[0].message.content)
print(r.usage)