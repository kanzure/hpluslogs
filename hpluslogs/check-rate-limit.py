import os
import requests
import json

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise RuntimeError("Environment variable OPENROUTER_API_KEY is not set.")

response = requests.get(
    url="https://openrouter.ai/api/v1/key",
    headers={
        "Authorization": f"Bearer {OPENROUTER_API_KEY}"
    }
)

print(json.dumps(response.json(), indent=2))

# This script is kind of useless because apparently you can hammer the API for
# embeddings and it doesn't really matter. So never mind, I guess.
