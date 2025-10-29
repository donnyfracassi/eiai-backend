import os
from openai import OpenAI

# Create the client using your environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# A simple test prompt
prompt = "Write a short motivational quote about learning AI."

# Send the request to OpenAI
response = client.chat.completions.create(
    model="gpt-4o-mini",  # or "gpt-4o" if you want the full version
    messages=[
        {"role": "user", "content": prompt}
    ]
)

# Print the AI's reply
print(response.choices[0].message.content)
