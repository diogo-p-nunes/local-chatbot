from openai import OpenAI

# Replace with your OpenAI API key
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

model = "Qwen/Qwen3-0.6B"

# The prompt for the VLM (Vision-Language Model)
prompt = "Hello! Who are you and who owns you?"

response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }
    ],
    max_tokens=500,
)

print("Response:", response.choices[0].message.content)