import os
from azure.ai.inference.models import SystemMessage
from azure.ai.inference.models import UserMessage

NUM_SAMPLES = 2

topic = f"""
Contoso Electronics call center QnA related expected spoken utterances for {CUSTOM_SPEECH_LANG} and English languages.
"""
question = f"""
create {NUM_SAMPLES} lines of jsonl of the topic in {CUSTOM_SPEECH_LANG} and english. jsonl format is required. use 'no' as number and '{CUSTOM_SPEECH_LOCALE}', 'en-US' keys for the languages.
only include the lines as the result. Do not include ```jsonl, ``` and blank line in the result. 
"""

system_message = """
Generate plain text sentences of #topic# related text to improve the recognition of domain-specific words and phrases.
Domain-specific words can be uncommon or made-up words, but their pronunciation must be straightforward to be recognized. 
Use text data that's close to the expected spoken utterances. The nummber of utterances per line should be 1. 
Here is examples of the expected format:
{"no": 1, "string": "string", "string": "string"}
{"no": 2, "string": "string", "string": "string"}
"""

user_message = f"""
#topic#: {topic}
Question: {question}
"""

# Simple API Call
response = client.chat.completions.create(
    model=aoai_deployment_name,
    messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ],
    temperature=0.8,
    max_tokens=1024,
    top_p=0.1
)

content = response.choices[0].message.content
print(content)
print("Usage Information:")
#print(f"Cached Tokens: {response.usage.prompt_tokens_details.cached_tokens}") #only o1 models support this
print(f"Completion Tokens: {response.usage.completion_tokens}")
print(f"Prompt Tokens: {response.usage.prompt_tokens}")
print(f"Total Tokens: {response.usage.total_tokens}")