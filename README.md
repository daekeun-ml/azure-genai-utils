# Azure GenAI Utils

This repository contains a set of utilities for working with Azure GenAI. The utilities are written in Python and are designed to be used for Hackathons, Workshops, and other events where you need to quickly get started with Azure GenAI.

## Requirements
- Azure Subscription
- Azure AI Foundry
- Bing Search API Key
- Python 3.8 or later
- `.env` file: Please do not forget to modify the `.env` file to match your account. Rename `.env.sample` to `.env` or copy and use it

## Installation
`pip install azure-genai-utils`

## Usage 

### Azure OpenAI Test
```
from azure_genai_utils.aoai_test import AOAI
aoai = AOAI()
aoai.simple_test()
```

### Bing Search

<details markdown="block">
<summary>Expand</summary>

```
from azure_genai_utils.tools import BingSearch
from dotenv import load_dotenv

# You need to add BING_SUBSCRIPTION_KEY=xxxx in .env file
load_dotenv()

# Basic usage
bing = BingSearch(max_results=2, locale="ko-KR")
results = bing.invoke("마이크로소프트 오토젠")
print(results)

## Include news search results and format output
bing = BingSearch(
    max_results=2,
    locale="ko-KR",
    include_news=True,
    include_entity=False,
    format_output=True,
)
results = bing.invoke("마이크로소프트 오토젠")
print(results)
```
</details>

### Synthetic Data Generation

<details markdown="block">
<summary>Expand</summary>

```
from azure_genai_utils.synthetic.qa_generator import (
    QADataGenerator,
    CustomQADataGenerator,
    QAType,
    generate_qas,
)

input_batch = [
    "The quick brown fox jumps over the lazy dog.",
    "What is the capital of France?",
]

model_config = {
    "deployment": "gpt-4o-mini",
    "model": "gpt-4o-mini",
    "max_tokens": 256,
}

try:
    qa_generator = QADataGenerator(model_config=model_config)
    # qa_generator = CustomQADataGenerator(
    #     model_config=model_config, templates_dir=f"./azure_genai_utils/synthetic/prompt_templates/ko"
    # )
    task = generate_qas(
        input_texts=input_batch,
        qa_generator=qa_generator,
        qa_type=QAType.LONG_ANSWER,
        num_questions=2,
        concurrency=3,
    )
except Exception as e:
    print(f"Error generating QAs: {e}")
```
</details>

## License Summary
This sample code is provided under the Apache 2.0 license. See the LICENSE file.