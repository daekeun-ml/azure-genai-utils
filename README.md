# Azure GenAI Utils

This repository contains a set of utilities for working with Azure GenAI. The utilities are written in Python and are designed to be used for Hackathons, Workshops, and other events where you need to quickly get started with Azure GenAI.

## Requirements
- Azure Subscription
- Azure AI Foundry
- Bing Search API Key
- Python 3.8 or later
- `.env` file: Please do not forget to modify the `.env` file to match your account. Rename `.env.sample` to `.env` or copy and use it
    ```bash
    AZURE_OPENAI_ENDPOINT=xxxxx
    AZURE_OPENAI_API_KEY=xxxxx
    OPENAI_API_VERSION=2024-12-01-preview
    AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini

    # Optinoal, but required for LangChain
    LANGCHAIN_TRACING_V2=false
    LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
    LANGCHAIN_API_KEY=xxxxx
    LANGCHAIN_PROJECT="YOUR-PROJECT"
    ```

## Installation

### PyPI
- `pip install azure-genai-utils`

### From Source
- `python setup.py install`

## Usage 

### Azure OpenAI Test
<details markdown="block">
<summary>Expand</summary>

```python
from azure_genai_utils.aoai import AOAI
aoai = AOAI()
aoai.test_api_call()
```
</details>

### PDF RAG Chain
<details markdown="block">
<summary>Expand</summary>

```python
from azure_genai_utils.rag.pdf import PDFRetrievalChain

pdf_path = "[YOUR-PDF-PATH]"

pdf = PDFRetrievalChain(
    source_uri=[pdf_path],
    loader_type="PDFPlumber",
    model_name="gpt-4o-mini",
    embedding_name="text-embedding-3-large",
    chunk_size=500,
    chunk_overlap=50,
).create_chain()

question = "[YOUR-QUESTION]"
docs = pdf.retriever.invoke(question)
results = pdf.chain.invoke({"chat_history": "", "question": question, "context": docs})
```
</details>

### Bing Search
Please make sure to set the following environment variables in your `.env` file:
```bash
BING_SUBSCRIPTION_KEY=xxxxx
```

<details markdown="block">
<summary>Expand</summary>

```python
from azure_genai_utils.tools import BingSearch
from dotenv import load_dotenv

# You need to add BING_SUBSCRIPTION_KEY=xxxx in .env file
load_dotenv()

# Basic usage
bing = BingSearch(max_results=2, locale="ko-KR")
results = bing.invoke("Microsoft AutoGen")
print(results)

## Include news search results and format output
bing = BingSearch(
    max_results=2,
    locale="ko-KR",
    include_news=True,
    include_entity=False,
    format_output=True,
)
results = bing.invoke("Microsoft AutoGen")
print(results)
```
</details>

### LangGraph Example (Bing Search + Azure GenAI)
<details markdown="block">
<summary>Expand</summary>

```python
import json
from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import ToolMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, END
from azure_genai_utils.tools import BingSearch
from dotenv import load_dotenv

load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = AzureChatOpenAI(model="gpt-4o-mini")
tool = BingSearch(max_results=3, format_output=False)
tools = [tool]
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    answer = llm_with_tools.invoke(state["messages"])
    return {"messages": [answer]}

def route_tools(
    state: State,
):
    if messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"

    return END

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    source="chatbot",
    path=route_tools,
    path_map={"tools": "tools", END: END},
)

graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()

# Test
inputs = {"messages": "Microsoft AutoGen"}

for event in graph.stream(inputs, stream_mode="values"):
    for key, value in event.items():
        print(f"\n==============\nSTEP: {key}\n==============\n")
        print(value[-1])
```
</details>

### Synthetic Data Generation
<details markdown="block">
<summary>Expand</summary>

```python
from azure_genai_utils.synthetic import (
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

### Azure Custom Speech
Please make sure to set the following environment variables in your `.env` file:
```bash
AZURE_AI_SPEECH_REGION=xxxxx
AZURE_AI_SPEECH_API_KEY=xxxxx
```
<details markdown="block">
<summary>Expand</summary>

```python
from azure_genai_utils.stt.stt_generator import CustomSpeechToTextGenerator

# Initialize the CustomSpeechToTextGenerator
stt = CustomSpeechToTextGenerator(
    custom_speech_lang="Korean",
    synthetic_text_file="cc_support_expressions.jsonl",
    train_output_dir="synthetic_data_train",
    train_output_dir_aug="synthetic_data_train_aug",
    eval_output_dir="synthetic_data_eval",
)

### Training set
# Generate synthetic text
topic = "Call center QnA related expected spoken utterances"
content = stt.generate_synthetic_text(
    topic=topic, num_samples=2, model_name="gpt-4o-mini"
)
stt.save_synthetic_text(output_dir="plain_text")

# Generate synthetic wav files for training
train_tts_voice_list = [
    "ko-KR-InJoonNeural",
    "zh-CN-XiaoxiaoMultilingualNeural",
    "en-GB-AdaMultilingualNeural",
]
stt.generate_synthetic_wav(
    mode="train", tts_voice_list=train_tts_voice_list, delete_old_data=True
)

# Augment the train data (Optional)
stt.augment_wav_files(num_augments=4)
# Package the train data to be used in the training pipeline
stt.package_trainset(use_augmented_data=True)

### Evaluation set
# Generate synthetic wav files for evaluation
eval_tts_voice_list = ["ko-KR-YuJinNeural"]
stt.generate_synthetic_wav(
    mode="eval", tts_voice_list=eval_tts_voice_list, delete_old_data=True
)
# Package the eval data to be used in the evaluation pipeline
stt.package_evalset(eval_dataset_dir="eval_dataset")
```
</details>

## License Summary
This sample code is provided under the Apache 2.0 license. See the LICENSE file.