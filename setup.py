from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()
setup(
    name="azure-genai-utils",
    version="0.0.2.9",
    description="Utility functions for Azure GenAI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    url="https://github.com/daekeun-ml/azure-genai-utils",
    install_requires=[
        "langchain>=0.3.0",
        "langgraph>=0.2.62",
        "audiomentations>=0.38.0",
        "azure-identity",
        "azure-ai-documentintelligence==1.0.0",
        "azure-search-documents==11.5.2",
        "azure-cognitiveservices-speech>=1.42.0",
        "azure-ai-contentsafety==1.0.0",
        "azure-ai-inference==1.0.0b7",
        "tiktoken~=0.8.0",
        "openai>=1.59.7",
        "python-dotenv==1.0.1",
        "pybase64",
    ],
    packages=find_packages(exclude=[]),
    keywords=[
        "azure",
        "genai",
        "azure-genai-utils",
        "azure-genai",
        "langchain",
        "langgraph",
    ],
    python_requires=">=3.8",
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
)
