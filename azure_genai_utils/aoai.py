import os
from openai import AzureOpenAI
from dotenv import load_dotenv
from typing import Optional


class AOAI:
    """
    Helper class for Azure OpenAI
    """

    aoai_api_endpoint: Optional[str] = None
    aoai_api_key: Optional[str] = None
    aoai_api_version: Optional[str] = None
    aoai_deployment_name: Optional[str] = None
    client: AzureOpenAI

    def __init__(
        self,
        aoai_api_endpoint: Optional[str] = None,
        aoai_api_key: Optional[str] = None,
        aoai_api_version: Optional[str] = None,
        aoai_deployment_name: Optional[str] = None,
    ):
        """
        Initialize Azure OpenAI client. You can set the environment variables in .env file.
        If you do not have .env file, you can pass the values as arguments.
        Args:
            aoai_api_endpoint: Azure OpenAI endpoint
            aoai_api_key: Azure OpenAI API key
            aoai_api_version: Azure OpenAI API version
            aoai_deployment_name: Azure OpenAI deployment name
        """

        if aoai_api_endpoint is None:
            aoai_api_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

        if aoai_api_key is None:
            aoai_api_key = os.getenv("AZURE_OPENAI_API_KEY")

        if aoai_api_version is None:
            aoai_api_version = os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv(
                "OPENAI_API_VERSION"
            )

        if aoai_deployment_name is None:
            aoai_deployment_name = os.getenv(
                "AZURE_OPENAI_DEPLOYMENT_NAME"
            ) or os.getenv("OPENAI_DEPLOYMENT_NAME")

        self.aoai_api_endpoint = aoai_api_endpoint
        self.aoai_api_key = aoai_api_key
        self.aoai_api_version = aoai_api_version
        self.aoai_deployment_name = aoai_deployment_name

        try:
            if self.aoai_api_endpoint is None:
                raise ValueError("Azure OpenAI endpoint cannot be None")

            self.client = AzureOpenAI(
                azure_endpoint=self.aoai_api_endpoint,
                api_key=self.aoai_api_key,
                api_version=self.aoai_api_version,
            )
            print("=== Initialized AzureOpenAI client ===")
            print(f"AZURE_OPENAI_ENDPOINT={aoai_api_endpoint}")
            print(f"AZURE_OPENAI_API_VERSION={aoai_api_version}")
            print(f"AZURE_OPENAI_DEPLOYMENT_NAME={aoai_deployment_name}")

        except (ValueError, TypeError) as e:
            print("=== Failed to initialize AzureOpenAI client ===")
            print(e)

    def set_deployment_name(self, deployment_name: str):
        """
        Set the deployment name for Azure OpenAI
        Args:
            deployment_name: Azure OpenAI deployment name
        """
        self.aoai_deployment_name = deployment_name

    def get_aoai_client(self):
        return self.client

    def test_api_call(self):
        """
        Simple API Call
        """
        system_message = """
        You are an AI assistant that helps customers find information. As an assistant, you respond to questions in a concise and unique manner.
        You can use Markdown to answer simply and concisely, and add a personal touch with appropriate emojis.

        Add a witty joke starting with "By the way," at the end of your response. Do not mention the customer's name in the joke part.
        The joke should be related to the specific question asked.
        For example, if the question is about tents, the joke should be specifically related to tents.

        Use the given context to provide a more personalized response. Write each sentence on a new line:
        """
        context = """
            The Alpine Explorer Tent features a detachable partition to ensure privacy, 
            numerous mesh windows and adjustable vents for ventilation, and a waterproof design. 
            It also includes a built-in gear loft for storing outdoor essentials. 
            In short, it offers a harmonious blend of privacy, comfort, and convenience, making it a second home in nature!
        """
        question = "What are features of the Alpine Explorer Tent?"

        user_message = f"""
        Context: {context}
        Question: {question}
        """

        if self.aoai_deployment_name is None:
            raise ValueError("Deployment name cannot be None")

        response = self.client.chat.completions.create(
            model=self.aoai_deployment_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
            max_tokens=300,
        )

        print(response.choices[0].message.content)
