import os


def get_langchain_api_key():
    """
    Get LangChain API Key from environment variable.
    """
    langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
    has_key = langchain_api_key is not None and langchain_api_key.strip() != ""
    return langchain_api_key, has_key


def set_langsmith(project_name, tracing=True):
    """
    Set LangSmith tracing and project name.
    """
    if tracing:
        langchain_key, has_key = get_langchain_api_key()
        if not has_key:
            print(
                "You have not set the LangChain API Key. Please set it first if you want to use LangSmith."
            )
        else:
            os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
            os.environ["LANGCHAIN_TRACING_V2"] = "true"

            if project_name is not None or project_name.strip() != "":
                os.environ["LANGCHAIN_PROJECT"] = project_name
                print(f"[LangSmith Project] {project_name}")
            else:
                print("Please set LangSmith Project name.")
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        print("Does not use LangSmith tracing.")
