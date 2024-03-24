import os
from langchain_openai.chat_models.azure import AzureChatOpenAI


def azure_chat_openai(**kwargs):

    return AzureChatOpenAI(
        deployment_name=os.environ["AZURE_DEPLOYMENT_NAME"], **kwargs
    )
