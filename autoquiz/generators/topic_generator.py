from operator import itemgetter
from typing import Iterable
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables.base import Runnable
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.prompts import ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from autoquiz.prompts import TOPIC_GENERATION_PROMPT, DocumentTopics


class TopicGenerator:
    _documents: Iterable[Document] = None
    prompt: ChatPromptTemplate = TOPIC_GENERATION_PROMPT
    llm = None

    def __init__(
        self,
        llm: BaseLanguageModel,
        documents: Iterable[Document],
        prompt: ChatPromptTemplate = TOPIC_GENERATION_PROMPT,
    ):
        """
        Initializes the class.

        Args:
            llm: The language model to use.
            documents: An iterable of documents.
            prompt: The prompt to use (default: TOPIC_GENERATION_PROMPT).

        Raises:
            ValueError: If the documents list is empty.
        """
        self.llm = llm

        if len(documents) == 0:
            raise ValueError("The documents list must not be empty.")

        self._documents = documents
        self.prompt = prompt

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Document],
        llm: BaseLanguageModel,
        chunk_size: int = 1000,
        chunk_overlap: int = 0,
        prompt: ChatPromptTemplate = TOPIC_GENERATION_PROMPT,
    ):
        """
        Create an instance of the class from a collection of documents.

        Args:
            documents: An iterable of documents.
            llm: The language model to use.
            chunk_size: The size of the chunks (default: 1000).
            chunk_overlap: The overlap between chunks (default: 0).
            prompt: The prompt to use (default: TOPIC_GENERATION_PROMPT).

        Returns:
            An instance of the class.

        Raises:
            ValueError: If the documents list is empty.
        """

        splitter = CharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        docs = splitter.split_documents(documents)

        return cls(
            llm=llm,
            documents=docs,
            prompt=prompt,
        )

    @property
    def documents(self) -> Iterable[Document]:
        """
        Returns the documents.
        """
        return self._documents

    def generate(self):

        results = self._chain().batch([{"input": doc} for doc in self._documents])
        # print("Documents", self._documents)
        # print(results)
        for index, doc in enumerate(self._documents):
            doc.metadata["topics"] = results[index]["topics"]

        return self.documents

    def _chain(self) -> Runnable:

        tools = [DocumentTopics]
        llm_with_tools = self.llm.bind_tools(tools)
        parser = PydanticToolsParser(tools=tools)

        doc_topics_chain = self.prompt | llm_with_tools | parser

        return (
            RunnablePassthrough.assign(intermediate_steps={})
            | RunnablePassthrough.assign(
                agent_scratchpad=lambda x: format_to_openai_tool_messages(
                    x["intermediate_steps"]
                )
            )
            | RunnablePassthrough.assign(document_topics=doc_topics_chain)
            | RunnableParallel(
                context=itemgetter("input"),
                topics=lambda x: x["document_topics"][0].topics,
            )
        )
