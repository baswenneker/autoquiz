import pytest
from autoquiz.generators.topic_generator import TopicGenerator
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from autoquiz.utils import azure_chat_openai


@pytest.fixture(scope="session", name="documents")
def fixture_documents():
    return [
        Document(
            page_content="The lion inhabits grasslands, savannahs, and shrublands. It is usually more diurnal than other wild cats, but when persecuted, it adapts to being active at night and at twilight. During the Neolithic period, the lion ranged throughout Africa and Eurasia, from Southeast Europe to India, but it has been reduced to fragmented populations in sub-Saharan Africa and one population in western India. It has been listed as Vulnerable on the IUCN Red List since 1996 because populations in African countries have declined by about 43% since the early 1990s. Lion populations are untenable outside designated protected areas. Although the cause of the decline is not fully understood, habitat loss and conflicts with humans are the greatest causes for concern."
        ),
        Document(
            page_content="Lion samples from some parts of the Ethiopian Highlands cluster genetically with those from Cameroon and Chad, while lions from other areas of Ethiopia cluster with samples from East Africa. Researchers, therefore, assume Ethiopia is a contact zone between the two subspecies.[18] Genome-wide data of a wild-born historical lion sample from Sudan showed that it clustered with P. l. leo in mtDNA-based phylogenies, but with a high affinity to P. l. melanochaita. This result suggested that the taxonomic position of lions in Central Africa may require revision."
        ),
    ]


@pytest.fixture(scope="session", name="topic_generator")
def create_topic_generator(documents):

    llm = azure_chat_openai()

    return TopicGenerator.from_documents(llm=llm, documents=documents)


@pytest.fixture(scope="session", name="documents_with_topics")
def fixture_documents_with_topics(topic_generator):
    return topic_generator.generate()


def test_fixture_documents(documents):
    assert isinstance(documents, list)
    assert all(isinstance(doc, Document) for doc in documents)


def test_from_documents(topic_generator):
    assert isinstance(topic_generator, TopicGenerator)


def test_generation(documents_with_topics):
    assert documents_with_topics is not None
    assert isinstance(documents_with_topics, list)
    assert all(isinstance(doc, Document) for doc in documents_with_topics)


def test_metadata_topics(documents_with_topics):
    assert all("topics" in doc.metadata for doc in documents_with_topics)


def test_metadata_topic_count(documents_with_topics):
    assert all(
        (len(doc.metadata["topics"]) >= 2 and len(doc.metadata["topics"]) <= 4)
        for doc in documents_with_topics
    )
