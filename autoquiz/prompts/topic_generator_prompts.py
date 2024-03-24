from typing import List
from langchain.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.pydantic_v1 import BaseModel, Field, validator


class DocumentTopics(BaseModel):
    """
    Represents the main topics discussed in depth in a given context.
    """

    topics: List[str] = Field(
        description="2 to 4 main topics discussed in depth in the given context"
    )

    @validator("topics")
    def topic_list_length(cls, field):  # pylint: disable=no-self-argument
        """
        Validates the length of the topics list.
        """
        if len(field) < 2 or len(field) > 4:
            raise ValueError(
                f"The topics field must contain 2 to 4 topics. Length: {len(field)}."
            )
        return field


def _generate_input_prompt(context: str) -> str:
    """Generates a prompt for the input task."""
    return f"""\
        Identify and extract 2 to 4 main topics discussed in depth in the given context. 
        It is important to answer with at least 2 and at most 4 topics. \nContext: {context}\
    """


_few_shot_examples = FewShotChatMessagePromptTemplate(
    example_prompt=ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    ),
    examples=[
        {
            "input": _generate_input_prompt(
                context="Photosynthesis in plants involves converting light energy into chemical energy, using chlorophyll and other pigments to absorb light. This process is crucial for plant growth and the production of oxygen.",
            ),
            "output": DocumentTopics(
                topics=[
                    "The process of photosynthesis in plants, including the conversion of light energy into chemical energy.",
                    "The role of chlorophyll and other pigments in absorbing light, crucial for plant growth and oxygen production.",
                ]
            ),
        },
        {
            "input": _generate_input_prompt(
                context="De leeuw is een grote katachtige met een brede kop, een korte snuit en relatief kleine, ronde oren. Hij heeft een kortharige asgrijze of zandgele vacht (maar variërend van okerbruin tot bijna wit) en een donker kwastje aan het puntje van de staart. Over de vacht verspreid liggen vage vlekken, die vooral bij jongere dieren goed te zien zijn. De vlekken vervagen naarmate het dier ouder wordt en zullen meestal uiteindelijk verdwijnen. De meeste mannetjes hebben dichte zwarte, bruine of gele manen met een variërende lengte op de kop, hals en schouders. Bij de nu uitgestorven ondersoorten uit Noord-Afrika en de Kaap liep de maan als een franje over de buik. Het duurt meestal een jaar of zes eer de maan goed ontwikkeld is. Wijfjes zijn kleiner en hebben geen manen.",
            ),
            "output": DocumentTopics(
                topics=[
                    "Fysieke kenmerken van leeuwen, inclusief hun grootte, vachtkleur en onderscheidende kenmerken zoals de manen en de staart.",
                    "Veranderingen in hun uiterlijk gerelateerd aan leeftijd, specifiek de ontwikkeling en het vervagen van vlekken en de groei van de manen bij mannetjes.",
                    "De vacht van de leeuw.",
                    "Beschrijving van de manen van de leeuw.",
                ]
            ),
        },
    ],
)


# The TOPIC_GENERATION_PROMPT consists of:
# - A system message that informs the AI about the task.
# - Few-shot examples to help the AI understand the task.
# - A human message that asks the AI to identify and extract the main topics discussed in
#   depth in the given context.
# - A placeholder for the AI's scratchpad.
TOPIC_GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert in finding the main topics in any given text. Write the values of your answer in Dutch.",
        ),
        _few_shot_examples,
        (
            "human",
            "Identify and extract the main topics discussed in depth in the given context.\nContext: {input}",
        ),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
