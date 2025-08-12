from webai_element_sdk.element.settings import ElementSettings, BoolSetting, TextSetting
from webai_element_sdk.element.variables import ElementInputs, ElementOutputs, Input, Output
from webai_element_sdk.comms.messages import Frame 

class Inputs(ElementInputs):
    default = Input[Frame]()

class Outputs(ElementOutputs):
    default = Output[Frame]()

class Settings(ElementSettings):
    is_ingestion = BoolSetting(
        name="is_ingestion",
        display_name="Is Ingestion",
        default=True,
        description="Whether the element is in ingestion mode",
        required=False,
    )

    text_model = TextSetting(
        name="text_model",
        display_name="Text Embedding Model",
        default="BAAI/bge-base-en-v1.5",
        description="Sentence-transformers model for text embeddings.",
        valid_values=[
            "BAAI/bge-small-en-v1.5",
            "BAAI/bge-base-en-v1.5",
            "BAAI/bge-large-en-v1.5",
            "intfloat/e5-small-v2",
            "intfloat/e5-base-v2",
            "intfloat/e5-large-v2",
            "intfloat/multilingual-e5-small",
            "intfloat/multilingual-e5-base",
        ],
        hints=["dropdown"],
        required=False,
    )

    clip_model = TextSetting(
        name="clip_model",
        display_name="CLIP Image/Text Model",
        default="openai/clip-vit-base-patch32",
        description="CLIP model for image embeddings (used during ingestion).",
        valid_values=[
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-base-patch16",
            "openai/clip-vit-large-patch14",
            "openai/clip-vit-large-patch14-336",
        ],
        hints=["dropdown"],
        required=False,
    )    
