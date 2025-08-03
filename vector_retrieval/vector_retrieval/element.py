from webai_element_sdk.element.settings import ElementSettings, NumberSetting, TextSetting
from webai_element_sdk.element.variables import ElementInputs, ElementOutputs, Input, Output
from webai_element_sdk.comms.messages import Frame

class Inputs(ElementInputs):
    default = Input[Frame]()

class Outputs(ElementOutputs):
    default = Output[Frame]()

class Settings(ElementSettings):
    top_k = NumberSetting[int](
        name="top_k",
        display_name="Top K",
        description="Max number of results to return",
        default=3,
        min_value=1,
        step=1,
    )
    vector_db_folder_path = TextSetting(
        name="vector_db_folder_path",
        display_name="Vector DB Path",
        description="Folder where the Chroma vector database is stored (used if the input frame does not provide a path).",
        default="",
        required=False,
        hints=["folder_path"],
    )

    top_k_mmr = NumberSetting[int](
        name="top_k_mmr",
        display_name="Top K (MMR)",
        description="Number of Maximal Marginal Relevant (MMR)-selected candidates from the vector search before reranking.",
        default=40,
        min_value=1,
        step=1,
    )

    top_k_bm25 = NumberSetting[int](
        name="top_k_bm25",
        display_name="Top K (BM25)",
        description="Number of BM25 lexical candidates to merge before reranking.",
        default=10,
        min_value=1,
        step=1,
    )

    top_k_image = NumberSetting[int](
        name="top_k_image",
        display_name="Top K (Image)",
        description="Number of image results to retrieve via CLIP.",
        default=2,
        min_value=1,
        step=1,
    )