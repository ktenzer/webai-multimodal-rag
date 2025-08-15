from webai_element_sdk.element.settings import ElementSettings, TextSetting, NumberSetting, equals
from webai_element_sdk.element.variables import ElementInputs, ElementOutputs, Input, Output
from webai_element_sdk.comms.messages import Frame

class Inputs(ElementInputs):
    default = Input[Frame]()  

class Outputs(ElementOutputs):
    default = Output[Frame]()  

class Settings(ElementSettings):
    # Strategy
    chunk_strategy = TextSetting(
        name="chunk_strategy",
        display_name="Chunking Strategy",
        description="Choose Recursive header-based or Semantic sentence-based chunking.",
        default="simple",
        valid_values=["simple","recursive", "semantic"],
        hints=["dropdown"],
        required=True,
    )

    # Recursive parameters
    chunk_size = NumberSetting[int](
        name="chunk_size",
        display_name="Chunk Size",
        description="Max words per header chunk",
        default=1800,
        min_value=250,
        max_value=2000,
        step=50,
        hints=["advanced"],
    )
    chunk_overlap = NumberSetting[int](
        name="chunk_overlap",
        display_name="Chunk Overlap",
        description="Overlap words between header chunks",
        default=200,
        min_value=10,
        max_value=300,
        step=10,
        hints=["advanced"],
    )

    # Semantic parameters
    semantic_model = TextSetting(
        name="semantic_model",
        display_name="Sentence Embedding Model",
        description="sentence-transformers model for semantic chunking",
        default="BAAI/bge-base-en-v1.5",
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
        required=True,
    )
    semantic_window_size = NumberSetting[int](
        name="semantic_window_size",
        display_name="Semantic Window Size",
        description="Sentences per semantic chunk",
        default=15,
        min_value=1,
        max_value=20,
        step=1,
        depends_on=equals("chunk_strategy", "semantic"),
        hints=["advanced"],
    )
    semantic_overlap = NumberSetting[int](
        name="semantic_overlap",
        display_name="Semantic Overlap",
        description="Sentences to overlap between semantic chunks",
        default=2,
        min_value=1,
        max_value=5,
        step=1,
        depends_on=equals("chunk_strategy", "semantic"),
        hints=["advanced"],
    )
    llm_token_model = TextSetting(
        name="llm_token_model",
        display_name="LLM Tokenizer Model",
        default="BAAI/bge-base-en-v1.5",
        description="Which tokenizer to use for max-token enforcement",
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
    )
    max_chunk_tokens = NumberSetting[int](
        name="max_chunk_tokens",
        display_name="Max Tokens per Chunk",
        default=512,
        description="Token budget per chunk (ensures LLM window fit)",
        min_value=64,
        max_value=512,
        step=50,
    )