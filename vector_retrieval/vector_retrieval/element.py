from webai_element_sdk.element.settings import ElementSettings, NumberSetting, TextSetting, BoolSetting, equals
from webai_element_sdk.element.variables import ElementInputs, ElementOutputs, Input, Output
from webai_element_sdk.comms.messages import Frame

class Inputs(ElementInputs):
    default = Input[Frame]()

class Outputs(ElementOutputs):
    default = Output[Frame]()

class Settings(ElementSettings):
    vector_store_type = TextSetting(
        name="vector_store_type",
        display_name="Vector Store Type",
        description="Which vector database backend to query",
        default="chromadb",
        valid_values=["chromadb", "postgresml"],
        hints=["dropdown"],
    )

    # Chroma path
    vector_db_folder_path = TextSetting(
        name="vector_db_folder_path",
        display_name="Vector DB Path",
        description="Folder where the Chroma vector database is stored (used if the input frame does not provide a path).",
        default="",
        required=False,
        hints=["folder_path"],
        depends_on=equals("vector_store_type", "chromadb"),
    )

    # PostgresML connection
    postgres_table_name = TextSetting(
        name="postgres_table_name",
        display_name="Postgres Table Name",
        description="Table (with pgvector column) to query.",
        default="webai",
        required=True,
        depends_on=equals("vector_store_type", "postgresml"),
    )
    pgml_host = TextSetting(
        name="pgml_host",
        display_name="PostgresML Host",
        default="localhost",
        depends_on=equals("vector_store_type", "postgresml"),
    )
    pgml_port = NumberSetting[int](
        name="pgml_port",
        display_name="PostgresML Port",
        default=5433,
        depends_on=equals("vector_store_type", "postgresml"),
    )
    pgml_db = TextSetting(
        name="pgml_db",
        display_name="PostgresML Database",
        default="postgresml",
        depends_on=equals("vector_store_type", "postgresml"),
    )
    pgml_user = TextSetting(
        name="pgml_user",
        display_name="PostgresML User",
        default="postgresml",
        depends_on=equals("vector_store_type", "postgresml"),
    )
    pgml_password = TextSetting(
        name="pgml_password",
        display_name="PostgresML Password",
        default="",
        depends_on=equals("vector_store_type", "postgresml"),
        sensitive=True,
        required=False,
    )    
    top_k = NumberSetting[int](
        name="top_k",
        display_name="Top K",
        description="Max number of results to return",
        default=3,
        min_value=1,
        step=1,
    )

    top_k_mmr = NumberSetting[int](
        name="top_k_mmr",
        display_name="Top K (MMR)",
        description="Number of Maximal Marginal Relevant (MMR)-selected candidates from the vector search before reranking.",
        default=20,
        min_value=1,
        step=1,
        hints=["advanced"],
    )

    top_k_bm25 = NumberSetting[int](
        name="top_k_bm25",
        display_name="Top K (BM25)",
        description="Number of BM25 lexical candidates to merge before reranking.",
        default=10,
        min_value=1,
        step=1,
        hints=["advanced"],
    )
    
    debug = BoolSetting(
        name="debug",
        display_name="Debug",
        description="If enabled, print detailed info about retrieved documents sent to LLM.",
        default=False,
        hints=["advanced"],
    )