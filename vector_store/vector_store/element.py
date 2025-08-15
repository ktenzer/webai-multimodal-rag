from webai_element_sdk.element.settings import ElementSettings, TextSetting, NumberSetting, equals
from webai_element_sdk.element.variables import ElementInputs, Input
from webai_element_sdk.comms.messages import Frame

class Inputs(ElementInputs):
    default = Input[Frame]()

class Settings(ElementSettings):
    # Choose between Chroma or PostgresML
    vector_store_type = TextSetting(
        name="vector_store_type",
        display_name="Vector Store Type",
        description="Which vector store to write to",
        default="chromadb",
        valid_values=["chromadb", "postgresml"],
        hints=["dropdown"],
        required=True,
    )

    # Chroma settings
    vector_db_folder_path = TextSetting(
        name="vector_db_folder_path",
        display_name="Chroma DB Path",
        description="Folder where the Chroma vector database will be stored.",
        default="",
        required=True,
        hints=["folder_path"],
        depends_on=equals("vector_store_type", "chromadb"),
    )
    max_batch_size = NumberSetting[int](
        name="max_batch_size",
        display_name="Max Chroma Batch Size",
        description="Maximum number of items per batch to Chroma.",
        default=2000,
        min_value=1,
        step=1,
        depends_on=equals("vector_store_type", "chromadb"),
        hints=["advanced"],
    )

    # PostgresML settings
    postgres_table_name = TextSetting(
        name="postgres_table_name",
        display_name="PostgresML Table",
        description="Name of table holding (id, embedding vector, metadata jsonb).",
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
