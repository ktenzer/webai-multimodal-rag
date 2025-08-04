from webai_element_sdk.element.settings import ElementSettings, TextSetting, NumberSetting
from webai_element_sdk.element.variables import ElementInputs, Input
from webai_element_sdk.comms.messages import Frame

class Inputs(ElementInputs):
    default = Input[Frame]()

class Settings(ElementSettings):
    vector_db_folder_path = TextSetting(
        name="vector_db_folder_path",
        display_name="Vector DB Path",
        description="Folder where the Chroma vector database will be stored.",
        default="",           
        required=False,
        hints=["folder_path"],  
    )

    max_batch_size = NumberSetting[int](
        name="max_batch_size",
        display_name="Max Chroma Batch Size",
        description="Maximum number of items to send to Chroma in a single add() call",
        default=2000,
        min_value=1,
        step=1,
    )