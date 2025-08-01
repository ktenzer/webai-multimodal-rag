from webai_element_sdk.element.settings import ElementSettings, TextSetting
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
