from webai_element_sdk.element.settings import ElementSettings, TextSetting
from webai_element_sdk.element.variables import ElementOutputs, Output
from webai_element_sdk.comms.messages import TextFrame  

class Outputs(ElementOutputs):
    default = Output[TextFrame]()

class Settings(ElementSettings):
    dataset_folder_path = TextSetting(
        name="dataset_folder_path",
        display_name="Data Path",
        description="The location on your system where the dataset used for OCR is located.",
        default="",
        required=True,
        hints=["folder_path"],
    )
