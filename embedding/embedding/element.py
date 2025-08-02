from webai_element_sdk.element.settings import ElementSettings, BoolSetting
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
