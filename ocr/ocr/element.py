from webai_element_sdk.element.settings import ElementSettings, TextSetting, BoolSetting
from webai_element_sdk.element.variables import ElementOutputs, Output
from webai_element_sdk.comms.messages import Frame

class Outputs(ElementOutputs):
    default = Output[Frame]()

class Settings(ElementSettings):
    dataset_folder_path = TextSetting(
        name="dataset_folder_path",
        display_name="Data Path",
        description="The location on your system where the dataset used for OCR is located.",
        default="",
        required=True,
        hints=["folder_path"],
    )
    image_store_folder = TextSetting(
        name="image_store_folder",
        display_name="Image Storage Folder",
        description="Where extracted images will be saved (absolute or ~/…)",
        default="~/.webai/assets/images",
        hints=["folder_path"],
    )
    keep_chart_like = BoolSetting(
        name="keep_chart_like",
        display_name="Keep Chart/Diagram Images Without Text",
        description="If true, keep images predicted as chart/diagram/table/screenshot even when no text is found.",
        default=True,
    )
    # IMPORTANT: BoolSetting should not be parameterized with [bool]
    use_blip = BoolSetting(
        name="use_blip",
        display_name="Use BLIP captioner",
        description="Generate natural-language captions for images (slower).",
        default=False,
    )
    use_chartqa = BoolSetting(
        name="use_chartqa",
        display_name="Use Chart/Table interpreter (ChartQA)",
        description="Attempt to read structured data from chart-like images (slower).",
        default=False,
    )