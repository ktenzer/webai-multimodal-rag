from webai_element_sdk.element.variables import ElementInputs, ElementOutputs, Input, Output
from webai_element_sdk.comms.messages import Frame 

class Inputs(ElementInputs):
    default = Input[Frame]()

class Outputs(ElementOutputs):
    default = Output[Frame]()

