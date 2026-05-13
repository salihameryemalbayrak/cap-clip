
from pydantic import Field, validator
from typing import List, Optional, Union, Literal, Dict
from sdks.novavision.src.base.model import Package, Image, Inputs, Configs, Outputs, Response, Request, Output, Input, Config, Detection


class InputImage(Input):
    name: Literal["inputImage"] = "inputImage"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get('value')
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

    class Config:
        title = "Image"


class OutputData(Output):
    name: Literal["outputData"] = "outputData"
    value: Union[List[Image], Image, List[Detection], Detection, Dict, List]
    type: str = "object"

    class Config:
        title = "Output Data"

class DeviceCpu(Config):
    name: Literal["cpu"] = "cpu"
    value: Literal["cpu"] = "cpu"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "CPU"

class DeviceGpu(Config):
    name: Literal["cuda"] = "cuda"
    value: Literal["cuda"] = "cuda"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "GPU"


class Device(Config):
    """
            ...
    """
    name: Literal["device"] = "device"
    value: Union[DeviceGpu, DeviceCpu]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Device"

class ModelName(Config):
    """
    ...
    """
    name: Literal["modelName"] = "modelName"
    value: str = Field(default="openai/clip-vit-base-patch32")
    type: Literal["string"] = "string"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Model Name"

class BatchSize(Config):
    """
    ...
    """
    name: Literal["batchSize"] = "batchSize"
    value: str = Field(default=32)
    type: Literal["string"] = "string"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Batch Size"



class ClipInputs(Inputs):
    inputImage: InputImage


class ClipConfigs(Configs):
    device:Device
    modelName: ModelName
    batchSize: BatchSize


class ClipOutputs(Outputs):
    outputData: OutputData


class ClipRequest(Request):
    inputs: Optional[ClipInputs]
    configs: ClipConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class ClipResponse(Response):
    outputs: ClipOutputs


class ClipExecutor(Config):
    name: Literal["Clip"] = "Clip"
    value: Union[ClipRequest, ClipResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Package"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[ClipExecutor]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"
        json_schema_extra = {
            "target": "value"
        }


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["capsule"] = "capsule"
    name: Literal["Clip"] = "Clip"
