
from sdks.novavision.src.helper.package import PackageHelper
from capsules.Clip.src.models.PackageModel import PackageModel, PackageConfigs, ConfigExecutor, ClipStringOutputs, ClipStringResponse, ClipStringExecutor, ClipImageOutputs, ClipImageResponse, ClipImageExecutor, OutputData


def build_response_image(context):
    outputData = OutputData(value=context.outputData)
    clipImageOutputs = ClipImageOutputs(outputData=outputData)
    clipImageResponse = ClipImageResponse(outputs=clipImageOutputs)
    clipImageExecutor = ClipImageExecutor(value=clipImageResponse)
    executor = ConfigExecutor(value=clipImageExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel

def build_response_string(context):
    outputData = OutputData(value=context.outputData)
    clipStringOutputs = ClipStringOutputs(outputData=outputData)
    clipStringResponse = ClipStringResponse(outputs=clipStringOutputs)
    clipStringExecutor = ClipStringExecutor(value=clipStringResponse)
    executor = ConfigExecutor(value=clipStringExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel