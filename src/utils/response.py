
from sdks.novavision.src.helper.package import PackageHelper
from capsules.Clip.src.models.PackageModel import PackageModel, PackageConfigs, ConfigExecutor, ClipImageOutputs, ClipImageResponse, ClipImageExecutor, OutputData


def build_response(context):
    outputData = OutputData(value=context.outputData)
    clipImageOutputs = ClipImageOutputs(outputData=outputData)
    clipImageResponse = ClipImageResponse(outputs=clipImageOutputs)
    clipImageExecutor = ClipImageExecutor(value=clipImageResponse)
    executor = ConfigExecutor(value=clipImageExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel