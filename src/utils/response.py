
from sdks.novavision.src.helper.package import PackageHelper
from capsules.Package.src.models.PackageModel import PackageModel, PackageConfigs, ConfigExecutor, ClipOutputs, ClipResponse, ClipExecutor, OutputData


def build_response(context):
    outputData = OutputData(value=context.outputData)
    clipOutputs = ClipOutputs(outputData=outputData)
    clipResponse = ClipResponse(outputs=clipOutputs)
    clipExecutor = ClipExecutor(value=clipResponse)
    executor = ConfigExecutor(value=clipExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel