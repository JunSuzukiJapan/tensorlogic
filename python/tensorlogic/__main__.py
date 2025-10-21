"""
Entry point for running TensorLogic kernel via `python -m tensorlogic.kernel`
"""

if __name__ == '__main__':
    from .kernel import TensorLogicKernel
    from ipykernel.kernelapp import IPKernelApp
    IPKernelApp.launch_instance(kernel_class=TensorLogicKernel)
