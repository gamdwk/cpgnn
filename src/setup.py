from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

extra_compile_args={}
extra_compile_args['nvcc'] = ['-g', '-G']
setup(
    name='cpgnn',
    ext_modules=[
        CUDAExtension('cpgnn', [
            'op_kernel.cu',
            'op.cpp',], include_dirs=['.'], 
            extra_compile_args=extra_compile_args)
    ],
    cmdclass={
        'build_ext': BuildExtension
    })