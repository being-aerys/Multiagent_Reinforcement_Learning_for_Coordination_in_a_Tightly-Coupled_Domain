from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize('Cython_Code/ccea.pyx'))
setup(ext_modules=cythonize('Cython_Code/neural_network.pyx'))
setup(ext_modules=cythonize('Cython_Code/heterogeneous_rewards.pyx'))
setup(ext_modules=cythonize('Cython_Code/supervisor.pyx'))
setup(ext_modules=cythonize('Cython_Code/homogeneous_rewards.pyx'))
