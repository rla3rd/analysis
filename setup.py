from setuptools import setup
from analysis import __version__
 
setup( name='analysis',
    description='analysis module',
    author='Richard Albright',
    version=__version__,
    requires=['pandas', 'numpy', 'scipy', 'matplotlib'],
    py_modules=['analysis', 'chart', 'colors', 'XLpandas'],
    license='MIT License' )
