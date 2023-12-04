from setuptools import setup
from analysis import __version__
 
setup( name='analysis',
    description='analysis module',
    author='Rick Albright',
    version=__version__,
    requires=['pandas', 'numpy', 'scipy', 'matplotlib', 'RollingStats'],
    py_modules=['analysis'],
    license='MIT License' )
