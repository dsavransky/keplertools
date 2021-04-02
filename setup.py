import setuptools
import os.path
import re

try:
    from Cython.Build import cythonize
    import numpy
    extensions = [setuptools.Extension("keplertools.CyKeplerSTM",
                                       [os.path.join("keplertools","KeplerSTM_C","CyKeplerSTM.pyx"),
                                        os.path.join("keplertools","KeplerSTM_C","KeplerSTM_C.c")],
                                       include_dirs = [numpy.get_include()]),
                  setuptools.Extension("keplertools.Cyeccanom",
                                       [os.path.join("keplertools","eccanom_C","Cyeccanom.pyx"),
                                        os.path.join("keplertools","eccanom_C","eccanom_C.c")],
                                       include_dirs = [numpy.get_include()])]
    extensions = cythonize(extensions)
except ImportError:
    extensions = []


with open("README.md", "r") as fh:
    long_description = fh.read()

with open(os.path.join("keplertools","__init__.py"), "r") as f:
    version_file = f.read()

version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",\
        version_file, re.M)

if version_match:
    version_string = version_match.group(1)
else:
    raise RuntimeError("Unable to find version string.")


setuptools.setup(
    name="keplertools",
    version=version_string,
    author="Dmitry Savransky",
    author_email="ds264@cornell.edu",
    description="Two-body orbital propagation tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dsavransky/keplertools",
    packages=setuptools.find_packages(exclude=['tests*']),
    install_requires=['numpy'],
    extras_require={
        'C': ['cython'],
    },
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    ext_modules=extensions
)
