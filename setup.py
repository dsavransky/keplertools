import setuptools
import numpy


try:
    from Cython.Build import cythonize
except ImportError:
    use_cython = False
else:
    use_cython = True

ext = '.pyx' if use_cython else '.c'
extensions = [setuptools.Extension("keplertools.CyKeplerSTM", \
        ["keplertools/KeplerSTM_C/CyKeplerSTM"+ext, "keplertools/KeplerSTM_C/KeplerSTM_C.c"],\
        include_dirs = [numpy.get_include()])]

if use_cython:
    extensions = cythonize(extensions)


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="keplertools",
    version="0.0.1",
    author="Dmitry Savransky",
    author_email="ds264@cornell.edu",
    description="Two-body orbital propagation tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dsavransky/keplertools",
    packages=setuptools.find_packages(exclude=['tests*']),
    install_requires=['numpy'],
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    ext_modules = extensions
)
