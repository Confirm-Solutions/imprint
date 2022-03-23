import os
from setuptools import setup, find_packages


CWD = os.path.abspath(os.path.dirname(__file__))

# Get long description by reading README.md (as one should).
with open(os.path.join(CWD, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="pykevlar",
    description="Kevlar exports to Python.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mikesklar/kevlar",
    author="Confirm Solutions Modelling",
    author_email="contact@confirmsol.org",
    # TODO: lol we need one: license="BSD",
    classifiers=[  # Optional
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    packages=find_packages(),
    install_requires=["numpy", "pybind11"],
    data_files=[("../../pykevlar", ["core.so"])],
    zip_safe=False,
    version=os.environ["VERSION"],
)
