import re
from pathlib import Path

import setuptools
from setuptools import find_packages

FILE = Path(__file__).resolve()
PARENT = FILE.parent  # root directory
README = (PARENT / "README.md").read_text(encoding="utf-8")


def get_version():
    file = PARENT / "evaluations/__init__.py"
    return re.search(
        r'^__version__ = [\'"]([^\'"]*)[\'"]', file.read_text(encoding="utf-8"), re.M
    )[1]


setuptools.setup(
    name="evaluations",
    version=get_version(),
    author="Roboflow, Inc",
    author_email="support@roboflow.com",
    license="MIT",
    description="Evaluate ground truth and model predictions from Roboflow and supported zero-shot models",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/roboflow/evaluations",
    install_requires=["numpy>=1.20.0", "opencv-python", "matplotlib"],
    packages=find_packages(exclude=("tests",)),
    extras_require={
        "dev": [
            "flake8",
            "black==22.3.0",
            "isort",
            "twine",
            "pytest",
            "wheel",
            "mkdocs-material",
            "mkdocstrings[python]",
        ],
    },
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    keywords="Roboflow, computer vision, CV, computer vision evaluation",
    python_requires=">=3.7",
)
