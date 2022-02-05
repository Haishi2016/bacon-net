from setuptools import setup
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.1'
DESCRIPTION = 'A neural network architecture for building fully explainable neural network for arithmetic and gradient logic expression approximation.'

# Setting up
setup(
    name="baconnet",
    version=VERSION,
    author="haishibai (Haishi Bai)",
    author_email="<haishi.bai@live.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    install_requires=[],
    keywords=['ai', 'explainable', 'gradient logic',
              'approximation', 'formula', 'explainability', 'decision', 'decision making'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Indepedent",
        "License :: OSI Approved :: Apache Software License",
    ],
    extras_require={
        "dev": [
            "pytest >= 3.7",
        ],
    },
    py_modules=["hello"],
    package_dir={'': 'src'}
)
