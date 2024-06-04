# setup.py
from setuptools import setup, find_packages

setup(
    name="BertConcatClassifier",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers"
    ],
    description="Custom BERT classifier with concatenation layer",
    author="kimduhyeon",
    author_email="kdhluck@naver.com"
)
