from setuptools import setup, find_packages

setup(
    name="TransUNet",
    version="0.0.1",
    description="TransUNet is a hybrid deep learning model that integrates Transformers with the U-Net architecture for medical image segmentation",
    author="Atikul Islam Sajib",
    author_email="atikulislamsajib137@gmail.com",
    url="https://github.com/atikul-islam-sajib/TransUNet",
    packages=find_packages(),
    install_requires=[],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="TransUNet, deep-learning",
    project_urls={
        "Bug Tracker": "https://github.com/atikul-islam-sajib/TransUNet/issues",
        "Documentation": "https://github.com/atikul-islam-sajib/TransUNet",
        "Source Code": "https://github.com/atikul-islam-sajib/TransUNet",
    },
)
