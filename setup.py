from setuptools import setup

setup(
    name="TFTrainer",
    version="0.0.1",
    description="tensorflow utility trainer.",
    author="Nam-SW",
    author_email="nsw0311@gmail.com",
    url="https://github.com/Nam-SW",
    license="Apache",
    python_requires=">=3.6.8",
    install_requires=[
        "tensorflow>=2.4.1",
        "tensorboard",
        "tqdm",
        "pytz",
    ],
    packages=["src"],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
