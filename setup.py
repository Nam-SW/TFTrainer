from setuptools import find_packages, setup

setup(
    name="TFTrainer",
    version="0.2.0",
    description="tensorflow utility trainer.",
    author="Nam-SW",
    author_email="nsw0311@gmail.com",
    url="https://github.com/Nam-SW/TFTrainer.git",
    project_urls={
        "Bug Tracker": "https://github.com/Nam-SW/TFTrainer/issues",
    },
    # license="Apache",
    python_requires=">=3.6",
    install_requires=[
        "tensorflow>=2.4.1",
        "tensorboard",
        "tqdm",
        "pytz",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
