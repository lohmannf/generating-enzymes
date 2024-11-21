from setuptools import setup, find_packages

setup(
    name="genzyme",
    version="0.0.1",
    description="Generative models for enzyme sequence design",
    url="https://github.com/lohmannf/generating-enzymes",
    author="Frederieke Lohmann",
    author_email="flohmann@ethz.ch",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.11",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "pandas",
        "mavenn",
        "torch",
        "datasets",
        "moviepy",
        "transformers",
        "pytorch-minimize",
        "blosum",
    ],
)
