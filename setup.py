from setuptools import setup, find_packages

setup(
    name="virophylo",
    version="2.0.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.4.0",
        "transformers>=4.40.0",
        "peft>=0.10.0",
        "accelerate>=0.30.0",
        "biopython",
        "ete3",
        "dendropy",
        "scikit-learn",
        "scipy",
        "numpy",
        "pandas",
        "pyyaml",
        "tqdm",
    ],
)
