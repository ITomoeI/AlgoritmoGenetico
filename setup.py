from setuptools import setup, find_packages

setup(
    name="genetic_algorithm",
    version="0.1.0",
    author="YourName",
    author_email="your.email@example.com",
    description="Librería de algoritmos genéticos para optimización de funciones",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/genetic_algorithm",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)