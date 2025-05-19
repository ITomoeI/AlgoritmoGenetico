from setuptools import setup, find_packages

setup(
    name="algoritmo_genetico_itomoei",  # Nombre único basado en tu usuario de GitHub
    version="0.1.0",
    author="Bryan Rojas, Juan Ayala",
    author_email="brrojas.h14@gmail.com",
    description="Librería de algoritmos genéticos para optimización de funciones",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ITomoeI/AlgoritmoGenetico",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Development Status :: 4 - Beta",
    ],
    keywords="genetic algorithm, optimization, machine learning, evolutionary algorithm",
    python_requires=">=3.6",
)