
"""
Setup para el paquete Axón Mealenizado integrado en Ruth R1
"""

from setuptools import setup, find_packages

setup(
    name="ruth_r1_axon_mealenizado",
    version="1.0.0",
    packages=find_packages(),
    description="Axón Mealenizado - Brincos neuroplásticos integrado en Ruth R1",
    long_description="""
    Sistema de Axón Mealenizado integrado en la arquitectura Ruth R1.
    Implementa brincos neuroplásticos y decisiones por impulsos con:
    - NeuronaA para almacenamiento de capas de experiencia
    - NeuronaB para decisiones por brinco cognitivo
    - Integración con grafos axónicos mielinizados
    - Algoritmo Amiloid para optimización neural
    - Conectividad con sistema de consciencia bayesiana
    """,
    author="Haim Ben Shaul Reyes U.",
    author_email="reyesurijaime@gmail.com",
    url="https://github.com/DeHaim/axon_mealenizado",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires='>=3.8',
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.21.0",
        "difflib",
        "datetime",
        "typing-extensions>=3.7.4",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
        "visualization": [
            "plotly>=5.0",
            "networkx>=2.6",
        ]
    },
    entry_points={
        "console_scripts": [
            "axon-mealenizado=modules.axon_mealenizado:main",
        ],
    },
    keywords="neural-networks neuroplasticity cognitive-jumps ruth-r1 consciousness",
    project_urls={
        "Bug Reports": "https://github.com/DeHaim/axon_mealenizado/issues",
        "Source": "https://github.com/DeHaim/axon_mealenizado",
        "Documentation": "https://github.com/DeHaim/axon_mealenizado/wiki",
    },
)
