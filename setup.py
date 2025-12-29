"""
HMLR - Hierarchical Memory with Lattice Retrieval
Setup configuration for pip installation
"""

from setuptools import setup, find_packages
import os

# Read README for long description
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "HMLR - Hierarchical Memory with Lattice Retrieval for AI agents"

# Read core requirements
requirements_path = os.path.join(os.path.dirname(__file__), "requirements-core.txt")
if os.path.exists(requirements_path):
    with open(requirements_path, "r", encoding="utf-8") as fh:
        requirements = [
            line.strip() 
            for line in fh 
            if line.strip() and not line.startswith("#")
        ]
else:
    # Fallback if file doesn't exist
    requirements = [
        "openai>=1.0.0",
        "sentence-transformers>=2.2.0",
        "numpy>=1.24.0",
    ]

setup(
    name="hmlr",
    version="0.1.2",
    author="Sean-V-Dev",
    description="Hierarchical Memory with Lattice Retrieval - AI agent memory system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Sean-V-Dev/HMLR-Agentic-AI-Memory-System",
    project_urls={
        "Homepage": "https://github.com/Sean-V-Dev/HMLR-Agentic-AI-Memory-System",
        "Repository": "https://github.com/Sean-V-Dev/HMLR-Agentic-AI-Memory-System",
        "Issues": "https://github.com/Sean-V-Dev/HMLR-Agentic-AI-Memory-System/issues",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*", "junk_drawer", "junk_drawer.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "langchain": [
            "langchain>=0.1.0",
            "langchain-openai>=0.1.0",
        ],
        "telemetry": [
            "arize-phoenix>=4.0.0",
            "opentelemetry-api>=1.20.0",
            "opentelemetry-sdk>=1.20.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "ragas>=0.4.0",
            "langsmith>=0.2.0",
            "datasets>=2.14.0",
            "python-dotenv>=1.0.0",
        ],
    },
    package_data={
        "hmlr": [
            "config/*.json",
            "config/*.template",
        ],
    },
    include_package_data=True,
    keywords=[
        "ai",
        "memory",
        "agents",
        "rag",
        "llm",
        "retrieval",
        "bridge-blocks",
        "hierarchical-memory",
        "openai",
        "gpt-4.1-mini",
    ],
    zip_safe=False,
)
