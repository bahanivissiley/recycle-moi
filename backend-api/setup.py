"""
Setup configuration for Recycle-moi Backend
"""

from setuptools import setup, find_packages

setup(
    name="recyclemoi-backend",
    version="1.0.0",
    description="Backend API for waste classification using deep learning",
    author="Bahani",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.11",
    install_requires=[
        "torch>=2.5.1",
        "torchvision>=0.20.1",
        "fastapi>=0.104.1",
        "uvicorn>=0.24.0",
        "pydantic>=2.5.0",
        "pillow>=11.0.0",
        "numpy>=2.0.0",
        "pyyaml>=6.0.0",
        "python-multipart>=0.0.6",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "httpx>=0.25.2",
        ]
    },
)