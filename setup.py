"""
XERV CRAYON SETUP v5.2.5 - PURE PYTHON FALLBACK
=============================================
Guaranteed to work everywhere
"""

from setuptools import setup, find_packages

VERSION = "5.2.5"

setup(
    name="xerv-crayon",
    version=VERSION,
    packages=find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.8,<3.14",
    install_requires=[
        "numpy>=1.21.0",
    ],
)
