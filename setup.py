from setuptools import setup

# Metadata goes in setup.cfg. These are here for GitHub's dependency graph.
setup(
    name="Sklearn2JSON",
    install_requires=[
        "scikit-learn",
        "numpy",
        "scipy",
    ],
)
