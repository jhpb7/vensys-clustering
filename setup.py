from setuptools import setup, find_packages

setup(
    name="vensys_clustering",
    version="0.1.0",
    description="Scenario reduction and probabilistic modeling for ventilation demand planning",
    author="Julius Breuer",
    author_email="juliusbreuer@posteo.de",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={"vensys_clustering": ["data/general.yml"]},
    install_requires=["numpy", "pandas", "scipy", "matplotlib", "scikit-learn"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
