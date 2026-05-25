from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path: str) -> List[str]:
    """
    Reads the requirements.txt file and returns a list of dependencies.
    Removes any editable install flags if accidentally included.
    """
    requirements = []
    try:
        with open(file_path, "r") as file_obj:
            # Read lines and strip whitespace/newlines
            lines = [line.strip() for line in file_obj.readlines()]
            
            for line in lines:
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue
                # Handle cases where '-e .' might be left in requirements
                if line == "-e .":
                    continue
                requirements.append(line)
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Installing with empty dependencies.")
    
    return requirements

setup(
    name="distilbert_clf_pipeline",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="An enterprise-grade production text classification pipeline using fine-tuned DistilBERT.",
    long_description=open("README.md").read() if open("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/distilbert-clf-pipeline",
    # Automatically finds all directories with an __init__.py file inside src/
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=get_requirements("requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)