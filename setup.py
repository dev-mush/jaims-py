from setuptools import setup, find_packages

# Read the main requirements
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Read OpenAI specific requirements
with open("requirements-openai.txt") as f:
    requirements_openai = f.read().splitlines()

# Read Google Cloud AI specific requirements
with open("requirements-googleai.txt") as f:
    requirements_google_ai = f.read().splitlines()

setup(
    name="jaims-py",
    version="2.0.0-beta.2",
    packages=find_packages(),
    description="A Python package for creating simple AI Agents using the OpenAI API.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Marco Musella",
    url="https://github.com/dev-mush/jaims-py",
    license="MIT",
    keywords="An extensible library to create AI agents using many providers.",
    install_requires=requirements,
    extras_require={
        "openai": requirements_openai,
        "google": requirements_google_ai,
    },
)
