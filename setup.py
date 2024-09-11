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

with open("requirements-mistral.txt") as f:
    requirements_mistral = f.read().splitlines()

with open("requirements-anthropic.txt") as f:
    requirements_anthropic = f.read().splitlines()

with open("requirements-anthropic-vertex.txt") as f:
    requirements_anthropic_vertex = f.read().splitlines()

with open("requirements-vertexai.txt") as f:
    requirements_vertexai = f.read().splitlines()


setup(
    name="jaims-py",
    version="2.0.0-beta.17",
    packages=find_packages(),
    description="A Python package for creating LLM powered, agentic, platform agnostic software.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Marco Musella",
    url="https://github.com/dev-mush/jaims-py",
    license="MIT",
    keywords="An extensible library to create LLM Agents and LLM based applications.",
    install_requires=requirements,
    extras_require={
        "openai": requirements_openai,
        "google": requirements_google_ai,
        "vertexai": requirements_vertexai,
        "mistral": requirements_mistral,
        "anthropic": requirements_anthropic,
        "anthropic-vertex": requirements_anthropic_vertex,
        "anthropic-all": requirements_anthropic + requirements_anthropic_vertex,
        "all": requirements_openai
        + requirements_google_ai
        + requirements_vertexai
        + requirements_mistral
        + requirements_anthropic
        + requirements_anthropic_vertex,
    },
)
