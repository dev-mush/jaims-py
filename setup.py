from setuptools import setup, find_packages

# Read requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="jaims-py",
    version="1.0.0-beta.1",
    packages=find_packages(),
    description="A Python package for creating simple AI Agents using the OpenAI API.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Marco Musella",
    url="https://github.com/dev-mush/jaims-py",
    license="MIT",
    keywords="OpenAI GPT-3 and GPT-4 function enabled agent",
    install_requires=requirements,
)
