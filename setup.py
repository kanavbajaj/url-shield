from setuptools import setup, find_packages

setup(
    name="urlclassifier",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'pyquery',
        'requests',
        'python-whois',
        'tensorflow',
        'scikit-learn',
        'streamlit',
        'interruptingcow',
        'matplotlib',
        'seaborn'
    ],
    author="Kanav Bajaj",
    author_email="kanavbajaj2004@gmail.com",
    description="A machine learning-based URL classification tool",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kanavbajaj/urlclassifier",
) 