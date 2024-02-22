from setuptools import setup, find_packages
import re

# Load version from gradient/__init__.py
with open('gradient/__init__.py', 'r') as f:
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if version_match:
        version = version_match.group(1)
    else:
        raise RuntimeError("Unable to find version string.")

# Load requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='gradient',
    version=version,
    author='Yuma Rao',
    author_email='example@example.com',  # Optional: Replace with Yuma Rao's email
    description='A ML AI project on Bittensor',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/YumaRao/gradient',  # Optional: Replace with the actual URL
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.6',
)
