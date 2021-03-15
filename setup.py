"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path
import codecs

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


### Set up tools to get version
def read(rel_path):
    here = path.abspath(path.dirname(__file__))
    with codecs.open(path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


### Do the setup
setup(
    name='darknet_highlights',
    author='Christophe Foyer',
    version=get_version("darknet_highlights/__init__.py"),
    description='A Python 3 package for generating highlights from darknet videos for Maui63',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='',
    author_email='c@cfoyer.com',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
    ],
    keywords='',
    packages=find_packages(exclude=['docs']),
    python_requires='>=3.6',
    install_requires=[
        'numpy >= 1',
        'tqdm >= 1',
        'opencv >= 4',
        'pandas >= 1',
        'filetype >= 1'
    ],
    include_package_data=True,
    package_data={
    },
    project_urls={  # Optional
    },
    entry_points={
    # 'console_scripts': [
    #         'darknet_highlights_to_csv=darknet_highlights.command_line.file_output:main',
    #         ],
    }
)