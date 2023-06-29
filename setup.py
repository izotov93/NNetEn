from setuptools import setup, find_packages

VERSION = '1.0.7'

def readme():
  with open('README.md', 'r') as f:
    return f.read()

requirements = [
    'numba==0.57.1',
    'numpy==1.24.4'
]

setup(
    name='NNetEn',
    version=VERSION,
    description='Python Package for Neural '
                'Network Entropy (NNetEn) calculation',
    long_description=readme(),
    long_description_content_type="text/markdown",
    url='https://github.com/izotov93/NNetEn',
    author="Yuriy Izotov et al.",
    author_email='izotov93@yandex.ru',
    install_requires=requirements,
    packages=find_packages(),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
		'Programming Language :: Python :: 3.8',
		'Operating System :: OS Independent',
        'License :: OSI Approved :: BSD License',
	            ],
    python_requires='>=3.8',
    package_data={"NNetEn.Database": ["*.txt",
                                      "*.idx3-ubyte",
                                      "*.idx1-ubyte",
                                      "*.VM"]},
    include_package_data=True,
)
