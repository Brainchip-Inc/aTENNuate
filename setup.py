from setuptools import setup, find_packages


def parse_requirements(filename):
    with open(filename, 'r') as file:
        return [line.strip() for line in file if line and not line.startswith("#")]


setup(
    name='attenuate',  # Name of your package
    version='0.1.1',  # Initial version of your package
    author='Rudy Pei',  # Your name
    author_email='yanrpei@gmail.com',  # Your email
    description='Real-time raw speech enhancement with deep state-space modeling',  # A short description of the package
    long_description=open('README.md').read(),  # This will pull from your README.md for detailed description
    long_description_content_type='text/markdown',  # Indicating that README is in markdown format
    url='https://https://github.com/Brainchip-Inc/aTENNuate',  # Replace with your actual GitHub URL
    packages=find_packages(),  # Automatically find all packages and sub-packages
    install_requires=parse_requirements('requirements.txt'), 
    license='Apache-2.0',  # Apache 2.0 License
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',  # Indicate that it's Apache 2.0
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Minimum Python version requirement
)
