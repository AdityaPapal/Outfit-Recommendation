from setuptools import setup,find_packages
from typing import List
HYPEN_E_DOT = '-e .'


def get_requirements(file_path:str)-> List['str']:
    requirements = []

    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
        
        return requirements
    


setup(
    name = "BE Project",
    version="0.0.1",
    author="Aadit",
    author_email="aaditrode2002@gmail.com",
    install_requires = get_requirements("requirements.txt"),
    packages = find_packages()
)