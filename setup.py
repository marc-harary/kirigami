from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(name="kirigami",
      version="1.0",
      description="RNA secondary structure prediction via deep learning",
      author="Marc Harary",
      author_email="marc.harary@yale.edu",
      packages=find_packages()
)
