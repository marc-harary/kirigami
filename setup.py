from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(name="kirigami",
      version="1.0",
      description="RNA secondary structure prediction via deep learning",
      author="Marc Harary",
      author_email="marc.harary@yale.edu",
      packages=find_packages(),
      entry_points = {"console_scripts": ["kirigami = kirigami.__main__:main"]},
      # ext_modules=[CppExtension("kirigami.cpp_utils", ["kirigami/src/convert/convert.cpp"])],
      # cmdclass={"build_ext": BuildExtension}
)
