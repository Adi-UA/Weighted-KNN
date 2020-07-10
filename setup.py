from distutils.core import setup, Extension

module = Extension("wknn", sources=["model.cpp"])

setup(name="WeightedKNN",
      version="1.0",
      description="Defines the weighted KNN algorithm",
      ext_modules=[module])
