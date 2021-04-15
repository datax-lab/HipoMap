import setuptools

setuptools.setup(
    name = "hipo-map",
    version = "0.1",
    author = "Jeongyeon Park",
    author_email ="ParkJYeon2808@gmail.com",
    url = "https://github.com/datax-lab/HipoMap",
    description = "Histopathological image analysis using Grad-CAM representation map",
    packages = setuptools.find_packages(),
    install_requires=[],
    python_requires = ">=3",
    classifiers = [
        "Programming Language :: Python :: 3",
    ],
)