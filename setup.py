import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name = "HipoMap",
    version = "0.1.3",
    author = "Jeongyeon Park",
    author_email ="ParkJYeon2808@gmail.com",
    url = "https://github.com/datax-lab/HipoMap",
    description = "Histopathological image analysis using Grad-CAM representation map",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages = setuptools.find_packages(),
    package_data={'hipo_map': ['WsSI_Preprocessing/Preprocessing/*.py']},
    install_requires=['numpy', 'opencv-python', 'pandas', 'Pillow', 'scikit-learn', 'scipy', 'seaborn', 'tensorflow', 'matplotlib'],
    python_requires = ">=3",
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
)