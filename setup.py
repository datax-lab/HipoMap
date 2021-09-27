import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="hipomap",
    version="0.3.4",
    author="Jeongyeon Park",
    author_email="ParkJYeon2808@gmail.com",
    url="https://github.com/datax-lab/HipoMap",
    description="Histopathological image analysis using Grad-CAM representation map",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'opencv-python', 'pandas', 'Pillow', 'scikit-learn', 'scipy', 'seaborn', 'tensorflow',
                      'matplotlib', 'openslide-python'],
    python_requires=">=3",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
)
