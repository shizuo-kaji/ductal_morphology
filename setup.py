import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ductal_morphology",
    version="0.0.1",
    install_requires=[
        "numpy","scikit-image","scikit-fmm","pynrrd","matplotlib","scipy","skan","networkx","seaborn","persim"
    ],
    author="Shizuo Kaji",
    author_email="skaji@imi.kyushu-u.ac.jp",
    description="Morphology of ductal structure",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shizuo-kaji/ductal_morphology",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.4',
)