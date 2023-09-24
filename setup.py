import setuptools

with open("readme.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='lkcomputervision',
    version='0.1.0',
    description='A package for computer vision using MediaPipe',
    author='Lakshay kumar',
    author_email='contact@lakshaykumar.tech',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    install_requires=[
        'opencv-python',
        'mediapipe',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
