import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="CERTEM",
    version="0.0.1",
    author="Tommaso Teofili",
    author_email="tommaso.teofili@gmail.com",
    description="CERTEM: Explaining and Debugging Black-box Entity Resolution Systems with CERTA",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url= 'https://github.com/tteofili/certem.git',
    packages=['certem'],
    install_requires=[
          'pandas',
          'numpy',
          'matplotlib',
      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        'License :: OSI Approved :: Apache Software License',
        "Operating System :: OS Independent",
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.6',
)
