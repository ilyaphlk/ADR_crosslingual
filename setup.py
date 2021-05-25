import setuptools

setuptools.setup(
    name="ADR_crosslingual",
    version="0.0.1",
    author="Ilya Pakhalko",
    author_email="ilyaphlk@github.com",
    description="A small example package",
    long_description='',
    long_description_content_type="text/markdown",
    url='https://github.com/ilyaphlk/ADR_crosslingual',
    project_urls={
        "Bug Tracker": 'https://github.com/ilyaphlk/ADR_crosslingual/issues',
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    #package_dir={"": "src"},
    #packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
