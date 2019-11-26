import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="impresso_stats",
    version="0.0.1",
    author="Justine Weber",
    author_email="justine.weber@epfl.ch",
    description="Statistics functions on the impresso dataset.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dhlab-epfl-students/impresso-metadata-explorer",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'dask',
        'matplotlib',
        'numpy',
        'pandas',
        'pandas',
        'PyMySQL',
        'scipy',
        'seaborn',
        'SQLAlchemy',
        'impresso_commons',
    ],
    dependency_links=[
      'https://github.com/impresso/impresso-pycommons/@v0.12.0#egg=impresso_commons',
      ]
)
