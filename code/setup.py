# local code/ann has to be installed manually if needed.
from setuptools import setup


setup(
    name='code',
    version='1.0.0',
    description='Tashfeen\'s research code.',
    author='Tashfeen Ahmad',
    author_email='tashfeen@ou.edu',
    python_requires='>=3.7',
    install_requires=[
        'numpy',
        'matplotlib',
        'scikit-learn',
        'pytest'
    ]
)
