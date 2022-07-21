# pip install -e ".[interactive]"
# pytest -v --durations=0
# https://visualstudiomagazine.com/articles/2017/09/01/neural-network-l2.aspx

from setuptools import setup, find_packages


setup(
    name='net',
    version='1.0.0',
    description='A feed-forward neural network implementation by Tashfeen.',
    author='Tashfeen Ahmad',
    author_email='tashfeen@ou.edu',
    license='MIT',
    python_requires='>=3.7',
    packages=find_packages(include=['net', 'net.*']),
    install_requires=[
        'numpy'
    ],
    extras_require={
        'interactive': [
            'pytest',
            'matplotlib'
        ]
    }
)
