from setuptools import setup, find_packages

setup(
    name='lib',
    version='1.0.0',
    description='Tashfeen\'s research library functions.',
    author='Tashfeen Ahmad',
    author_email='tashfeen@ou.edu',
    license='MIT',
    python_requires='>=3.7',
    packages=find_packages(include=['lib', 'lib.*']),
    package_data={'lib': ['data/half_gaps.bin']},
    install_requires=['numpy', 'matplotlib'],
    extras_require={
        'interactive': [
            'pytest',
            'numpydoc',
            'scikit-learn'  # 'scikit-learn==1.1.1'
        ]
    })
