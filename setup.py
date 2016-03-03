from setuptools import setup

classifiers = [
    "Development Status :: 3 - Alpha",
    "License :: OSI Approved :: MIT License",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

setup(
    name='scikit-chainer',
    version='0.4',
    packages=['skchainer'],
    url='https://github.com/lucidfrontier45/scikit-chainer',
    license='MIT',
    author='Shiqiao Du',
    author_email='lucidfrontier.45@gmail.com',
    description='scikit-learn like interface for chainer',
    classifiers=classifiers,
    install_requires=["numpy", "scikit-learn", "chainer >= 1.4.1"]

)
