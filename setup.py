from setuptools import setup
from setuptools import find_packages

setup(name='deepqn',
      version='0.0.2.dev1',
      description='A DQN implementation',
      url='https://github.com/linthieda/deepqn',
      author='Yuan Liu',
      author_email='linthieda@gmail.com',
      license='MIT',
      packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
      python_requires='>=3.6',
      zip_safe=False)