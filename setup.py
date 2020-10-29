from setuptools import find_packages, setup

setup(
   name='ken_tools',
   version='1.0',
   description='A toolkit of konglingshu',
   author='konglingshu',
   author_email='ignitemylife@gmail.com',
   packages=find_packages(exclude=('sql', 'workspace')),
   install_requires=['matplotlib', 'numpy', 'sklearn'], #external packages as dependencies
)