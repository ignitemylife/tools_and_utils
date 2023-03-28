from setuptools import find_packages, setup

setup(
   name='ken_tools',
   version='2.0',
   description='A toolkit of Kenneth Kong',
   author='Kenneth Kong',
   author_email='ignitemylife943@gmail.com',
   packages=find_packages(exclude=('sql', 'workspace')),
   install_requires=['matplotlib', 'numpy', 'sklearn', 'mxnet', 'scipy', 'scikit-learn', 'tqdm'], #external packages as dependencies
)