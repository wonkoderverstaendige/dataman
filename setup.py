try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='dataman',
      description='Data Manager',
      author='Ronny Eichler',
      author_email='ronny.eichler@gmail.com',
      version='0.1.1',
      install_requires=['nose', 'termcolor', 'vispy', 'numpy', 'tqdm'],
      packages=['dataman'],
      entry_points="""[console_scripts]
            dm=dataman.dataman:main""")
