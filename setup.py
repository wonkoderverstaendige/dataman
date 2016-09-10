try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
        'description': 'Data Manager',
        'author': 'Ronny Eichler',
        'author_email': 'reichler@science.ru.nl',
        'version': '0.0.2dev',
        'install_requires': ['nose', 'termcolor', 'numpy', 'vispy'],
        'extra-requires': {'vis': 'PyQt4'},
        'packages': ['dataman'],
        'name': 'dataman',
        'entry_points': """
            [console_scripts]
            dm=dataman.__main__:main
        """
        }

setup(**config)

