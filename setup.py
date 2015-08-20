try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
        'description': 'Data Manager',
        'author': 'Ronny Eichler',
        'url': '',
        'download_url': '',
        'author_email': 'ronny.eichler@gmail.com',
        'version': '0.0.1',
        'install_requires': ['nose'],
        'packages': ['dataman'],
        'scripts': [],
        'name': 'dataman'
        }

setup(**config)

        
