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
        'install_requires': ['nose', 'termcolor', 'vispy'],
        'packages': ['dataman'],
        'entry_points': """[console_scripts]
            dm=dataman.dataman:main""",
        'name': 'dataman'
        }

setup(**config)

        
