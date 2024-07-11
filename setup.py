from csvuniondiff import __version__
from setuptools import setup, find_packages

setup(
    name='csvuniondiff',
    version=__version__,    

    description='A package for comparing CSV-like files through union and difference operations.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',

    url='https://github.com/BrianWeiHaoMa/csvuniondiff',
    author='Brian Ma',
    author_email='brianmaytc@gmail.com',
    license='MIT',
    
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Financial and Insurance Industry',
        'License :: OSI Approved :: MIT License', 
        'Operating System :: OS Independent',  
        'Topic :: Software Development :: Testing',      
        'Topic :: Utilities',
        'Topic :: Office/Business :: Financial :: Spreadsheet',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.10',
    ],

    python_requires='>=3.10',
    packages=find_packages(exclude=('tests')),
    install_requires=[
        'numpy>=2.0.0',
        'pandas>=2.2.2',
        'python-dateutil>=2.9.0.post0',
        'pytz>=2024.1',
        'six>=1.16.0',
        'tzdata>=2024.1',
    ],

    entry_points={
        'console_scripts': [
            'csvuniondiff = csvuniondiff.command:main',
        ],
    },
)