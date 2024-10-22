from setuptools import setup, find_packages
from csvuniondiff import __version__

setup(
    name='csvuniondiff',
    version=__version__,    

    description='A package for comparing CSV-like files through union and difference operations.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',

    author='Brian Wei Hao Ma',
    author_email='brianmaytc@gmail.com',
    license='MIT',
    url='https://github.com/BrianWeiHaoMa/csvuniondiff',
    
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
    keywords=[
        'csv',
        'diff',
        'union',
        'compare',
        'files',
        'command-line',
        'tool',
        'data',
        'analysis',
        'pandas',
        'dataframes',
        'comparison',
        'merge',
        'difference',
        'columns',
        'rows',
        'data processing',
        'data integration',
        'file comparison',
        'open source',
        'csv files',
        'xlsx files',
        'json files',
        'xml files',
        'html files',
        'cli tool',
        'data matching',
        'data alignment',
        'data cleaning',
        'double-check',
        'data validation',
        'validation',
        'confirm',
        'error detection',
        'sql',
    ],

    python_requires='>=3.10',
    packages=find_packages(exclude=('tests')),
    install_requires=[
        'pandas>=2.2.2, <3.0.0',
    ],

    entry_points={
        'console_scripts': [
            'csvuniondiff = csvuniondiff.src.command:main',
        ],
    },
)