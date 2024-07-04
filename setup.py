from setuptools import setup

setup(
    name='csvcmp',
    version='0.1.0',    
    description='a package for comparing csv or csv-like files',
    url='https://github.com/BrianWeiHaoMa/csvcmp',
    author='Brian Ma',
    author_email='brianmaytc@gmail.com',
    license='MIT',
    packages=['csvcmp'],
    install_requires=[
        'numpy>=2.0.0',
        'pandas>=2.2.2',
        'python-dateutil>=2.9.0.post0',
        'pytz>=2024.1',
        'six>=1.16.0',
        'tzdata>=2024.1'
    ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)