# Impresso - Metadata mining of large collections of historical newspapers 

### Basic information

- Student : Justine Weber
- Supervisors : Maud Ehrmann and Matteo Romanello
- Academic year : 2019-2020 (autumn semester)

### Introduction

This project contributes to the Impresso project (cf. https://impresso-project.ch/). 
It provides a python library - impresso_stats - to produce and visualize descriptive statistics 
about the impresso newspaper corpus. This library is intended to be used by historians who know 
the basis of code, and the team working on Impresso, in order to get information on the dataset, 
in an intuitive, fast and universal way. 
### Project summary

The package gathers a set of functions, made for providing statistics on the newspaper corpus and visualize them. 
Most functions which are intended to be used, perform a group-by and aggregrate operation (typically count or mean), 
return the aggregated dataframe, and display a bar plot of the result.

Statistics which can be obtained using the library's functions mainly concern:
 - issues frequency
 - content items frequency
 - licences
 - title length (of content items)

Full description of the functionalities is provided in the three tutorial jupyter notebooks.

This library should be enriched in the future, to provide more statistics, greater modularity, and better maintenance protection.

Below are some snapshots of what can be done.
![alt text][plot1]
![alt text][plot2]
![alt text][plot3]
![alt text][plot4]
![alt text][plot5]

[plot1]: https://github.com/dhlab-epfl-students/impresso-metadata-explorer/blob/master/images/plt-freq-ci-1d.png "title1"
[plot2]: https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 2"
[plot3]: https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 2"
[plot4]: https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 2"
[plot5]: https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 2"


### Repository description
This repository contains :
- a folder _notebooks_ gathering 4 jupyter notebooks : 3 tutorials and 1 use case, providing examples and indications on how to use the package.
- a folder _impresso_stats_ constituting the python package which one can install, and containing the code in 3 python files
	- `helpers.py` : set of helper functions
	- `sql.py` : set of functions for loading the dataset from SQL
	- `visualization.py` : set of main functions of the package
- a folder _report_ where you can find the report of the project
- a file `setup.py` useful for installing the package
- a file `requirements_basic.txt` containing basic dependencies of the project


### Installation and Usage
- Dependencies: the libraries that need to be installed are summarized in the `requirements_basic.txt` file.

- Package Installation (needed for running the tutorial notebooks) <br/>
	0. (Create and activate your environment)
	1. Install `impresso_pycommons` with `$ pip install https://github.com/impresso/impresso-pycommons/archive/v0.12.0.zip`
	2. Install requirements with `$ pip install -r requirements_basic.txt`
	3. Install package with `$ pip install https://github.com/dhlab-epfl-students/impresso-metadata-explorer/archive/master.zip`

- Usage: 
	1. activate your environment
	2. create a jupyter notebook
	3. import the functions you wish to use
		(example: `from impresso_stats.visualization import plt_freq_time_issues` >> cf. tutorials from more details)
	4. explore !

- Additional notes:
    - Functions in `sql.py` file load data from SQL, based on environment variables 
        - User name: 'IMPRESSO_MYSQL_USER'
        - Host name: 'IMPRESSO_MYSQL_HOST'
        - Database name: 'IMPRESSO_MYSQL_DB'
        Password: 'IMPRESSO_MYSQL_PWD'
        
        In order to use these functions, you need to define these environment variables in your `.bash_profile`.

### License  
**Impresso - Metadata mining of large collections of historical newspapers** - Justine Weber    
Copyright (c) 2020 EPFL    
This program is licensed under the terms of the MIT license.   


