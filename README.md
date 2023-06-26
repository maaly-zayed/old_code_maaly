# Disaster-Response-Pipeline
<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [Setting up the environment](#setting-up-the-environment)
- [Installation](#installation)
- [Project Motivation](#project-motivation)
- [Project Descriptions](#project-descriptions)
- [Files Descriptions](#files-descriptions)
- [Instructions](#instructions)


<!-- END doctoc generated TOC please keep comment here to allow auto update -->



## Setting up the environment

It is very important to create separate environment to avoid dependecies conflict with other projects.
1. Have Anaconda installed [click here for Anaconda installation](https://www.anaconda.com)
2. Make sure you have [Anaconda path variable](https://www.geeksforgeeks.org/how-to-setup-anaconda-path-to-environment-variable/) set in windows environment variables. 
3. Open cmd and run this command to create environment (must be python 3.6 or later):
```bash
conda create --name Disaster-Response-Pipeline python==3.8 
```
4. Activate the environment:
```bash
conda activate Disaster-Response-Pipeline
```
Now, continue while the environemt is active.

## Installation

The required libraries are included in the file ```bash requirements.txt```

1. Install required libraries.
```bash
pip install -r requirements.txt
```
2. To run ETL Pipeline that clean and store data
```bash
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```
