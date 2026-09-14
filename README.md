Machine Learning project using PyTorch for food delivery time predicting

The main Framework used in this project is "PyTorh" and the most important libraries is numpy, pandas, sklearn, matplotlib, seaborn

The project's structure is:

linearregressionML/
├─ configs/
│  └─ config.json
├─ data/
│  ├─ data.csv
│  └─ row_data.csv
├─ models/
│  └─ DeliveryModelV0.pth
├─ notebooks/
│  └─ notebook.ipynb
├─ src/
│  ├─ config.py
│  ├─ data.py
│  ├─ helper_functions.py
│  ├─ main.py
│  ├─ model.py
│  └─ train.py
└─ README.md

config.json: have the parameters that used in the project

data: have the data of the project include 2 main csv files 1-row_data.csv 2-data.csv
    1-row_data: the pure dataset data.
    2-data: the data after cleaning and turn strings into numbers by nominal and one-hot encoder

models: have all the models from experences

notebooks/notebook.ipynb: the notebook that have all experences and data cleaning and preparing

src/
    config.py: have all data in config.json to help import everything from 1 python file
    data.py: the file that adminstrator of splitting data into train and test data
    helper_functions: have some functions to help the model to have (shorter & cleaner code)
    main.py: the main code combine all the project's code in this file to train the model and give the result in the terminal
    train.py: the file that have the "train_test_func" to train and test the model

date: 9/14/2026
version: 0
