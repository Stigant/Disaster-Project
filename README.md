# Disaster Response Pipeline Project

This app classifies messages according to their relation to disaster relief. The model was trained on data from Figure Eight (now a part of [Appen](https://appen.com/)). 
A working copy can be found [here](https://stigant-disaster-app.herokuapp.com/).

The aim is to be able to determine which categories a given tweet, news report etc. fit into so that they can be passsed to relevant authorities and organisations. The categories themselves are things like 'Weather', 'Aid-related, 'Medical Help' and are specified in the training data, making this a multi-output, supervised classification task.
There is also a 'related' category in the training data, and messages in the training data are said to be related if and only if they are part of at least one other category. This restriction is also implemented in the model.

## Instructions
#### Data
Data can be loaded and cleaned with process_data.py. Run `(python) data/process_data.py [message filepath] [categories filepath] [database filepath] [table name]` to execute. Or, from the main directory, used default arguments with `(python) data/process_data.py default`.

#### Model
Model can be built with train_classifier.py. Run `(python) data/process_data.py [database filepath] [model filepath] [evalute model (True/False)]` to execute. Default arguments can be run with `(python) models/train_classifier.py default` or `(python) data/process_data.py default [evaluate model (True/False)]`.

#### App
The App folder contains the flask app. It can be run locally by running `(python) app/run.py from` the main repository directory. It will say the app can be found at http://0.0.0.0:3001/ but you may need to use localhost:3001. An online version is available [here](https://stigant-disaster-app.herokuapp.com/).

## Model Details

For a problem of this kind, it is much more important that relevant messages are flagged up than some extra are incorreclty identified as important, so it is natural to try and improve recall at the expense of precision. On the other hand, a model which is too imprecise is functionally useless. By splitting the task into two we are afforded more control over this tradeoff.

The model therefore breaks into the following parts:

1. Tokenize messages
2. Attempt to classify into related/not related
3. If classified as related perform mulit-output classification on disaster categories
4. If classified as related but not as part of any category then reclassify as unrelated

Here step 2 has both good recall and good precision. Step 3 is primarily tuned for recall and has relatively poor precision on some categories - essentially those for which there was little data. This imbalance was addressed somewhat by using SMOTE to resample data before training.

Step 1 was done with nltk. </br>
Step 2 was done with an SGDC classifier.</br>
Step 3 was done using a Logistic Regression with a custom threshold - implemented in ThresholdClassifier. OneByOneClassifier was used to allow for mulitoutput regression with ThresholdClassifier. </br>
Step 4 was done as part of the IfThenClassifier wrapper used to combine steps 1 and 2.

A DefaultClassifier class is also implemented. This allows default classifications to be assigned according to a dictionary of (Boolean) functions. In this case it was used to default empty/totally unseen messages to unrelated (these were previously predicted as related). This may not be needed when step 4 is used, but was kept anyway so that step 4 can be easily ommitted - set repredict=False in train_classifier.build_model to do so.

## Training Data

The training data consists of messages together with their genre (The method of delivery e.g. Social Media) and which, if any, disaster categories they belong to.
Some minor cleaning was needed - there were duplicate messages, some of which did not belong to the same disaser categories; plus some "#NAME?" messages which had to be removed.

## Dependencies
This project uses Python 3.
A full list of dependencies can be found in requirements.txt.
The main packages used for the model and data loading are 
* numpy
* pandas
* sklearn
* nltk
 
The web app runs on Flask.

