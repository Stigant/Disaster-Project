# Disaster Response Pipeline Project

This app classifies messages according to their relation to disaster relief. The model was trained on data from Figure Eight (now a part of [Appen](https://appen.com/)). 
A working copy can be found [here](https://stigant-disaster-app.herokuapp.com/).

The layout is fairly self-explanatatory. 
The data file contains the data plus the data cleaning code and the database it outputs.
Models contains the model script, the custom classes and the pickled model.
App contains the flask app. It can be run locally by running (python) app/__init__.py from the main repository directory. It will say the app can be found at http://0.0.0.0:3001/ but you may need to use localhost:3001.

## Model:

The aim of the model is to determine which if any disaster related categories it is falls into. Messages may be a part of any category, and they are said to be related if and only if they are part of at least one category.

For a problem of this kind, it is much more important that messages are flagged up than some extra are incorreclty identified as important, so it is natural to try and improve recall at the expense of precision. On the other hand, a model which is too imprecise is functionally useless. By splitting the task into two we are afforded more control over this tradeoff.

The model therefore breaks into the following parts:

1. Tokenize messages
2. Attempt to classify into related/not related
3. If classified as related perform mulit-output classification on disaster categories
4. If classified as related but not as part of any category then reclassify as unrelated

Here step 2 has both good recall and good precision. Step 3 is primarily tuned for recall and has relatively poor precision on some categories - essentially those for which there was little data. This imbalance was addressed somewhat by using SMOTE to resample data before training.

Step 1 was done with nltk
Step 2 was done with an SGDC classifier.
Step 3 was done using a Logistic Regression with a custom threshold - implemented in ThresholdClassifier
Step 4 was done as part of the IfThenClassifier wrapper used to combine steps 1 and 2. This step can be turned off by setting reprodict=False.

## Training Data

The training data consists of messages together with their genre (The method of delivery e.g. Social Media) and which, if any, disaster categories they belong to.
Some minor cleaning was needed - there were duplicate messages, some of which did not belong to the same disaser categories; plus some "#NAME?" messages which had to be removed.

## Dependencies
This project uses Python 3.
A full list of dependencies can be found in requirements.txt.
The main packages used for the model and data loading are 
* numpy
* sklearn
* nltk
 
The web app runs on Flask.

## Things that might improve the model

* The text vectorisation isn't great as it stands and could be improved. For example it handles numbers very poorly - being able to identify phone numbers, co-ordinates and other location data would be very useful here.
* There are too few datapoints for some categories, one even has no datapoints. This imblance could possibly be better addressed with more sophisticated tools. At the end of the day, however, there is no real substitute for more data.
* More sophisticated models might help, though the limitations of the model seem fairly in line with those of the data itself.
* The model could be tuned for something other than recall, if a different metric is deemed more important.


