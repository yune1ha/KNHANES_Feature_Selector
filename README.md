# KNHANES_Feature_Selector
Various feature selection methods designed to be performed on KNHANES (Korea Ntional Health and Nutrition Examination Survey) data.

Concatenator is included which combines all years' tables (initial feature set selection is recommended as using all columns will result in too large of a cache).

Feature selection methods available: PCA, MCA, FAMD. Other methods may be easily added. 

Current example is designed to select all nutrition related features and performs feature selection only among them. 
Desired initial set can be easily changed in the features/features.json .

Double check all your paths have been localized first. 

Notebooks have been added as reference only.

To run script, simply run run.py 
