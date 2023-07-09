# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
We are using K-Nearest Neighbors for classification. Please refer the [train_model()](/starter/ml/model.py) function for further information.
Default number of neighbors: 50. To change this number, please change the [num_nb](./config.yaml)

## Usage
To run the model training and inference: [./scripts/run_train.sh](./scripts/run_train.sh)
To run the unit test: [./scripts/unit_test.sh](./scripts/unit_test.sh)
To deploy Fast APIs: [./scripts/run_APIs.sh](./scripts/run_APIs.sh)

## Intended Use
Predicting a person's annual salary is more than $50K based on his/her personal information, including:  
- workclass
- education
- marital-status 
- occupation
- relationship
- race
- sex
- native-country

## Training Data
Dataset: [UCI Census Income](https://archive.ics.uci.edu/ml/datasets/census+income)
Training set ratio: 80%. To change the ratio, please change the [test_size](./config.yaml)

## Evaluation Data
Dataset: [UCI Census Income](https://archive.ics.uci.edu/ml/datasets/census+income)
Training set ratio: 20%. To change the ratio, please change the [test_size](./config.yaml)

## Metrics
- Precision: 0.9423
- Recall: 0.1555
- F1: 0.2670 

## Ethical Considerations
This repository is used for the 3rd project of the Udacity course "Machine Learning DevOps Engineer".

## Caveats and Recommendations
The trained model currently consider the only categorical features of the dataset. 