# Enron Project - Data Analysis Nanodegree

## Report

### Goal

The goal of the project is to develop a machine learning-based solution to automatically identifies people who have been involved in the [Enron financial scandal](https://en.wikipedia.org/wiki/Enron_scandal), which led the company goes to bankrupt. The data set is composed of financial information of 146 Enron’s employees, including their salary, bonuses, total payment received, long advances, exercised stock options. Also, there is information about the number of emails messages sent/received by the person as well as how many of these messages are from or to people who were suspected to be involved in the scandal. Data set also contains information about the number of emails in which the person shared receipt with a suspected employee (a.k.a. a person of interest or poi). This information can be used to identify patterns in the financial movements and communication activities of the Enron’s employees who were involved in the scandal.

I identified two records with nonsense names (i.e., TOTAL and THE TRAVEL AGENCY IN THE PARK) and 1 with all missing values. The three were removed before features selection. Also, depending on the features selected, I checked whether they have at least 10% of the values available. Features that don’t comply with this requirement are removed before training the model.

### Features

I ended up using 16 features, namely: 

```
[{'score': 15.858730905995131, 'name': 'salary'}, 
{'score': 8.959136647690858, 'name': 'total_payments'}, 
{'score': 7.037932798193461, 'name': 'loan_advances'}, 
{'score': 30.728774633399713, 'name': 'bonus'}, 
{'score': 8.79220385270476, 'name': 'deferred_income'}, 
{'score': 10.633852048382538, 'name': 'total_stock_value'}, 
{'score': 4.180721484647058, 'name': 'expenses'}, 
{'score': 9.680041430380985, 'name': 'exercised_stock_options'}, 
{'score': 3.2044591402721507, 'name': 'other'}, 
{'score': 7.555119777320294, 'name': 'long_term_incentive'}, 
{'score': 8.058306312280525, 'name': 'restricted_stock'}, 
{'score': 1.6410979261701475, 'name': 'director_fees'}, 
{'score': 2.616183004679366, 'name': 'to_messages'}, 
{'score': 4.958666683966142, 'name': 'from_poi_to_this_person'}, 
{'score': 10.722570813682712, 'name': 'shared_receipt_with_poi'}, 
{'score': 15.838094949193755, 'name': 'fraction_to_poi'}]
```

Almost all of them contain information provided in the original data set except fraction_to_poi which is a feature that computes the proportion of emails that a given person sends to poi. I haven’t done any scaling because I did not consider necessary for the algorithms used. The selection of features was performed automatically and iterative using SelectKBest. Initially, a minimum number of features was chosen arbitrarily as a baseline (half of the total 10 in this case), and then we iterate from this baseline to the total features (20) trying to identify the number and combination of features that produce the best result.

### Algorithms

I tried three algorithms Naive-Bayes, Support Vector Machine, and Decision Trees. I ended up using a Decision Trees classifier because it showed to be the best classifier according to the precision and recall metrics. 

```
Algorithm: Decision Trees
Accuracy: 0.93023255814
Precision: 0.8
Recall: 0.666666666667
```

As it is shown below the other algorithms perform worse than Decision Trees

```
Algorithm: Gaussian
Accuracy: 0.906976744186
Precision: 0.6
Recall: 0.6
```

```
Algorithm: SVM
Accuracy: 0.883720930233
Precision: 0.0
Recall: 0.0
```

### Tunning Parameters

Tuning parameters allow improving the performance of the algorithm making it more tailored to the data at hand. Algorithms can underperform if they are not parameterized considering the data available and context-specific information. In my case, I manually tried different combinations of parameters until being satisfied with the result.

In the case of Decision Trees, I modified the criterion setting it as entropy and also the minimum sample split. By following a manual trial and error approach, I tested different numbers for the minimum sample split, ended up choosing five which showed to best option in this case. A similar procedure was followed in the case of the Support Vector Machine algorithm, where I played with different combinations of the parameters gamma and C until finding the mix with the highest accuracy. The precision and recall metrics did change with the variations I tried. The Naive-Bayes (Gaussian) algorithm only has one parameter which I left untouched.

### Validation

Validation means checking the accuracy of the model. The most common approach includes splitting the data into two independent sets: train and test. Then, the idea is to train the build model using the train set and assess its accuracy through the test set. If not conducting a correct validation (i.e., not using independent set to train and test), models overfit the data and cannot generalize to work with unseen data. Apart from splitting data in training and testing, there are other more advanced techniques to validate machine learning models, like k-fold cross validation where the data set is divided into k folds (say 10). Then, k validation iterations are run using some folds for training the model and the rest for testing.

In this case, the model is validated following the basic approach of splitting the data set into two independent sets of different sizes and then training in the biggest set and testing in the other. For my case, I reserved 30% of the data for the test and 70% for the training.

### Evaluation

In this case, precision and recall metrics are used for evaluation. Precision tells how many of the people identified as a person of interest are indeed a person of interest while recall gives how many of the persons of interest available in the dataset were correctly discovered by the algorithm.
