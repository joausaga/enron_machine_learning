#!/usr/bin/python

import pickle
import copy
import sys
sys.path.append('../tools/')

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn import model_selection, metrics, svm, tree
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.naive_bayes import GaussianNB
from new_features import compute_fraction


# Define constants
UNSENSE_PERSON_NAMES = ['THE TRAVEL AGENCY IN THE PARK', 'TOTAL']
MIN_NUMBER_OF_FEATURES = 10
ALGORITHMS = ['Gaussian', 'Decision Trees', 'SVM']
MIN_PERCENTAGE_OF_NO_NAN_FETURES = 10  # data points should more than 10% of their values distinct from 0.0
RESULTS_FORMAT_STRING = "Algorithm: {0}\nModel: {1}\nFeatures: {2}\nNumber of features: {3}\
\nSelected Features and scores: {4}\nData points in data set: {5}\nAccuracy: {6}\nPrecision: {7}\nRecall: {8}"

# Define initial list of features
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
                 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
                 'shared_receipt_with_poi']

# Load the dictionary containing the dataset
with open('final_project_dataset.pkl', 'r') as data_file:
    data_dict = pickle.load(data_file)

# Print initial stats of the data set
initial_set_size = len(data_dict)
print 'Initial data points: ', initial_set_size
num_features = len(data_dict['SKILLING JEFFREY K'])
print 'Initial number of features: ', num_features-1

# Remove people with nonsense names
# e.g., THE TRAVEL AGENCY IN THE PARK
for name in data_dict.keys():
    if name in UNSENSE_PERSON_NAMES:
        del data_dict[name]

# Create new features.
# The creation of the new features
# is based on the algorithm of the
# feature selection lesson, which is defined
# in the module new_features
for person_name in data_dict.keys():
    data_point = data_dict[person_name]
    # from poi to this person
    from_poi_to_this_person = data_point['from_poi_to_this_person']
    to_messages = data_point['to_messages']
    fraction_from_poi = compute_fraction( from_poi_to_this_person, to_messages )
    data_point['fraction_from_poi'] = fraction_from_poi
    # from this person to poi
    from_this_person_to_poi = data_point['from_this_person_to_poi']
    from_messages = data_point['from_messages']
    fraction_to_poi = compute_fraction( from_this_person_to_poi, from_messages )
    data_point['fraction_to_poi'] = fraction_to_poi
features_list.append('fraction_from_poi')
features_list.append('fraction_to_poi')
num_features = len(data_dict['SKILLING JEFFREY K'])


# From here on the program runs a series of steps to
# find the best model to predict whether a person is
# a POI.
# The program tries three different algorithms (e.g.,
# SVM, Decision Trees, and Naive-Bayes with. It executes
# each algorithm with different number of features
# trying to discover the best combination of features to
# fit the given data.

# Define an dictionary to store the information
# of the best model
empty_model_dict = {'algo_name': '',
                    'model_obj': None,
                    'num_features': 0,
                    'features_list': [],
                    'feature_scores': [],
                    'num_datapoints': 0,
                    'dataset': None,
                    'accuracy': -1,
                    'precision': -1,
                    'recall': -1}
best_models = []
best_model = empty_model_dict.copy()

# Iterate over the list of algorithms
for algo_name in ALGORITHMS:
    model_dict = empty_model_dict.copy()
    model_dict['algo_name'] = algo_name
    # Define the model using some fixed parameters.
    # Parameter tuning was performed manually.
    if algo_name == 'Gaussian':
        clf = GaussianNB()
    elif algo_name == 'SVM':
        clf = svm.SVC(gamma=100, C=10)
    elif algo_name == 'Decision Trees':
        clf = tree.DecisionTreeClassifier(min_samples_split=5,
                                          criterion='entropy')
    else:
        print 'Unknown algorithm: ', algo_name
        break
    # Iterate over different number of features, from a defined
    # minimum (10 in this case) to the maximum number of fetures
    # available
    for k in range(MIN_NUMBER_OF_FEATURES, num_features):
        # Store to my_dataset for easy export below.
        my_dataset = data_dict

        # Extract features and labels from data set
        data = featureFormat(my_dataset, features_list, sort_keys=True)
        labels, features = targetFeatureSplit(data)

        # Split data
        features_train, features_test, labels_train, labels_test = \
            model_selection.train_test_split(features, labels, test_size=0.3, random_state=42)

        # Select the k most influential features
        if k == num_features-1:
            selector = SelectKBest(f_classif, k='all')
        else:
            selector = SelectKBest(f_classif, k=k)
        selector.fit(features_train, labels_train)
        idx_selected_features = selector.get_support(indices=True)

        # Save the selected features and their scores into arrays
        features_selected = ['poi']
        features_and_scores = []
        for idx_feature in idx_selected_features:
            features_selected.append(features_list[idx_feature+1])
            features_and_scores.append(
                {'name': features_list[idx_feature+1], 'score': selector.scores_[idx_feature]}
            )

        # Remove data points which have less than
        # a minimum percentage of value distinct
        # from NaN on the selected features
        for person_name in my_dataset.keys():
            feature_nas = 0
            for feature in features_selected:
                if my_dataset[person_name][feature] == 'NaN':
                    feature_nas += 1
            per_no_nas_features = 100 - (feature_nas * 100 / k)
            if per_no_nas_features <= MIN_PERCENTAGE_OF_NO_NAN_FETURES:
                del my_dataset[person_name]

        # Extract selected features and labels from data set
        data = featureFormat(my_dataset, features_selected, sort_keys=True)
        labels, features = targetFeatureSplit(data)

        # Split data
        features_train, features_test, labels_train, labels_test = \
            model_selection.train_test_split(features, labels, test_size=0.3, random_state=42)

        # Train and test the model
        clf.fit(features_train, labels_train)
        pred = clf.predict(features_test)

        # Evaluate model's accuracy
        acc = metrics.accuracy_score(pred, labels_test)

        # Evaluate model's precision
        pres = metrics.precision_score(pred, labels_test)

        # Evaluate model's recall
        rec = metrics.recall_score(pred, labels_test)

        if pres > model_dict['precision'] or rec > model_dict['recall']:
            # Update the model dictionary if the current model has a
            # better precision or recall
            model_dict['model_obj'] = clf
            model_dict['num_features'] = k
            model_dict['features_list'] = features_selected
            model_dict['feature_scores'] = features_and_scores
            model_dict['num_datapoints'] = len(my_dataset)
            model_dict['dataset'] = my_dataset
            model_dict['accuracy'] = acc
            model_dict['precision'] = pres
            model_dict['recall'] = rec
    best_models.append(copy.deepcopy(model_dict))
    if best_model['precision'] < model_dict['precision'] or \
       best_model['recall'] < model_dict['recall']:
        # Update the dictionary that contains the so far
        # best model if in the previous iteration was found
        # a model with a better precision or recall
        best_model = copy.deepcopy(model_dict)

# Dump the best classifier together with the data set used
# and the list of features
dump_classifier_and_data(best_model['model_obj'], best_model['dataset'],
                         best_model['features_list'])

print '#### Best models for each algorithm ####'
for bm in best_models:
    # Remove poi since it isn't a feature
    bm['features_list'].remove('poi')

    # Update the features counter
    bm['num_features'] = bm['num_features']

    # Print information about the best model
    print RESULTS_FORMAT_STRING.format(
        bm['algo_name'],
        bm['model_obj'],
        bm['features_list'],
        bm['num_features'],
        bm['feature_scores'],
        len(bm['dataset']),
        bm['accuracy'],
        bm['precision'],
        bm['recall'],
    )
    print '#############\n'

print '\n'
print '#### The best of best models ####'

# Remove poi since it isn't a feature
best_model['features_list'].remove('poi')

# Update the features counter
best_model['num_features'] = best_model['num_features']

# Print information about the best model
print RESULTS_FORMAT_STRING.format(
    best_model['algo_name'],
    best_model['model_obj'],
    best_model['features_list'],
    best_model['num_features'],
    best_model['feature_scores'],
    len(best_model['dataset']),
    best_model['accuracy'],
    best_model['precision'],
    best_model['recall'],
)