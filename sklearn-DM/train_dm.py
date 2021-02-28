
import argparse
import os

# importing necessary libraries
import numpy as np

from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import joblib

from azureml.core import Dataset, Run

run = Run.get_context()
ws = run.experiment.workspace


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--kernel', type=str, default='linear',
                        help='Kernel type to be used in the algorithm')
    parser.add_argument('--penalty', type=float, default=1.0,
                        help='Penalty parameter of the error term')

    args = parser.parse_args()
    run.log('Kernel type', np.str(args.kernel))
    run.log('Penalty', np.float(args.penalty))

    # get the diabetes input dataset by ID
    dataset_name = 'early_stage_diabetes_transformed'
    dataset = Dataset.get_by_name(workspace=ws, name=dataset_name)

    df = dataset.to_pandas_dataframe()

    # dividing X, y into train and test data
    train_set, test_set = train_test_split(df, test_size = 0.2, random_state = 42)
    dia_train = train_set.drop("class", axis=1) # drop labels for training set
    dia_labels = train_set["class"].copy()

    dia_test = test_set.drop("class", axis=1) # drop labels for testing set
    dia_test_labels = test_set["class"].copy()

    # training a linear SVM classifier
    from sklearn.svm import SVC
    svm_model_linear = SVC(kernel=args.kernel, C=args.penalty).fit(dia_train, dia_labels)
    svm_predictions = svm_model_linear.predict(dia_test)

    # model accuracy for dia_test
    accuracy = svm_model_linear.score(dia_test, dia_test_labels)
    print('Accuracy of SVM classifier on test set: {:.2f}'.format(accuracy))
    run.log('Accuracy', np.float(accuracy))
    # creating a confusion matrix
    cm = confusion_matrix(dia_test_labels, svm_predictions)
    print(cm)

    os.makedirs('outputs', exist_ok=True)
    # files saved in the "outputs" folder are automatically uploaded into run history
    joblib.dump(svm_model_linear, 'outputs/model.joblib')


if __name__ == '__main__':
    main()