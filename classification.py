import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from skopt import BayesSearchCV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.lib import logger as logManager

from src.lib.printer import plot_classification_report, plot_confusion_matrix, plot_feature_importance

from skopt.space import Integer, Real, Categorical


def optimizer(
    dataset,
    classifier=RandomForestClassifier(),
    search=BayesSearchCV.__name__,
    logger=None,
    outdir=None,
    param_args={},
):
    """Optimize a classification model using a specified search algorithm.

    Args:
        dataset (pandas.DataFrame): The dataset to use for training and testing the model. The last column should contain the target values.
        classifier (sklearn.base.BaseEstimator, optional): The classifier to optimize. Defaults to RandomForestClassifier().
        search (str, optional): The name of the search algorithm to use. Accepted values are 'GridSearchCV', 'BayesSearchCV', and 'RandomizedSearchCV'. Defaults to BayesSearchCV.__name__.
        logger (logging.Logger, optional): A logger to log information about the optimization process. If not specified, no logger will be used.
        outdir (str, optional): The directory in which to save the optimization results. If not specified, no results will be saved.
        param_args (dict, optional): Additional keyword arguments to pass to the search algorithm.

    Returns:
        tuple: A tuple containing the accuracy of the optimized model on the test set and the best parameters found by the search algorithm.
    """
    # features are in the first n columns of the dataset
    features = dataset.iloc[:, :-1]
    # targets are in the last column of the dataset
    targets = dataset.iloc[:, -1]

    logger.info("features:\n%s", features.head().to_string()) if logger is not None else None
    logger.info("targets: %s", targets.value_counts().to_dict()) if logger is not None else None

    # split the dataset in training e test set, with stratify to keep the classes balanced
    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.3, train_size=0.7, stratify=targets
    )

    # TRAINING 70%
    x_train_values = np.nan_to_num(X_train.values)
    y_train_values = np.nan_to_num(y_train.values).ravel()

    # VALIDATION 30%
    x_test_values = np.nan_to_num(X_test.values)
    y_test_values = np.nan_to_num(y_test.values)

    logger.info("Training set: %s", y_train.value_counts().to_dict()) if logger is not None else None
    logger.info("Test set: %s", y_test.value_counts().to_dict()) if logger is not None else None

    parameters = get_parameters(estimator=classifier, search=search, **param_args)

    logManager.redirect_stdout(logger) if logger is not None else None

    if search == GridSearchCV.__name__:
        clf = GridSearchCV(**parameters)
    elif search == BayesSearchCV.__name__:
        clf = BayesSearchCV(**parameters)
    elif search == RandomizedSearchCV.__name__:
        clf = RandomizedSearchCV(**parameters)
    else:
        raise ValueError("Invalid search algorithm")

    # fit and prediction
    clf.fit(x_train_values, y_train_values)

    logManager.restore_stdout() if logger is not None else None

    y_pred = clf.predict(x_test_values)

    # compute the accuracy
    accuracy = accuracy_score(y_test_values, y_pred)
    logger.info("accuracy: %s", str(accuracy)) if logger is not None else None


    # if the outdir is present, here we save the result
    if outdir != None:

        model_type = clf.best_estimator_.__class__.__name__
        
        save_args = {
            "report": {"y_true": y_test_values, "y_pred": y_pred, "output_dict": True, "n_class": targets.nunique()},
            "matrix": {"y_true": y_test_values, "y_pred": y_pred, "labels": targets.unique().tolist()},
            "dataset": {"X_train": X_train.copy(), "y_train": y_train, "X_test": X_test.copy(), "y_test": y_test},
            "params": clf.best_params_,
            "cv_result": pd.DataFrame(clf.cv_results_),
            }
    
        if hasattr(clf.best_estimator_, 'feature_importances_'):
            save_args["features"] = {"importance": np.array(clf.best_estimator_.feature_importances_), "names": features.columns}
        
        save_result(outdir=outdir, name=model_type + "_" + targets.name, logger=logger, **save_args)

    return accuracy


def save_result(outdir, name, logger=None, **kwargs):
    """Save the results of a classification model to an Excel file.

    Args:
        outdir (str): The directory in which to save the Excel file.
        name (str): The name of the Excel file to save.
        logger (logging.Logger, optional): A logger to log information about the saving process. If not specified, no logger will be used.
        **kwargs: A dictionary containing data to be saved to the Excel file. Accepted keys are:
            - "report": A dictionary containing parameters for scikit-learn's `classification_report` function. If present, a classification report will be generated and saved to the Excel file.
            - "matrix": A dictionary containing parameters for scikit-learn's `confusion_matrix` function. If present, a confusion matrix will be generated and saved to the Excel file.
            - "features": A dictionary containing parameters for the `plot_feature_importance` function. If present, a feature importance plot will be generated and saved to the Excel file.
            - "dataset": A dictionary containing the train and test dataframes. If present, the dataframes will be concatenated and saved to the Excel file.
            - "params": A dictionary containing the best parameters of the model. If present, they will be saved to the Excel file.
            - "cv_result": A DataFrame containing cross-validation results. If present, it will be saved to the Excel file.

    Returns:
        None
    """
    
    os.makedirs(outdir, exist_ok=True)
    logger.info("model_name: %s", name) if logger is not None else None
    
    writer = pd.ExcelWriter(os.path.join(outdir, name + ".xlsx"))

    if "report" in kwargs:
        writer = plot_classification_report(writer=writer, **kwargs.get('report'))

    if "matrix" in kwargs:
        writer = plot_confusion_matrix(writer=writer, **kwargs.get('matrix'))

    if "features" in kwargs:
        writer = plot_feature_importance(writer=writer, **kwargs.get('features'))

    if "dataset" in kwargs:
        # Add a dataset column to both the train and test dataframes
        df_train = kwargs.get('dataset').get("X_train")
        df_train["TARGET"] = kwargs.get('dataset').get("y_train")
        df_train["DATASET"] = "Train"

        df_test = kwargs.get('dataset').get("X_test")
        df_test["TARGET"] = kwargs.get('dataset').get("y_test")
        df_test["DATASET"] = "Test"

        # Concatenate the train and test dataframes back together
        dataset = pd.concat([df_train, df_test])
        dataset.to_excel(writer, sheet_name="Dataset", index=False)
    
    if "params" in kwargs:
        param_df = pd.DataFrame.from_dict(kwargs.get('params'), orient="index", columns=["value"])
        param_df.to_excel(writer, sheet_name="Best Params")

    if "cv_result" in kwargs:
        cv_result = kwargs.get('cv_result')
        cv_result.to_excel(writer, sheet_name="CV Result")

    writer.save()

def get_parameters(estimator, search, **kwargs):
    """Get the parameters for a given estimator and search strategy.

    This function takes an estimator instance and a search strategy name as input. It then loads the configuration
    parameters for the specified estimator using the load_params function and constructs a dictionary of common and
    specific parameters for the specified search strategy.

    The common parameters are shared by all search strategies and include the estimator instance, the cross-validation
    strategy, and other settings like verbosity and the number of jobs to use.

    The specific parameters depend on the search strategy and include the search space or parameter grid, the number of
    iterations, and other settings specific to each search strategy.

    Args:
        estimator: An instance of an estimator class.
        search (str): The name of the search strategy to use.
        **kwargs: Additional keyword arguments to override the default parameter values.

    Returns:
        dict: A dictionary containing the common and specific parameters for the specified search strategy.

    Raises:
        ValueError: If an unrecognized search strategy is specified.
    """
    
    params = load_params(type(estimator).__name__)

    common_params = {
        "estimator": estimator,
        "cv": kwargs.get("cv", RepeatedStratifiedKFold(n_splits=5, n_repeats=5)),
        "verbose": kwargs.get("verbose", 1),
        "n_jobs": kwargs.get("n_jobs", -1),
        "scoring": kwargs.get("scoring", None),
        "pre_dispatch": kwargs.get("pre_dispatch", "2*n_jobs"),
        "refit": kwargs.get("refit", True),
        "error_score": kwargs.get("error_score", "raise"),
        "return_train_score": kwargs.get("return_train_score", False),
    }

    if search == BayesSearchCV.__name__:
        specific_params = {
            "search_spaces": params,
            "n_iter": kwargs.get("n_iter", 50),
            "optimizer_kwargs": kwargs.get("optimizer_kwargs", None),
            "n_points": kwargs.get("n_points", 1),
            "random_state": kwargs.get("random_state", None),
        }
    elif search == GridSearchCV.__name__:
        grid = search_space_to_lists(params)
        specific_params = {"param_grid": grid}
    elif search == RandomizedSearchCV.__name__:
        distributions = search_space_to_lists(params)
        specific_params = {
            "param_distributions": distributions,
            "n_iter": kwargs.get("n_iter", 10),
            "random_state": kwargs.get("random_state", None),
        }
    else:
        raise ValueError(f"search_strategy '{search}' non riconosciuta.")

    return {**common_params, **specific_params}

def load_params(classifier_name):
    """Load the configuration parameters for a given classifier.

    This function reads a JSON configuration file named 'params.json' located in the same directory as this file.. It then
    extracts the configuration parameters for the specified classifier and returns them as a dictionary.

    The configuration parameters are instances of the Real, Integer, or Categorical classes from the skopt.space
    module. These instances are created by passing the keyword arguments stored in the JSON file to the appropriate
    class constructor.

    Args:
        classifier_name (str): The name of the classifier to load the configuration parameters for.

    Returns:
        dict: A dictionary containing the configuration parameters for the specified classifier.

    Raises:
        ValueError: If the specified classifier is not found in the configuration file or if an invalid class type is
                    encountered.
    """

    params_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'params.json')

    with open(params_path, "r") as f:
        config_json = json.load(f)
    
    config = {}
    if classifier_name in config_json:
        for key, value in config_json[classifier_name].items():
            if value["class"] == Integer.__name__:
                config[key] = Integer(**value["kwargs"])
            elif value["class"] == Real.__name__:
                config[key] = Real(**value["kwargs"])
            elif value["class"] == Categorical.__name__:
                config[key] = Categorical(value["choices"])
            else:
                raise ValueError("Invalid class type: %s not yet implemented", value["class"])    
    else:
        raise ValueError(f"{classifier_name} not found in the configuration file")
    
    return config

def search_space_to_lists(search_space):
    """Convert a search space dictionary to a dictionary of lists.
    
    This function takes a search space dictionary as input, where the keys are parameter names and the values are
    instances of the Real, Integer, or Categorical classes from the skopt.space module. It converts the search space
    into a new dictionary where the keys are the same parameter names and the values are lists of possible values for
    each parameter.

    For Real and Integer parameters, this function generates a list of evenly spaced values between the low and high
    bounds of the parameter. The number of samples is determined by the num_samples variable.

    For Categorical parameters, this function uses the categories attribute to generate a list of possible values.

    Args:
        search_space (dict): A dictionary representing the search space.

    Returns:
        dict: A new dictionary where the keys are parameter names and the values are lists of possible values for each parameter.
    """
   
    new_search_space = {}

    for param_name, param_value in search_space.items():
        if isinstance(param_value, Real):
            low = param_value.low
            high = param_value.high
            num_samples = 10  # Set the number of samples needed

            param_list = np.linspace(low, high, num_samples).tolist()
            new_search_space[param_name] = param_list
        
        elif isinstance(param_value, Integer):
            low = param_value.low
            high = param_value.high
            num_samples = 10  # Set the number of samples needed

            param_list = np.linspace(low, high, num_samples, dtype=int).tolist()
            new_search_space[param_name] = param_list
        
        elif isinstance(param_value, Categorical):
            new_search_space[param_name] = param_value.categories

        else:
            new_search_space[param_name] = param_value

    return new_search_space