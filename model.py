import json
import os

import numpy as np
import pandas as pd
from deprecated import deprecated
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     RepeatedStratifiedKFold, train_test_split)
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real

from src.lib import logger as logManager
from src.lib.printer import (plot_classification_report, plot_confusion_matrix,
                             plot_feature_importance, plot_scatter)


def train(
    dataset,
    estimator=RandomForestClassifier(),
    search=BayesSearchCV.__name__,
    logger=None,
    param_args={},
):
    """Train the given estimator using the dataset.

    Args:
        dataset: The dataset to use for training and testing.
        estimator: The estimator to use for training. Default is RandomForestClassifier.
        search: The search algorithm to use for hyperparameter tuning. Default is BayesSearchCV.
        logger: A logger to use for logging information. Default is None.
        param_args: Additional hyperparameters for the search algorithm. Default is an empty dictionary.

    Returns:
        A tuple containing the best estimator found by the search algorithm and a Dataset object containing the training and testing data.
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

    parameters = get_parameters(estimator=estimator, search=search, **param_args)

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

    return clf.best_estimator_, get_dataset(X_train.copy(), y_train, X_test.copy(), y_test)

def predict(model, dataset, logger=None):
    """This function predicts the target values for a given test dataset using a trained model.

    Args:
        model: a trained model object that can predict the target values of the dataset.
        dataset: a pandas dataframe object containing the test dataset to be used for prediction.
        logger: a logger object used to print the results to the console. (default=None)

    Returns:
        A tuple containing:
        - y_pred: a numpy array with the predicted target values for the test dataset.
        - accuracy or dictionary: the accuracy of the model if the model is a classifier, or a dictionary 
                                  with the R^2 score and RMSE values if the model is a regressor.
    """
    result = None

    # select all the rows with SPLIT == "Train"
    testset = dataset.loc[dataset["SPLIT"] == "Test"].copy()
    # remove the "SPLIT" column
    testset.drop(columns=["SPLIT"], inplace=True)

    logger.info("Test.shape = %s", testset.shape) if logger is not None else None

    # features are in the first n columns of the dataset
    features = testset.iloc[:, :-1]
    # targets are in the last column of the dataset
    targets = testset.iloc[:, -1]

    logger.info("Test features:\n%s", features.head().to_string()) if logger is not None else None
    logger.info("Test targets: %s", targets.value_counts().to_dict()) if logger is not None else None

    # predict on test data
    y_pred = model.predict(features)

    if hasattr(model, 'predict_proba'):
        # if the model is a classifier, compute the accuracy
        accuracy = accuracy_score(targets, y_pred)
        logger.info("accuracy = %s", accuracy) if logger is not None else None

        result = y_pred, accuracy
    else:
        # if the model is a regressor, compute the R^2 score and RMSE
        r2 = r2_score(targets, y_pred)
        rmse = mean_squared_error(targets, y_pred, squared=False)
        logger.info("r2 = %s", r2) if logger is not None else None
        logger.info("rmse = %s", rmse) if logger is not None else None

        result = y_pred, {'R^2 score': r2, 'RMSE': rmse}

    return result

def optimize(
    dataset,
    exp_name,
    estimator=RandomForestClassifier(),
    search=BayesSearchCV.__name__,
    logger=None,
    outdir=None,
    param_args={},
):
    """Optimize a model's hyperparameters using a search algorithm.
    
    This function trains a model with hyperparameters optimized by a search algorithm. It accepts a dataset,
    the name of the experiment, an estimator object, the search algorithm, and an optional logger and output
    directory. It returns predictions and metrics for the trained model.

    Args:
        dataset (pd.DataFrame): A Pandas DataFrame containing the dataset.
        exp_name (str): A string representing the name of the experiment.
        estimator (Union[ClassifierMixin, RegressorMixin], optional): An estimator object. Defaults to RandomForestClassifier().
        search (str, optional): A string representing the search algorithm. Defaults to BayesSearchCV.__name__.
        logger (Optional[logging.Logger], optional): A logger object. Defaults to None.
        outdir (Optional[str], optional): A string representing the output directory. Defaults to None.
        param_args (Optional[dict], optional): A dictionary of additional parameters to pass to the search algorithm. Defaults to {}.

    Returns:
        result: A variable containing metrics for the trained model.
    """
    model, dataset = train(dataset, estimator, search, logger, param_args)

    pred_output = predict(model, dataset, logger)

    model_type = get_model_type(estimator)

    # if the outdir is present, here we save the result
    if outdir != None:
        save(model, dataset, pred_output, outdir, exp_name, logger=logger, **model_type)

    return pred_output[1]

def save(model, dataset, pred_output, outdir, name="", logger=None, **kwargs):
    """Saves the model's output to an Excel file in the specified output directory.

    This function saves the trained model, predictions, and evaluation metrics to an Excel file in the specified
    output directory. The file name is automatically generated based on the model class name and an optional name
    parameter.

    Args:
        model (object): The trained model object.
        dataset (pandas.DataFrame): The dataset used to train and test the model.
        pred_output (tuple): A tuple containing two objects: the predicted target values and the evaluation metrics.
        outdir (str): The output directory where the Excel file will be saved.
        name (str, optional): The name to use for the Excel file. If not specified, the name of the model's class will be used.
        logger (logging.Logger, optional): A logger object for logging the progress of the function. Defaults to None.
        **kwargs: Additional keyword arguments specifying which outputs to include in the Excel file. Possible values include "report", "matrix", "features", "dataset", "params", "scatter", and "metrics".

    Returns:
        None
    """
    name = f"{model.__class__.__name__}_{name}" if name else model.__class__.__name__

    os.makedirs(outdir, exist_ok=True)
    logger.info("model_name: %s", name) if logger is not None else None

    predictions, metrics = pred_output

    writer = pd.ExcelWriter(os.path.join(outdir, name + ".xlsx"))

    if "report" in kwargs:
        y_true = dataset.loc[dataset["SPLIT"] == "Test"].drop(columns=["SPLIT"]).iloc[:, -1]
        n_class = dataset["TARGET"].nunique()
        writer = plot_classification_report(writer=writer, y_true=y_true, y_pred=predictions, output_dict=True, n_class=n_class)

    if "matrix" in kwargs:
        y_true = dataset.loc[dataset["SPLIT"] == "Test"].drop(columns=["SPLIT"]).iloc[:, -1]
        labels = dataset["TARGET"].unique().tolist()
        writer = plot_confusion_matrix(writer=writer, y_true=y_true, y_pred=predictions, labels=labels)

    if "features" in kwargs and hasattr(model, 'feature_importances_'):
        importance = np.array(model.feature_importances_)
        names = dataset.iloc[:, :-2].columns
        writer = plot_feature_importance(writer=writer, importance=importance, names=names)

    if "dataset" in kwargs:
        dataset.to_excel(writer, sheet_name="Dataset", index=False)
    
    if "params" in kwargs:
        param_df = pd.DataFrame.from_dict(model.get_params(), orient="index", columns=["value"])
        param_df.to_excel(writer, sheet_name="Best Params")

    if "scatter" in kwargs:
        y_true = dataset.loc[dataset["SPLIT"] == "Test"].drop(columns=["SPLIT"]).iloc[:, -1]

        writer = plot_scatter(writer=writer, y_true=y_true, y_pred=predictions)

    if "metrics" in kwargs:
        metrics_df = pd.DataFrame.from_dict(metrics, orient="index", columns=["value"])
        metrics_df.to_excel(writer, sheet_name="Metrics")

    # if "cv_result" in kwargs:
    #     cv_result = kwargs.get('cv_result')
    #     cv_result.to_excel(writer, sheet_name="CV Result")

    writer.save()

def get_model_type(model):
    """Determines the type of a scikit-learn model and returns a set of available information types necessary to save the results of the finded model.

    Args:
        model (object): A scikit-learn model instance.

    Returns:
        set: A set of available information types for the given scikit-learn model, including "report", "matrix",
        "features", "dataset", "params", "scatter", and "metrics".
    """
    type = {}

    if issubclass(model.__class__, ClassifierMixin):
        type = {"report", "matrix", "features", "dataset" "params"}
    elif issubclass(model.__class__, RegressorMixin):
        type = {"features", "dataset", "params", "scatter", "metrics"}

    return type

def get_dataset(X_train, y_train, X_test, y_test):
    """Converts the training and testing dataframes with their respective target variables into a single pandas DataFrame.
    
    Args:
        X_train (pd.DataFrame): The training dataframe.
        y_train (pd.Series): The target variable for the training dataset.
        X_test (pd.DataFrame): The testing dataframe.
        y_test (pd.Series): The target variable for the testing dataset.
    
    Returns:
        pd.DataFrame: A concatenated pandas DataFrame with columns for "TARGET" (values from y_train and y_test) and "SPLIT"
        (either "Train" or "Test" depending on the corresponding dataframe).
    """
    # Add a dataset column to both the train and test dataframes
    df_train = X_train
    df_train["TARGET"] = y_train
    df_train["SPLIT"] = "Train"

    df_test = X_test
    df_test["TARGET"] = y_test
    df_test["SPLIT"] = "Test"

    # Concatenate the train and test dataframes back together
    return pd.concat([df_train, df_test])

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
    for key in config_json:
        if key in classifier_name:
            for k, v in config_json[key].items():
                if v["class"] == Integer.__name__:
                    config[k] = Integer(**v["kwargs"])
                elif v["class"] == Real.__name__:
                    config[k] = Real(**v["kwargs"])
                elif v["class"] == Categorical.__name__:
                    config[k] = Categorical(v["choices"])
                else:
                    raise ValueError("Invalid class type: %s not yet implemented", v["class"])
            return config
    
    raise ValueError(f"No configuration found for classifier {classifier_name}")

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

@deprecated(reason="This function is deprecated and will be removed in the next version. Use the new function save instead.")
def _save_result(outdir, name, logger=None, **kwargs):
    """DEPRECATED
    ---
    Save the results of a classification model to an Excel file.

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
        df_train["SPLIT"] = "Train"

        df_test = kwargs.get('dataset').get("X_test")
        df_test["TARGET"] = kwargs.get('dataset').get("y_test")
        df_test["SPLIT"] = "Test"

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