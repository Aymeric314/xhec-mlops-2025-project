from prefect import task
from sklearn.metrics import root_mean_squared_error


@task(name="predict_rings")
def predict_rings(input_data, model):
    """
    Predicts the target variable using the provided model and input data.

    Args:
        input_data (csr_matrix): The input feature matrix in sparse format.
        model (LinearRegression): The trained linear regression model.

    Returns:
        np.array: The predicted values for the input data.
    """
    return model.predict(input_data)


@task(name="evaluate_model")
def evaluate_model(y_true, y_pred):
    """
    Evaluates the model's performance using the root mean squared error (RMSE).

    Args:
        y_true (np.ndarray): The true target values.
        y_pred (np.ndarray): The predicted target values.

    Returns:
        float: The computed RMSE value.
    """
    return root_mean_squared_error(y_true, y_pred)
