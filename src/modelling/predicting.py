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
