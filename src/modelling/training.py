from prefect import task
from sklearn.ensemble import RandomForestRegressor


@task(name="train_model")
def train_model(X_train, y_train):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model
