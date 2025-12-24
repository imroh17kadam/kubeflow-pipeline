from sklearn.linear_model import LinearRegression
import joblib

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def save_model(model, path="models/linear_model.pkl"):
    joblib.dump(model, path)