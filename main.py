from src.data_preprocessing import load_data, preprocess_data
from src.train_model import train_model, save_model
from src.evaluate_model import evaluate_model

if __name__ == "__main__":
    df = load_data("data/raw/boston.csv")
    X_train, X_test, y_train, y_test = preprocess_data(df, target_column="medv")
    model = train_model(X_train, y_train)
    mse = evaluate_model(model, X_test, y_test)
    save_model(model)
    print(f"Model trained and saved! Test MSE: {mse}")