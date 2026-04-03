import joblib
import pandas as pd

# Path to the regression model
MODEL_PATH = "models/predict_freight_model.pkl"

def load_model(model_path: str = MODEL_PATH):
    """
    Load trained freight cost prediction model.
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model

def predict_freight_cost(input_data):
    """
    Predict freight cost for new vendor invoices.
    
    Parameters
    ----------
    input_data : dict
    
    Returns
    -------
    pd.DataFrame with predicted freight cost
    """
    # Inside predict_freight.py

    model = load_model()
    input_df = pd.DataFrame(input_data)
    
    # Force the dataframe to ONLY have the 'Dollars' column before predicting
    features = ["Dollars"] 
    input_df['Predicted_Freight'] = model.predict(input_df[features]).round()
    
    return input_df
if __name__ == "__main__":
    # Example inference run (local testing)
    sample_data = {
        "Dollars": [18500, 9000, 3000, 200]
    }
    
    prediction = predict_freight_cost(sample_data)
    print("\n--- Freight Cost Prediction Results ---")
    print(prediction)