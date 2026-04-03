import joblib
import pandas as pd

# Path to the classification model and scaler
MODEL_PATH = "models/predict_flag_invoice.pkl"
SCALER_PATH = "models/scaler.pkl"

def load_model(model_path: str = MODEL_PATH):
    """
    Load trained classifier model.
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model

def predict_invoice_flag(input_data):
    """
    Predict invoice flag for new vendor invoices.
    
    Parameters
    ----------
    input_data : dict
    
    Returns
    -------
    pd.DataFrame with predicted flag
    """
    model = load_model()
    scaler = joblib.load(SCALER_PATH)  # Necessary for normalization
    
    input_df = pd.DataFrame(input_data)
    
    # Must match features used in training
    features = [
        "invoice_quantity", "invoice_dollars", "Freight", 
        "total_item_quantity", "total_item_dollars"
    ]
    
    # Scale inputs before prediction
    scaled_data = scaler.transform(input_df[features])
    
    # Generate prediction and round
    input_df['Predicted_Flag'] = model.predict(scaled_data).round()
    
    return input_df

if __name__ == "__main__":
    # Example inference run (local testing)
    sample_data = {
        "invoice_quantity": [50],
        "invoice_dollars": [352.95],
        "Freight": [1.73],
        "total_item_quantity": [162],
        "total_item_dollars": [2476.0]
    }
    
    prediction = predict_invoice_flag(sample_data)
    print("\n--- Invoice Risk Prediction Results ---")
    print(prediction)