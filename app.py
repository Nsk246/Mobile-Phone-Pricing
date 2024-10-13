import gradio as gr
import joblib
import numpy as np

# Load the saved XGBoost model and the StandardScaler
model = joblib.load('mobileprice_model.pkl')
scaler = joblib.load('scaler.pkl')  # Load the saved StandardScaler

# Function to make predictions with preprocessing
def predict_price(battery_power, blue, clock_speed, dual_sim, fc, four_g, int_memory, m_deep, mobile_wt, 
                  n_cores, pc, px_height, px_width, ram, sc_h, sc_w, talk_time, three_g, touch_screen, wifi):
    
    # Collecting the input features into a numpy array
    features = np.array([[battery_power, blue, clock_speed, dual_sim, fc, four_g, int_memory, m_deep, 
                          mobile_wt, n_cores, pc, px_height, px_width, ram, sc_h, sc_w, talk_time, 
                          three_g, touch_screen, wifi]])
    
    # Apply the same scaling that was used during training
    scaled_features = scaler.transform(features)

    # Making the prediction using the scaled input
    prediction = model.predict(scaled_features)[0]

    # Mapping the prediction to the price label
    price_dict = {0: "Low Cost", 1: "Medium Cost", 2: "High Cost", 3: "Very High Cost"}
    return price_dict[prediction]

# Building the Gradio interface
interface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Slider(500, 5000, step=100, label="Battery Power (mAh)"),
        gr.Checkbox(label="Has Bluetooth"),
        gr.Slider(0.5, 3.0, step=0.1, label="Clock Speed (GHz)"),
        gr.Checkbox(label="Has Dual SIM"),
        gr.Slider(0, 20, step=1, label="Front Camera Megapixels"),
        gr.Checkbox(label="Has 4G"),
        gr.Slider(2, 256, step=1, label="Internal Memory (GB)"),
        gr.Slider(0.1, 1.0, step=0.01, label="Mobile Depth (cm)"),
        gr.Slider(80, 300, step=1, label="Mobile Weight (gm)"),
        gr.Slider(1, 8, step=1, label="Number of Cores"),
        gr.Slider(2, 30, step=1, label="Primary Camera Megapixels"),
        gr.Slider(0, 2000, step=10, label="Pixel Height"),
        gr.Slider(0, 2000, step=10, label="Pixel Width"),
        gr.Slider(512, 8192, step=128, label="RAM (MB)"),
        gr.Slider(5, 20, step=1, label="Screen Height (cm)"),
        gr.Slider(5, 20, step=1, label="Screen Width (cm)"),
        gr.Slider(2, 30, step=1, label="Talk Time (hours)"),
        gr.Checkbox(label="Has 3G"),
        gr.Checkbox(label="Has Touch Screen"),
        gr.Checkbox(label="Has WiFi")
    ],
    outputs="text",
    title="Mobile Price Prediction",
    description="Enter the mobile features to predict the price range."
)

# Launching the Gradio interface
interface.launch()
