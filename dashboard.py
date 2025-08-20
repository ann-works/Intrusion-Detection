import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
from tensorflow.keras.models import load_model

# --- Page Configuration ---
st.set_page_config(
    page_title="Real-time Intrusion Detection",
    page_icon="🛡️",
    layout="wide"
)

# --- Load Model and Preprocessor ---
@st.cache_resource
def load_artifacts():
    """Loads the trained model, preprocessor, and label encoder."""
    try:
        # MODIFICATION: Updated file paths to look inside the 'models' folder
        model = load_model('models/lstm_model_multiclass_balanced.keras') 
        preprocessor = joblib.load('models/preprocessor_multiclass_balanced.joblib')
        label_encoder = joblib.load('models/label_encoder_multiclass.joblib')
        return model, preprocessor, label_encoder
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None, None

model, preprocessor, label_encoder = load_artifacts()

# --- Load Data for Simulation ---
@st.cache_data
def load_simulation_data():
    """Loads and prepares the full dataset for the simulation feed."""
    # MODIFICATION: Updated file path to look inside the 'data' folder
    df = pd.read_csv('data/Time-Series_Network_logs.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp').reset_index(drop=True)
    return df

df_full = load_simulation_data()

# --- Helper Function to Create a Balanced Simulation Feed ---
def create_simulation_feed(df, intrusion_ratio=0.25, total_logs=500):
    """Creates a shuffled feed with a specific ratio of intrusion events."""
    normal_logs = df[df['Intrusion'] == 0]
    intrusion_logs = df[df['Intrusion'] == 1]

    num_intrusions = int(total_logs * intrusion_ratio)
    num_normals = total_logs - num_intrusions

    # Sample from each category
    intrusion_sample = intrusion_logs.sample(n=num_intrusions, random_state=42)
    normal_sample = normal_logs.sample(n=num_normals, random_state=42)

    # Combine and shuffle
    simulation_df = pd.concat([intrusion_sample, normal_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
    return simulation_df

# --- Main App UI ---
st.title("🛡️ Real-time Network Intrusion Detection Dashboard")
st.write("This dashboard simulates a live network feed, using a trained LSTM model to classify traffic into Normal, BotAttack, or PortScan in real-time.")

# --- Sidebar for Controls ---
st.sidebar.header("Simulation Controls")
intrusion_percentage = st.sidebar.slider("Intrusion Event Ratio (%)", 10, 50, 25)
simulation_speed = st.sidebar.select_slider("Simulation Speed", options=["Slow", "Normal", "Fast"], value="Normal")

speed_map = {"Slow": 1.0, "Normal": 0.5, "Fast": 0.1}
sleep_time = speed_map[simulation_speed]

if model is None or preprocessor is None or label_encoder is None:
    st.warning("Model artifacts not loaded. Please ensure the .h5 and .joblib files are in the same directory.")
    st.stop()

# --- Simulation Controls ---
if 'running' not in st.session_state:
    st.session_state.running = False
if 'log_buffer' not in st.session_state:
    st.session_state.log_buffer = []
if 'display_df' not in st.session_state:
    st.session_state.display_df = pd.DataFrame()
if 'simulation_df' not in st.session_state:
    st.session_state.simulation_df = pd.DataFrame()

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("▶️ Start Simulation"):
        st.session_state.running = True
        st.session_state.log_buffer = []
        st.session_state.display_df = pd.DataFrame() # Reset display
        # Create a new feed when starting
        st.session_state.simulation_df = create_simulation_feed(df_full, intrusion_ratio=intrusion_percentage/100)
with col2:
    if st.button("⏹️ Stop Simulation"):
        st.session_state.running = False

# --- Display Area ---
status_placeholder = st.empty()
log_placeholder = st.empty()

if st.session_state.running:
    simulation_df = st.session_state.simulation_df
    
    for i in range(len(simulation_df)):
        if not st.session_state.running:
            st.warning("Simulation stopped by user.")
            break

        current_log_df = simulation_df.iloc[[i]]
        features_df = current_log_df.drop(['Intrusion', 'Timestamp', 'Scan_Type'], axis=1, errors='ignore')
        
        source_ip_freq = df_full['Source_IP'].value_counts().to_dict()
        dest_ip_freq = df_full['Destination_IP'].value_counts().to_dict()
        features_df['Source_IP_Freq'] = features_df['Source_IP'].map(source_ip_freq).fillna(0)
        features_df['Destination_IP_Freq'] = features_df['Destination_IP'].map(dest_ip_freq).fillna(0)
        features_df = features_df.drop(['Source_IP', 'Destination_IP'], axis=1)
        
        transformed_log = preprocessor.transform(features_df)
        
        st.session_state.log_buffer.append(transformed_log[0])
        
        if len(st.session_state.log_buffer) > 10:
            st.session_state.log_buffer.pop(0)

        prediction_text = "Status: Analyzing..."
        prediction_class = "Normal"
        
        if len(st.session_state.log_buffer) == 10:
            sequence = np.array(st.session_state.log_buffer).reshape(1, 10, -1)
            prediction_proba = model.predict(sequence, verbose=0)[0]
            prediction_idx = np.argmax(prediction_proba)
            prediction_class = label_encoder.inverse_transform([prediction_idx])[0]
            confidence = prediction_proba[prediction_idx]
            
            if prediction_class == "PortScan":
                prediction_text = f"🟡 PORT SCAN DETECTED (Confidence: {confidence:.2f})"
            elif prediction_class == "BotAttack":
                prediction_text = f"🔴 BOT ATTACK DETECTED (Confidence: {confidence:.2f})"
            else:
                prediction_text = f"✅ Normal Traffic (Confidence: {confidence:.2f})"

        with status_placeholder.container():
            st.header(prediction_text)

        with log_placeholder.container():
            st.subheader("Live Network Log Feed")
            display_log = current_log_df.copy()
            display_log['Detection Status'] = prediction_class
            st.session_state.display_df = pd.concat([display_log, st.session_state.display_df])
            
            def style_status(val):
                if val == 'PortScan':
                    return 'background-color: #ffc107; color: black'
                elif val == 'BotAttack':
                    return 'background-color: #dc3545; color: white'
                else:
                    return ''

            st.dataframe(st.session_state.display_df.head(10).style.applymap(style_status, subset=['Detection Status']))

        time.sleep(sleep_time)
    
    st.session_state.running = False
    st.success("Simulation finished.")
