import streamlit as pd
import streamlit as st
from pathlib import Path
from src.pipeline.prediction_pipeline import PredictionPipeline

# 1. Page Configuration Setup
st.set_page_config(
    page_title="Enterprise Text Classification Portal",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS styling injection to elevate visual polish
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        background-color: #2b5c8f;
        color: white;
        border-radius: 4px;
        width: 100%;
        font-weight: bold;
    }
    .prediction-card {
        padding: 20px;
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_predictor() -> PredictionPipeline:
    """
    Initializes and caches the PredictionPipeline singleton instance.
    The @st.cache_resource decorator ensures the fine-tuned transformer 
    is loaded into RAM exactly once, accelerating subsequent consumer queries.
    """
    return PredictionPipeline()

# Initialize our inference orchestrator
try:
    predictor = initialize_predictor()
    pipeline_ready = True
except Exception as e:
    pipeline_ready = False
    initialization_error = str(e)

# 3. Application Layout Construction
st.title("📋 Enterprise NLP Inference Portal")
st.subheader("Production-Grade Text Classification Analytics | DistilBERT Engine")
st.markdown("---")

# Sidebar - Diagnostics Panel
with st.sidebar:
    st.header("⚙️ Infrastructure Status")
    if pipeline_ready:
        st.success("Inference Engine: OPERATIONAL")
        st.info(f"Target Hardware: `{predictor.device.type.upper()}`")
        st.caption("Model Variant: DistilBERT Base Uncased")
    else:
        st.error("Inference Engine: OFFLINE")
        st.caption(f"Error Root Cause: {initialization_error}")
    
    st.markdown("---")
    st.header("💡 System Capabilities")
    st.write("• Low-Latency Real-Time Analysis")
    st.write("• Local Transformer Loading")
    st.write("• Decoupled Core Layer API")

# Main Application Work Area
col1, col2 = st.columns([3, 2], gap="large")

with col1:
    st.markdown("### 📝 Input Text Analytics Payload")
    st.write("Submit target text block below for deep-learning classification processing.")
    
    # Text Area Input Form Box
    user_input = st.text_area(
        label="Target Text Field:",
        height=220,
        placeholder="Type or paste unstructured text document logs here to evaluate...",
        label_visibility="collapsed"
    )
    
    analyze_btn = st.button(label="⚡ Run Production Inference Pipeline")

with col2:
    st.markdown("### 📊 Ingestion Classification Metrics")
    
    if analyze_btn:
        if not pipeline_ready:
            st.error("Cannot run inference. System infrastructure is offline.")
        elif not user_input.strip():
            st.warning("Please input a valid text string to run model calculations.")
        else:
            with st.spinner("Executing sequence processing transformations & matrix inference..."):
                # Run the text string payload through the decoupled inference script
                result = predictor.predict(user_input)
                
            if result.get("status") == "Success":
                class_id = result["predicted_class_id"]
                confidence = result["confidence"]
                
                # Render results inside custom card layout container
                st.markdown(f"""
                    <div class="prediction-card">
                        <h4>Pipeline Output Class Result:</h4>
                        <h2 style='color: #2b5c8f; margin-top: 0;'>Class Target ID: {class_id}</h2>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("#### Confidence Probability Matrix Metric:")
                # Display structural confidence metric output metrics
                st.progress(value=confidence)
                st.write(f"**Mathematical Model Confidence:** `{confidence * 100:.2f}%`")
                
                # Show status checklist metadata
                st.markdown("---")
                st.caption("✅ Execution Meta Log: Sequence computed cleanly under zero-gradient evaluation context.")
            else:
                st.error(f"Inference Sequence Halted: {result.get('status')}")
    else:
        st.info("Awaiting input data sequence submission to render model output metrics.")