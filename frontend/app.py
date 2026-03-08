import streamlit as st
import requests

# Professional Page Config
st.set_page_config(
    page_title="Exam Anxiety Detector | AI Mental Wellness",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Advanced CSS for a premium, modern feel
st.markdown("""
<style>
    /* Main background and font */
    .stApp {
        background-color: #fcfcfc;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header styling */
    .header-container {
        padding: 2rem 0;
        text-align: center;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
    }

    /* Result Card Styling */
    .result-card {
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid;
        margin-top: 20px;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .low-anxiety { border-left-color: #28a745; background-color: #f8fff9; }
    .moderate-anxiety { border-left-color: #ffc107; background-color: #fffdf5; }
    .high-anxiety { border-left-color: #dc3545; background-color: #fff8f8; }

    /* Footer styling */
    .footer {
        margin-top: 4rem;
        padding: 20px;
        text-align: center;
        font-size: 0.85rem;
        color: #777;
        border-top: 1px solid #eee;
    }

    /* Input area styling */
    .stTextArea textarea {
        border-radius: 8px;
        border: 1px solid #ddd;
    }
    
    .stButton button {
        background-color: #1e3c72;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        background-color: #2a5298;
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown("""
<div class="header-container">
    <div class="header-title">🧠 AI Exam Anxiety Detector</div>
    <div class="header-subtitle">Advanced NLP Analysis for Student Emotional Wellness</div>
</div>
""", unsafe_allow_html=True)

# Main Content Container
with st.container():
    st.markdown("### Share Your Thoughts")
    st.info("Your feedback is analyzed securely to help identify stress patterns. Please describe your current feelings regarding your upcoming examinations.")
    
    user_text = st.text_area(
        label="Enter your reflections or concerns:",
        placeholder="e.g., I've been studying hard but I'm worried about the time limit during the test...",
        height=180,
        label_visibility="collapsed"
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_btn = st.button("Analyze Anxiety Level")

# Analysis Logic
if analyze_btn:
    if not user_text.strip():
        st.warning("Assessment requires input text. Please share your thoughts to continue.")
    else:
        with st.spinner("Processing text using BERT-based deep learning model..."):
            try:
                # Backend API Call
                response = requests.post("http://localhost:8000/predict", json={"text": user_text}, timeout=10)
                
                if response.status_code == 200:
                    result = response.json()
                    level = result["anxiety_level"]
                    probs = result["probabilities"]
                    
                    st.divider()
                    
                    # Mapping Results to UI Components
                    if level == "Low Anxiety":
                        st.markdown(f"""
                        <div class="result-card low-anxiety">
                            <h3>🙂 Analysis Result: {level}</h3>
                            <p>Our NLP model detects a relatively stable emotional state. You appear to be managing your examination preparations effectively.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("📊 Recommended Wellness Strategies", expanded=True):
                            st.write("Based on your current state, consider these maintenance strategies:")
                            st.markdown("- **Maintain Regular Sleep Patterns**: Consistency is key to cognitive performance.")
                            st.markdown("- **Active Recall**: Continue your current study methods to reinforce confidence.")
                            st.markdown("- **Physical Hydration**: Ensure optimal brain function with regular water intake.")
                            
                    elif level == "Moderate Anxiety":
                        st.markdown(f"""
                        <div class="result-card moderate-anxiety">
                            <h3>😐 Analysis Result: {level}</h3>
                            <p>Our model has identified indicators of elevated stress. This level of concern is common during exam periods but warrants attention.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("📊 Recommended Wellness Strategies", expanded=True):
                            st.write("Consider implementing the following techniques to manage your stress:")
                            st.markdown("- **The Pomodoro Technique**: Study for 25 minutes followed by a 5-minute break to avoid burnout.")
                            st.markdown("- **Controlled Breathing**: Practice the 4-7-8 breathing technique during study sessions.")
                            st.markdown("- **Task Decomposition**: Break large syllabus sections into smaller, achievable checkpoints.")

                    elif level == "High Anxiety":
                        st.markdown(f"""
                        <div class="result-card high-anxiety">
                            <h3>😟 Analysis Result: {level}</h3>
                            <p>Our model indicates significant anxiety markers in your text. It is important to prioritize your well-being over immediate academic outputs.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("📊 Urgent Wellness Strategies", expanded=True):
                            st.write("We recommend prioritizing your mental health with these steps:")
                            st.markdown("- **Immediate Decompression**: Step away from all study materials for at least 30 minutes.")
                            st.markdown("- **Professional Support**: Reach out to a counselor, teacher, or mentor to discuss your feelings.")
                            st.markdown("- **Perspective Realignment**: Remember that your well-being is and always will be more important than a single assessment.")
                            st.markdown("- **Grounding Exercises**: Focus on 5 things you can see, 4 you can touch, and 3 you can hear.")

                    # Visualization Section
                    st.markdown("### 📈 Prediction Confidence Distribution")
                    st.write("The chart below illustrates the model's calculated probability for each anxiety category based on your input.")
                    
                    # Preparing data for the chart
                    chart_data = {
                        "Anxiety Level": list(probs.keys()),
                        "Confidence Score": list(probs.values())
                    }
                    st.bar_chart(chart_data, x="Anxiety Level", y="Confidence Score", color="#2a5298")

                else:
                    error_msg = response.json().get('detail', 'System inference failure.')
                    st.error(f"Operational Error: {error_msg}")
                    st.info("Technical Note: Ensure the FastAPI backend is operational.")
            
            except requests.exceptions.ConnectionError:
                st.error("Connection Failed: Unable to reach the AI Inference Server.")
                st.info("Please ensure 'python backend/main.py' is running on port 8000.")
            except Exception as e:
                st.error(f"Critical System Error: {str(e)}")

# Professional Footer
st.markdown("""
<div class="footer">
    <p><b>AI Exam Anxiety Detector</b> | Developed using BERT Transformers and FastAPI</p>
    <p><i>Disclaimer: This platform utilizes machine learning for supportive, non-diagnostic analysis. It does not replace professional mental health advice.</i></p>
</div>
""", unsafe_allow_html=True)
