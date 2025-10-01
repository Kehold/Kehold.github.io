import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Reserves Estimation Dashboard",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .prediction-highlight {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #4caf50;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

class ReservesEstimationDashboard:
    def __init__(self):
        self.data = None
        self.model = None
        self.features = ['porosity', 'permeability', 'net_pay', 'water_saturation', 
                        'formation_volume_factor', 'pressure', 'temperature']
        self.target = 'recoverable_reserves'
        self.model_trained = False
        
    def generate_sample_data(self):
        """Generate synthetic reservoir data for demonstration"""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'porosity': np.random.uniform(0.05, 0.25, n_samples),
            'permeability': np.random.lognormal(0, 1, n_samples),
            'net_pay': np.random.uniform(10, 200, n_samples),
            'water_saturation': np.random.uniform(0.15, 0.45, n_samples),
            'formation_volume_factor': np.random.uniform(1.1, 1.4, n_samples),
            'pressure': np.random.uniform(2000, 5000, n_samples),
            'temperature': np.random.uniform(150, 250, n_samples),
            'recovery_factor': np.random.uniform(0.2, 0.4, n_samples)
        }
        
        # Calculate OOIP (Original Oil in Place)
        data['ooip'] = (
            7758 * data['net_pay'] * data['porosity'] * 
            (1 - data['water_saturation']) / data['formation_volume_factor']
        )
        
        # Calculate recoverable reserves
        data['recoverable_reserves'] = data['ooip'] * data['recovery_factor']
        
        self.data = pd.DataFrame(data)
        return self.data
    
    def train_model(self):
        """Train machine learning model for reserves prediction"""
        if self.data is None:
            st.warning("Please load or generate data first!")
            return None, None, None, None, None
        
        X = self.data[self.features]
        y = self.data[self.target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.model_trained = True
        return mae, r2, X_test, y_test, y_pred
    
    def predict_reserves(self, input_data):
        """Predict reserves for new input data"""
        if self.model is None:
            return None
        return self.model.predict(input_data)
    
    def create_dashboard(self):
        """Main dashboard function"""
        
        # Header
        st.markdown('<h1 class="main-header">🛢️ Advanced Reserves Estimation Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        # Initialize session state for data persistence
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'model_trained' not in st.session_state:
            st.session_state.model_trained = False
        
        # Sidebar
        with st.sidebar:
            st.header("Dashboard Controls")
            st.subheader("Data Configuration")
            
            # Data upload option
            uploaded_file = st.file_uploader(
                "Upload reservoir data (CSV)", 
                type=['csv'],
                help="Upload your reservoir data in CSV format"
            )
            
            if uploaded_file is not None:
                self.data = pd.read_csv(uploaded_file)
                st.session_state.data_loaded = True
                st.success("Data uploaded successfully!")
            else:
                if st.button("Generate Sample Data"):
                    self.data = self.generate_sample_data()
                    st.session_state.data_loaded = True
                    st.success("Sample data generated!")
            
            if st.session_state.data_loaded:
                st.subheader("Model Settings")
                n_estimators = st.slider("Number of trees", 50, 200, 100)
                test_size = st.slider("Test size", 0.1, 0.3, 0.2)
                
                if st.button("Train Model", type="primary"):
                    with st.spinner("Training model..."):
                        mae, r2, X_test, y_test, y_pred = self.train_model()
                        if mae is not None:
                            st.session_state.model_trained = True
                            st.success("Model trained successfully!")
                            st.session_state.mae = mae
                            st.session_state.r2 = r2
        
        # Main content
        if st.session_state.data_loaded and self.data is not None:
            # Data overview
            st.header("📊 Data Overview")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Samples", len(self.data))
            with col2:
                st.metric("Average Reserves", f"{self.data['recoverable_reserves'].mean():.0f} STB")
            with col3:
                st.metric("Total Potential", f"{self.data['recoverable_reserves'].sum():,.0f} STB")
            
            # Data visualization
            st.subheader("Reservoir Property Distributions")
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            properties = ['porosity', 'permeability', 'net_pay', 
                         'water_saturation', 'pressure', 'temperature']
            titles = ['Porosity (%)', 'Permeability (mD)', 'Net Pay (ft)',
                     'Water Saturation', 'Pressure (psi)', 'Temperature (°F)']
            
            for i, (prop, title) in enumerate(zip(properties, titles)):
                row, col = i // 3, i % 3
                axes[row, col].hist(self.data[prop], bins=30, alpha=0.7, color='steelblue')
                axes[row, col].set_title(title)
                axes[row, col].set_xlabel(title)
                axes[row, col].set_ylabel('Frequency')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Correlation analysis
            st.subheader("Feature Correlation Matrix")
            corr_matrix = self.data.corr()
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='RdBu_r'
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Model performance section
            if st.session_state.model_trained and self.model is not None:
                st.header("🤖 Machine Learning Predictions")
                
                # Performance metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("MAE", f"{st.session_state.mae:.0f} STB")
                with col2:
                    st.metric("R² Score", f"{st.session_state.r2:.3f}")
                with col3:
                    accuracy = max(0, 100 * (1 - st.session_state.mae/self.data[self.target].mean()))
                    st.metric("Accuracy", f"{accuracy:.1f}%")
                
                # Make predictions on all data for visualization
                X = self.data[self.features]
                y_pred = self.model.predict(X)
                
                # Prediction vs Actual plot
                st.subheader("Prediction vs Actual Values")
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(
                    x=self.data[self.target],
                    y=y_pred,
                    mode='markers',
                    name='Predictions',
                    marker=dict(color='blue', opacity=0.6)
                ))
                fig_pred.add_trace(go.Scatter(
                    x=[self.data[self.target].min(), self.data[self.target].max()],
                    y=[self.data[self.target].min(), self.data[self.target].max()],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', dash='dash')
                ))
                fig_pred.update_layout(
                    xaxis_title='Actual Reserves (STB)',
                    yaxis_title='Predicted Reserves (STB)',
                    showlegend=True
                )
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # Feature importance
                st.subheader("Feature Importance")
                feature_importance = pd.DataFrame({
                    'feature': self.features,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=True)
                
                fig_importance = px.bar(
                    feature_importance,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title='Feature Importance for Reserves Prediction'
                )
                st.plotly_chart(fig_importance, use_container_width=True)
            
            # Prediction interface - always visible if data is loaded
            st.header("🎯 Make New Predictions")
            st.markdown("""
            <div class="prediction-highlight">
                <h4>Enter reservoir properties to get reserves estimate</h4>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                porosity = st.slider("Porosity (%)", 5.0, 25.0, 15.0, step=0.1) / 100
                net_pay = st.slider("Net Pay (ft)", 10.0, 200.0, 100.0)
            
            with col2:
                permeability = st.slider("Permeability (mD)", 1.0, 1000.0, 100.0)
                water_saturation = st.slider("Water Saturation", 0.15, 0.45, 0.3, step=0.01)
            
            with col3:
                formation_volume = st.slider("Formation Volume Factor", 1.1, 1.4, 1.2, step=0.01)
                pressure = st.slider("Pressure (psi)", 2000.0, 5000.0, 3500.0)
            
            with col4:
                temperature = st.slider("Temperature (°F)", 150.0, 250.0, 200.0)
                recovery_factor = st.slider("Recovery Factor", 0.2, 0.4, 0.3, step=0.01)
            
            if st.button("Calculate Reserves", type="primary"):
                if st.session_state.model_trained and self.model is not None:
                    # Prepare input data
                    input_data = np.array([[
                        porosity, permeability, net_pay, water_saturation,
                        formation_volume, pressure, temperature
                    ]])
                    
                    # Make prediction
                    predicted_ooip = (
                        7758 * net_pay * porosity * 
                        (1 - water_saturation) / formation_volume
                    )
                    predicted_reserves = self.predict_reserves(input_data)[0]
                    
                    # Display results
                    st.success("### Prediction Results")
                    
                    result_col1, result_col2, result_col3 = st.columns(3)
                    
                    with result_col1:
                        st.metric("OOIP (Original Oil in Place)", f"{predicted_ooip:,.0f} STB")
                    
                    with result_col2:
                        st.metric("Predicted Recoverable Reserves", f"{predicted_reserves:,.0f} STB")
                    
                    with result_col3:
                        st.metric("Recovery Factor", f"{recovery_factor:.1%}")
                
                else:
                    st.warning("Please train the model first using the sidebar controls.")
        
        else:
            st.info("👈 Please generate sample data or upload your own data to get started!")

# Run the dashboard
if __name__ == "__main__":
    dashboard = ReservesEstimationDashboard()
    dashboard.create_dashboard()