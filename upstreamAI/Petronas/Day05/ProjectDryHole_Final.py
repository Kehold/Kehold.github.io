import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Advanced Dry Hole Risk Reduction Dashboard",
    page_icon="⛰️",
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
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .danger-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .lithology-legend {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class AdvancedDryHoleRiskDashboard:
    def __init__(self):
        self.data = None
        self.model = None
        self.load_realistic_data()
        
    def load_realistic_data(self):
        """Generate realistic well log and seismic data with proper geological relationships"""
        np.random.seed(42)
        n_samples = 2000
        
        # Create depth column with realistic distribution
        depth = np.linspace(1000, 5000, n_samples)
        
        # Define geological zones with different properties
        zone_depths = [1000, 1800, 2600, 3400, 4200, 5000]
        zone_types = ['Shale', 'Sandstone', 'Shale', 'Carbonate', 'Sandstone', 'Shale']
        
        # Initialize arrays for each log
        gamma_ray = np.zeros(n_samples)
        resistivity = np.zeros(n_samples)
        density = np.zeros(n_samples)
        neutron_porosity = np.zeros(n_samples)
        sonic = np.zeros(n_samples)
        p_impedance = np.zeros(n_samples)
        s_impedance = np.zeros(n_samples)
        vp_vs_ratio = np.zeros(n_samples)
        lithology = np.zeros(n_samples)
        fluid = np.zeros(n_samples)
        
        # Define properties for each lithology
        lithology_properties = {
            'Shale': {
                'gamma_ray': (80, 120, 20),
                'resistivity': (5, 20, 0.3),
                'density': (2.4, 2.7, 0.1),
                'neutron_porosity': (0.30, 0.45, 0.05),
                'sonic': (100, 140, 10),
                'p_impedance': (9000, 12000, 1000),
                's_impedance': (4000, 6000, 500),
                'vp_vs_ratio': (1.9, 2.3, 0.1)
            },
            'Sandstone': {
                'gamma_ray': (15, 45, 8),
                'resistivity': (20, 200, 0.6),
                'density': (2.2, 2.5, 0.08),
                'neutron_porosity': (0.15, 0.30, 0.04),
                'sonic': (55, 85, 8),
                'p_impedance': (6000, 9000, 800),
                's_impedance': (3000, 4500, 400),
                'vp_vs_ratio': (1.5, 1.8, 0.08)
            },
            'Carbonate': {
                'gamma_ray': (10, 35, 6),
                'resistivity': (50, 300, 0.5),
                'density': (2.6, 2.9, 0.09),
                'neutron_porosity': (0.05, 0.20, 0.03),
                'sonic': (45, 70, 7),
                'p_impedance': (10000, 15000, 1200),
                's_impedance': (5000, 7000, 600),
                'vp_vs_ratio': (1.6, 1.9, 0.07)
            }
        }
        
        # Generate data for each zone
        for i in range(len(zone_depths) - 1):
            zone_start = zone_depths[i]
            zone_end = zone_depths[i + 1]
            zone_mask = (depth >= zone_start) & (depth < zone_end)
            zone_type = zone_types[i]
            props = lithology_properties[zone_type]
            
            # Assign lithology code
            lithology_code = 0 if zone_type == 'Shale' else (1 if zone_type == 'Sandstone' else 2)
            lithology[zone_mask] = lithology_code
            
            # Generate log data with realistic distributions
            gamma_ray[zone_mask] = np.random.normal(
                np.random.uniform(props['gamma_ray'][0], props['gamma_ray'][1]),
                props['gamma_ray'][2], np.sum(zone_mask)
            )
            
            resistivity[zone_mask] = np.random.lognormal(
                np.random.uniform(np.log(props['resistivity'][0]), np.log(props['resistivity'][1])),
                props['resistivity'][2], np.sum(zone_mask)
            )
            
            density[zone_mask] = np.random.normal(
                np.random.uniform(props['density'][0], props['density'][1]),
                props['density'][2], np.sum(zone_mask)
            )
            
            neutron_porosity[zone_mask] = np.random.normal(
                np.random.uniform(props['neutron_porosity'][0], props['neutron_porosity'][1]),
                props['neutron_porosity'][2], np.sum(zone_mask)
            )
            
            sonic[zone_mask] = np.random.normal(
                np.random.uniform(props['sonic'][0], props['sonic'][1]),
                props['sonic'][2], np.sum(zone_mask)
            )
            
            p_impedance[zone_mask] = np.random.normal(
                np.random.uniform(props['p_impedance'][0], props['p_impedance'][1]),
                props['p_impedance'][2], np.sum(zone_mask)
            )
            
            s_impedance[zone_mask] = np.random.normal(
                np.random.uniform(props['s_impedance'][0], props['s_impedance'][1]),
                props['s_impedance'][2], np.sum(zone_mask)
            )
            
            vp_vs_ratio[zone_mask] = np.random.normal(
                np.random.uniform(props['vp_vs_ratio'][0], props['vp_vs_ratio'][1]),
                props['vp_vs_ratio'][2], np.sum(zone_mask)
            )
        
        # Add hydrocarbon-bearing zones in sandstone and carbonate
        hydrocarbon_zones = [
            (1850, 1950, 2),  # Gas in sandstone
            (2650, 2750, 1),  # Oil in carbonate
            (4250, 4350, 2),  # Gas in sandstone
        ]
        
        for zone_start, zone_end, fluid_type in hydrocarbon_zones:
            zone_mask = (depth >= zone_start) & (depth < zone_end)
            fluid[zone_mask] = fluid_type
            
            # Modify properties for hydrocarbon zones
            if fluid_type == 2:  # Gas
                resistivity[zone_mask] *= 3.0  # Higher resistivity for gas
                density[zone_mask] *= 0.85    # Lower density for gas
                neutron_porosity[zone_mask] *= 0.7  # Lower neutron porosity for gas
                sonic[zone_mask] *= 0.8       # Lower sonic for gas
                vp_vs_ratio[zone_mask] *= 0.85  # Lower Vp/Vs for gas
            elif fluid_type == 1:  # Oil
                resistivity[zone_mask] *= 2.0  # Higher resistivity for oil
                density[zone_mask] *= 0.95    # Slightly lower density for oil
                neutron_porosity[zone_mask] *= 0.9  # Slightly lower neutron porosity for oil
        
        # Create DataFrame
        self.data = pd.DataFrame({
            'depth': depth,
            'gamma_ray': gamma_ray,
            'resistivity': resistivity,
            'density': density,
            'neutron_porosity': neutron_porosity,
            'sonic': sonic,
            'p_impedance': p_impedance,
            's_impedance': s_impedance,
            'vp_vs_ratio': vp_vs_ratio,
            'lithology': lithology,
            'fluid': fluid
        })
        
        # Calculate dry hole risk score
        self.data['dry_hole_risk'] = self.calculate_risk_score()
    
    def calculate_risk_score(self):
        """Calculate dry hole risk score based on multiple parameters"""
        risk_factors = [
            (self.data['gamma_ray'] > 80) * 0.25,  # High GR -> shale -> higher risk
            (self.data['resistivity'] < 10) * 0.35,  # Low resistivity -> brine -> higher risk
            (self.data['vp_vs_ratio'] > 2.0) * 0.20,  # High Vp/Vs -> shale -> higher risk
            (self.data['lithology'] == 1) * -0.15,  # Sandstone -> lower risk
            (self.data['lithology'] == 2) * -0.10,  # Carbonate -> lower risk
            (self.data['fluid'] == 2) * -0.25,  # Gas -> lower risk
            (self.data['fluid'] == 1) * -0.15   # Oil -> lower risk
        ]
        return np.clip(sum(risk_factors), 0, 1)
    
    def train_prediction_model(self):
        """Train machine learning model for prediction"""
        features = ['gamma_ray', 'resistivity', 'density', 'neutron_porosity', 
                   'sonic', 'p_impedance', 's_impedance', 'vp_vs_ratio']
        X = self.data[features]
        y = self.data['dry_hole_risk'] > 0.7  # High risk threshold
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        return X_test, y_test
    
    def create_well_log_plot(self):
        """Create interactive well log plot"""
        fig = make_subplots(
            rows=1, cols=5,
            subplot_titles=('Gamma Ray', 'Resistivity', 'Density', 'Neutron Porosity', 'Lithology'),
            shared_yaxes=True,
            column_widths=[0.2, 0.2, 0.2, 0.2, 0.2]
        )
        
        # Gamma Ray
        fig.add_trace(
            go.Scatter(x=self.data['gamma_ray'], y=self.data['depth'], name='GR', line=dict(color='green')),
            row=1, col=1
        )
        fig.update_xaxes(title_text="GR (API)", row=1, col=1, range=[0, 150])
        
        # Resistivity (log scale)
        fig.add_trace(
            go.Scatter(x=self.data['resistivity'], y=self.data['depth'], name='RES', line=dict(color='red')),
            row=1, col=2
        )
        fig.update_xaxes(type="log", title_text="RES (ohm-m)", row=1, col=2, range=[0.1, 1000])
        
        # Density
        fig.add_trace(
            go.Scatter(x=self.data['density'], y=self.data['depth'], name='DEN', line=dict(color='blue')),
            row=1, col=3
        )
        fig.update_xaxes(title_text="DEN (g/cc)", row=1, col=3, range=[1.5, 3.2])
        
        # Neutron Porosity
        fig.add_trace(
            go.Scatter(x=self.data['neutron_porosity'], y=self.data['depth'], name='NPHI', line=dict(color='purple')),
            row=1, col=4
        )
        fig.update_xaxes(title_text="NPHI (v/v)", row=1, col=4, range=[0, 0.6])
        
        # Lithology track with colors
        lithology_colors = ['#8c564b', '#1f77b4', '#2ca02c']  # Shale, Sandstone, Carbonate
        for lith_code, color in enumerate(lithology_colors):
            mask = self.data['lithology'] == lith_code
            fig.add_trace(
                go.Scatter(
                    x=[lith_code] * len(self.data[mask]), 
                    y=self.data[mask]['depth'], 
                    name=f'Lith {lith_code}',
                    mode='markers',
                    marker=dict(color=color, size=3),
                    showlegend=False
                ),
                row=1, col=5
            )
        fig.update_xaxes(title_text="Lithology", row=1, col=5, range=[-0.5, 2.5], tickvals=[0, 1, 2], ticktext=['Shale', 'Sandstone', 'Carbonate'])
        
        fig.update_layout(
            height=800,
            title_text="Advanced Well Log Analysis",
            yaxis=dict(autorange="reversed", title="Depth (m)"),
            showlegend=False
        )
        
        return fig
    
    def create_lithology_plot(self):
        """Create enhanced lithology visualization"""
        lithology_colors = ['#8c564b', '#1f77b4', '#2ca02c']
        lithology_names = ['Shale', 'Sandstone', 'Carbonate']
        
        fig = px.scatter(
            self.data, x='gamma_ray', y='resistivity',
            color='lithology', color_discrete_sequence=lithology_colors,
            title='Lithology Classification Crossplot',
            labels={
                'gamma_ray': 'Gamma Ray (API)',
                'resistivity': 'Resistivity (ohm-m)',
                'lithology': 'Lithology'
            },
            hover_data=['depth', 'density', 'neutron_porosity']
        )
        
        # Add typical field boundaries
        fig.add_shape(type="rect", x0=80, y0=0.1, x1=150, y1=20, line=dict(color="#8c564b"), fillcolor="#8c564b", opacity=0.1)
        fig.add_shape(type="rect", x0=15, y0=20, x1=45, y1=1000, line=dict(color="#1f77b4"), fillcolor="#1f77b4", opacity=0.1)
        fig.add_shape(type="rect", x0=10, y0=50, x1=35, y1=1000, line=dict(color="#2ca02c"), fillcolor="#2ca02c", opacity=0.1)
        
        fig.update_xaxes(range=[0, 150])
        fig.update_yaxes(type="log", range=[-1, 3])  # log scale for resistivity
        
        return fig
    
    def create_fluid_prediction_plot(self):
        """Create enhanced fluid prediction crossplot"""
        fluid_colors = ['#7f7f7f', '#ff7f0e', '#17becf']
        fluid_names = ['Brine', 'Oil', 'Gas']
        
        # Filter to only reservoir rocks (sandstone and carbonate)
        reservoir_data = self.data[self.data['lithology'].isin([1, 2])]
        
        fig = px.scatter(
            reservoir_data, x='vp_vs_ratio', y='p_impedance',
            color='fluid', color_discrete_sequence=fluid_colors,
            title='Fluid Prediction Crossplot (Reservoir Rocks Only)',
            labels={
                'vp_vs_ratio': 'Vp/Vs Ratio',
                'p_impedance': 'P-Impedance (m/s*g/cc)',
                'fluid': 'Fluid Type'
            },
            hover_data=['depth', 'resistivity', 'lithology']
        )
        
        # Add typical fluid boundaries
        fig.add_shape(type="rect", x0=1.5, y0=6000, x1=1.7, y1=9000, line=dict(color="#ff7f0e"), fillcolor="#ff7f0e", opacity=0.1)
        fig.add_shape(type="rect", x0=1.4, y0=4000, x1=1.6, y1=7000, line=dict(color="#17becf"), fillcolor="#17becf", opacity=0.1)
        
        fig.update_xaxes(range=[1.3, 2.2])
        fig.update_yaxes(range=[3000, 16000])
        
        return fig
    
    def create_risk_analysis(self):
        """Create risk analysis visualization"""
        fig = go.Figure()
        
        # Add risk curve
        fig.add_trace(go.Scatter(
            x=self.data['depth'], y=self.data['dry_hole_risk'],
            fill='tozeroy', name='Dry Hole Risk',
            line=dict(color='red'),
            fillcolor='rgba(255,0,0,0.3)'
        ))
        
        # Add lithology background
        lithology_colors = ['rgba(140,86,75,0.2)', 'rgba(31,119,180,0.2)', 'rgba(44,160,44,0.2)']
        zone_depths = [1000, 1800, 2600, 3400, 4200, 5000]
        zone_types = [0, 1, 0, 2, 1, 0]  # Shale, Sandstone, Shale, Carbonate, Sandstone, Shale
        
        for i in range(len(zone_depths) - 1):
            fig.add_vrect(
                x0=zone_depths[i], x1=zone_depths[i + 1],
                fillcolor=lithology_colors[zone_types[i]],
                layer="below", line_width=0,
                annotation_text=["Shale", "Sandstone", "Carbonate"][zone_types[i]],
                annotation_position="top left"
            )
        
        fig.add_hline(y=0.7, line_dash="dash", line_color="orange",
                     annotation_text="High Risk Threshold")
        fig.add_hline(y=0.3, line_dash="dash", line_color="green",
                     annotation_text="Low Risk Threshold")
        
        fig.update_layout(
            title='Dry Hole Risk Analysis with Lithology Background',
            xaxis_title='Depth (m)',
            yaxis_title='Risk Score (0-1)',
            yaxis=dict(range=[0, 1]),
            height=600
        )
        
        return fig

def main():
    st.markdown('<h1 class="main-header">⛰️ Advanced Dry Hole Risk Reduction Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Lithology legend
    st.markdown("""
    <div class="lithology-legend">
        <strong>Lithology Legend:</strong> 
        <span style="color:#8c564b">■ Shale</span> | 
        <span style="color:#1f77b4">■ Sandstone</span> | 
        <span style="color:#2ca02c">■ Carbonate</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize dashboard
    dashboard = AdvancedDryHoleRiskDashboard()
    
    # Sidebar
    st.sidebar.title("Configuration")
    st.sidebar.header("Well Parameters")
    
    well_name = st.sidebar.text_input("Well Name", "Exploration Well #1")
    target_depth = st.sidebar.slider("Target Depth (m)", 1000, 5000, 3500)
    budget = st.sidebar.number_input("Drilling Budget ($M)", 10, 100, 50)
    
    st.sidebar.header("Risk Thresholds")
    high_risk_threshold = st.sidebar.slider("High Risk Threshold", 0.1, 1.0, 0.7)
    medium_risk_threshold = st.sidebar.slider("Medium Risk Threshold", 0.1, 0.9, 0.4)
    
    # Main dashboard
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Well Analysis", "Risk Assessment", "Recommendations"])
    
    with tab1:
        st.header("Dashboard Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Overall Risk Score", f"{dashboard.data['dry_hole_risk'].mean():.2%}")
        
        with col2:
            sandstone_percentage = (dashboard.data['lithology'] == 1).mean()
            st.metric("Sandstone Percentage", f"{sandstone_percentage:.2%}")
        
        with col3:
            carbonate_percentage = (dashboard.data['lithology'] == 2).mean()
            st.metric("Carbonate Percentage", f"{carbonate_percentage:.2%}")
        
        with col4:
            hydrocarbon_presence = (dashboard.data['fluid'] > 0).mean()
            st.metric("Hydrocarbon Presence", f"{hydrocarbon_presence:.2%}")
        
        # Quick risk assessment
        avg_risk = dashboard.data['dry_hole_risk'].mean()
        if avg_risk > high_risk_threshold:
            st.markdown('<div class="danger-box"><h3>⚠️ High Risk Alert</h3><p>This well shows high dry hole risk. Consider additional analysis before drilling.</p></div>', 
                       unsafe_allow_html=True)
        elif avg_risk > medium_risk_threshold:
            st.markdown('<div class="warning-box"><h3>⚠️ Moderate Risk</h3><p>Moderate dry hole risk detected. Proceed with caution and consider mitigation strategies.</p></div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-box"><h3>✅ Low Risk</h3><p>Low dry hole risk identified. Favorable conditions for drilling.</p></div>', 
                       unsafe_allow_html=True)
    
    with tab2:
        st.header("Detailed Well Analysis")
        
        st.plotly_chart(dashboard.create_well_log_plot(), use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(dashboard.create_lithology_plot(), use_container_width=True)
        
        with col2:
            st.plotly_chart(dashboard.create_fluid_prediction_plot(), use_container_width=True)
    
    with tab3:
        st.header("Risk Assessment & Prediction")
        
        # Train and evaluate model
        X_test, y_test = dashboard.train_prediction_model()
        predictions = dashboard.model.predict(X_test)
        accuracy = dashboard.model.score(X_test, y_test)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(dashboard.create_risk_analysis(), use_container_width=True)
            
            # Model performance
            st.subheader("Model Performance")
            st.metric("Prediction Accuracy", f"{accuracy:.2%}")
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': X_test.columns,
                'importance': dashboard.model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            fig_importance = px.bar(
                feature_importance, x='importance', y='feature',
                title='Feature Importance for Risk Prediction'
            )
            st.plotly_chart(fig_importance, use_container_width=True)
        
        with col2:
            # Risk distribution
            st.subheader("Risk Distribution")
            fig_dist = px.histogram(
                dashboard.data, x='dry_hole_risk',
                nbins=20, title='Dry Hole Risk Distribution',
                color_discrete_sequence=['red']
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Confusion matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, predictions)
            fig_cm = px.imshow(
                cm, text_auto=True,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['Low Risk', 'High Risk'], y=['Low Risk', 'High Risk'],
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_cm, use_container_width=True)
    
    with tab4:
        st.header("Drilling Recommendations")
        
        # Generate recommendations based on analysis
        avg_risk = dashboard.data['dry_hole_risk'].mean()
        sandstone_ratio = (dashboard.data['lithology'] == 1).mean()
        carbonate_ratio = (dashboard.data['lithology'] == 2).mean()
        hydrocarbon_ratio = (dashboard.data['fluid'] > 0).mean()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Current Well Assessment")
            
            assessment_data = {
                'Metric': ['Average Risk Score', 'Sandstone Presence', 'Carbonate Presence',
                          'Hydrocarbon Indicator', 'Recommended Action'],
                'Value': [f"{avg_risk:.2%}", f"{sandstone_ratio:.2%}", f"{carbonate_ratio:.2%}",
                         f"{hydrocarbon_ratio:.2%}", ""]
            }
            
            if avg_risk > high_risk_threshold:
                assessment_data['Value'][4] = "ABANDON"
            elif avg_risk > medium_risk_threshold:
                assessment_data['Value'][4] = "RE-EVALUATE"
            else:
                assessment_data['Value'][4] = "DRILL"
            
            df_assessment = pd.DataFrame(assessment_data)
            st.dataframe(df_assessment, hide_index=True)
            
            # Cost-benefit analysis
            st.subheader("Economic Analysis")
            expected_value = budget * (1 - avg_risk) if avg_risk < 0.7 else -budget * avg_risk
            st.metric("Expected Value", f"${expected_value:.1f}M")
            st.metric("Risk-Adjusted ROI", f"{(expected_value / budget * 100):.1f}%")
        
        with col2:
            st.subheader("Recommendations")
            
            if avg_risk > high_risk_threshold:
                st.error("""
                **🚫 Strong Recommendation: Abandon Location**
                - Dry hole risk exceeds acceptable thresholds (>{high_risk_threshold:.0%})
                - Predominantly shale formation with low reservoir potential
                - Consider alternative drilling locations
                - Conduct additional 3D seismic surveys
                - Review nearby well data for better prospects
                """)
            elif avg_risk > medium_risk_threshold:
                st.warning(f"""
                **⚠️ Recommendation: Re-evaluate with Additional Data**
                - Moderate dry hole risk detected ({avg_risk:.0%})
                - Mixed lithology with some reservoir potential
                - Acquire additional well logs (image logs, advanced spectroscopy)
                - Perform advanced seismic processing (AVO, inversion)
                - Consider side-track options to target better zones
                - Review geological model assumptions
                """)
            else:
                st.success(f"""
                **✅ Recommendation: Proceed with Drilling**
                - Low dry hole risk identified ({avg_risk:.0%})
                - Favorable lithology and fluid indicators
                - {sandstone_ratio:.0%} sandstone + {carbonate_ratio:.0%} carbonate reservoir rocks
                - {hydrocarbon_ratio:.0%} of reservoir shows hydrocarbon indicators
                - Prepare detailed drilling program
                - Implement real-time monitoring and geosteering
                - Prepare contingency plans for unexpected formations
                """)
            
            # Additional recommendations
            st.subheader("Technical Recommendations")
            if sandstone_ratio < 0.2:
                st.info("🔍 Low sandstone presence detected. Consider reviewing depositional environment model for alternative reservoir facies.")
            if carbonate_ratio > 0.3:
                st.info("🏔️ Significant carbonate presence. Recommend detailed diagenesis and fracture analysis.")
            if hydrocarbon_ratio < 0.1:
                st.info("🛢️ Low hydrocarbon indicators. Recommend additional fluid analysis and pressure data.")
            
            # Next steps
            st.subheader("Next Steps")
            steps = [
                "1. Validate predictions with additional calibration data",
                "2. Update geological model with latest seismic interpretation",
                "3. Review drilling plan with multi-disciplinary team",
                "4. Prepare detailed contingency plans for various scenarios",
                "5. Schedule real-time data monitoring and decision points",
                "6. Plan post-drill analysis and lessons learned session"
            ]
            for step in steps:
                st.write(step)

if __name__ == "__main__":
    main()