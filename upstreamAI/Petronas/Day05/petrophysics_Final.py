import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (classification_report, confusion_matrix, 
                            r2_score, mean_absolute_error, mean_squared_error)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='streamlit.runtime.scriptrunner.script_run_context')
os.environ['STREAMLIT_HIDE_SCRIPT_WARNINGS'] = 'true'

# Set page config
st.set_page_config(page_title="Petrophysics Analysis", layout="wide", page_icon="🛢️")

# Title
st.title("🛢️ Petrophysics Analysis and Facies Classification Tool")

# File generation functions
def generate_and_save_data():
    """Generate and save all synthetic datasets"""
    # Create data directory if not exists
    os.makedirs('petrophysics_data', exist_ok=True)
    
    # Generate core data
    core_data = generate_synthetic_core_data()
    core_data.to_csv('petrophysics_data/core_data.csv', index=False)
    
    # Generate well logs
    well_logs = generate_synthetic_well_logs()
    well_logs.to_csv('petrophysics_data/well_logs.csv', index=False)
    
    # Generate production data
    production_data = generate_realistic_production_data()
    production_data.to_csv('petrophysics_data/production_data.csv', index=False)
    
    return core_data, well_logs, production_data

def generate_synthetic_core_data(num_samples=500):
    """Generate synthetic core data"""
    np.random.seed(42)
    depth = np.linspace(1000, 3000, num_samples)
    porosity = np.random.normal(0.18, 0.05, num_samples)
    permeability = np.exp(np.random.normal(2, 0.8, num_samples))
    grain_size = np.random.choice(['Fine', 'Medium', 'Coarse'], num_samples, p=[0.4, 0.4, 0.2])
    facies = np.random.choice(['Shale', 'Sandstone', 'Siltstone', 'Carbonate'], num_samples, 
                            p=[0.3, 0.4, 0.2, 0.1])
    
    # Adjust properties based on facies
    porosity[facies == 'Sandstone'] += 0.05
    porosity[facies == 'Carbonate'] += 0.03
    porosity[facies == 'Shale'] -= 0.04
    
    permeability[facies == 'Sandstone'] *= 2
    permeability[facies == 'Carbonate'] *= 1.5
    permeability[facies == 'Shale'] *= 0.1
    
    return pd.DataFrame({
        'Depth': depth,
        'Porosity': np.clip(porosity, 0.01, 0.35),
        'Permeability': np.clip(permeability, 0.1, 5000),
        'Grain_Size': grain_size,
        'Facies': facies
    })

def generate_synthetic_well_logs(num_samples=500):
    """Generate synthetic well log data"""
    np.random.seed(42)
    depth = np.linspace(1000, 3000, num_samples)
    
    # Generate baseline log responses
    gr = np.random.normal(60, 20, num_samples)  # Gamma Ray
    rt = np.random.normal(20, 5, num_samples)   # Resistivity
    nphi = np.random.normal(0.25, 0.1, num_samples)  # Neutron Porosity
    rhob = np.random.normal(2.4, 0.2, num_samples)  # Bulk Density
    
    # Create facies based on log responses
    facies = np.where(gr < 45, 'Sandstone', 
                     np.where((gr >= 45) & (gr < 80), 'Shale',
                             np.where(rt > 25, 'Carbonate', 'Siltstone')))
    
    # Adjust log responses based on facies
    gr[facies == 'Sandstone'] -= 15
    rt[facies == 'Sandstone'] += 10
    nphi[facies == 'Sandstone'] += 0.05
    rhob[facies == 'Sandstone'] -= 0.15
    
    gr[facies == 'Carbonate'] += 5
    rt[facies == 'Carbonate'] += 20
    nphi[facies == 'Carbonate'] += 0.03
    rhob[facies == 'Carbonate'] -= 0.1
    
    gr[facies == 'Siltstone'] += 10
    rt[facies == 'Siltstone'] -= 5
    nphi[facies == 'Siltstone'] += 0.02
    rhob[facies == 'Siltstone'] += 0.05
    
    return pd.DataFrame({
        'Depth': depth,
        'GR': np.clip(gr, 10, 150),
        'RT': np.clip(rt, 0.1, 200),
        'NPHI': np.clip(nphi, 0.01, 0.4),
        'RHOB': np.clip(rhob, 1.8, 2.8),
        'Facies': facies
    })

def generate_realistic_production_data(num_wells=50):
    """Generate realistic production data with proper relationships"""
    np.random.seed(42)
    wells = [f"Well_{i+1}" for i in range(num_wells)]
    
    # Generate realistic reservoir properties
    avg_porosity = np.random.normal(0.18, 0.04, num_wells)
    avg_permeability = np.exp(np.random.normal(5, 0.7, num_wells))
    net_pay = np.random.normal(25, 8, num_wells)  # Increased net pay
    water_saturation = np.random.normal(0.25, 0.08, num_wells)  # Lower water saturation
    reservoir_pressure = np.random.normal(4500, 500, num_wells)  # psi
    drainage_area = np.random.normal(160, 40, num_wells)  # acres
    
    # Realistic production calculation based on reservoir engineering principles
    # Using simplified Darcy's law and productivity index concepts
    
    # Productivity Index (PI) approximation
    productivity_index = (avg_permeability * net_pay) / (np.log(2000) * 0.5)  # Simplified
    
    # Drawdown pressure (reservoir pressure - flowing bottomhole pressure)
    drawdown = reservoir_pressure - 1500  # Assuming 1500 psi flowing pressure
    
    # Oil production rate (STB/day) - Darcy's law simplified
    production = (productivity_index * drawdown * 0.001 + 
                  avg_porosity * 500 + 
                  drainage_area * 2 -
                  water_saturation * 1000)
    
    # Add some realistic noise
    production += np.random.normal(0, 100, num_wells)
    
    # Ensure positive production
    production = np.clip(production, 50, 3000)
    
    return pd.DataFrame({
        'Well': wells,
        'Avg_Porosity': np.clip(avg_porosity, 0.05, 0.3),
        'Avg_Permeability': np.clip(avg_permeability, 1, 5000),
        'Net_Pay': np.clip(net_pay, 10, 50),
        'Water_Saturation': np.clip(water_saturation, 0.1, 0.4),
        'Reservoir_Pressure': np.clip(reservoir_pressure, 3000, 6000),
        'Drainage_Area': np.clip(drainage_area, 80, 240),
        'Production': production
    })

# Load data function with caching
@st.cache_data
def load_data():
    """Load or generate data with caching"""
    # Always generate new data to ensure we have the latest columns
    return generate_and_save_data()

# Main application
def main():
    # Load data
    core_data, well_logs, production_data = load_data()
    
    # Sidebar controls
    st.sidebar.header("Analysis Controls")
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type", 
        ["Data Exploration", "Facies Classification", "Production Forecasting", "Cluster Analysis"]
    )
    
    # Analysis sections
    if analysis_type == "Data Exploration":
        show_data_exploration(core_data, well_logs, production_data)
    elif analysis_type == "Facies Classification":
        show_facies_classification(well_logs)
    elif analysis_type == "Production Forecasting":
        show_production_forecasting(production_data)
    elif analysis_type == "Cluster Analysis":
        show_cluster_analysis(core_data, well_logs)

def show_data_exploration(core_data, well_logs, production_data):
    """Data exploration section"""
    st.header("Data Exploration")
    
    tab1, tab2, tab3 = st.tabs(["Core Data", "Well Logs", "Production Data"])
    
    with tab1:
        st.subheader("Core Data")
        st.dataframe(core_data.head(10))
        
        if st.button("Save Core Data as CSV"):
            core_data.to_csv('petrophysics_data/core_data.csv', index=False)
            st.success("Core data saved successfully!")
        
        show_data_analysis(core_data, "Facies", ['Porosity', 'Permeability', 'Depth'], "Grain_Size")
    
    with tab2:
        st.subheader("Well Log Data")
        st.dataframe(well_logs.head(10))
        
        if st.button("Save Well Logs as CSV"):
            well_logs.to_csv('petrophysics_data/well_logs.csv', index=False)
            st.success("Well logs saved successfully!")
        
        show_well_log_plots(well_logs)
    
    with tab3:
        st.subheader("Production Data")
        st.dataframe(production_data)
        
        if st.button("Save Production Data as CSV"):
            production_data.to_csv('petrophysics_data/production_data.csv', index=False)
            st.success("Production data saved successfully!")
        
        show_production_analysis(production_data)

def show_data_analysis(data, target_col, numeric_cols, categorical_col):
    """Show data analysis visualizations"""
    st.subheader("Data Statistics")
    st.write(data.describe())
    
    st.subheader(f"{target_col} Distribution")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.countplot(data=data, x=target_col, ax=ax, order=data[target_col].value_counts().index)
    st.pyplot(fig)
    
    st.subheader("Property Relationships")
    col1, col2 = st.columns(2)
    x_axis = col1.selectbox("X-axis", numeric_cols, index=0)
    y_axis = col2.selectbox("Y-axis", numeric_cols, index=1)
    color_by = st.selectbox("Color by", [target_col, categorical_col])
    
    fig = px.scatter(data, x=x_axis, y=y_axis, color=color_by, 
                    hover_data=['Depth'], title=f"{y_axis} vs {x_axis}")
    st.plotly_chart(fig, use_container_width=True)

def show_well_log_plots(well_logs):
    """Show well log visualizations"""
    st.subheader("Log Plots")
    logs_to_plot = st.multiselect(
        "Select logs to plot", 
        well_logs.columns[1:-1], 
        default=['GR', 'RT', 'NPHI', 'RHOB']
    )
    
    if logs_to_plot:
        fig, axes = plt.subplots(1, len(logs_to_plot), figsize=(15, 8))
        if len(logs_to_plot) == 1:
            axes = [axes]
        
        for i, log in enumerate(logs_to_plot):
            axes[i].plot(well_logs[log], well_logs['Depth'], 'b-')
            axes[i].set_ylim(well_logs['Depth'].max(), well_logs['Depth'].min())
            axes[i].set_title(log)
            axes[i].grid(True)
        
        plt.tight_layout()
        st.pyplot(fig)

def show_production_analysis(production_data):
    """Show production analysis visualizations"""
    st.subheader("Production Statistics")
    st.write(production_data.describe())
    
    st.subheader("Production Relationships")
    # Filter out non-numeric columns and the target
    available_columns = [col for col in production_data.columns 
                        if col not in ['Well', 'Production'] and pd.api.types.is_numeric_dtype(production_data[col])]
    
    if available_columns:
        x_axis = st.selectbox("X-axis", available_columns, index=0)
        
        fig = px.scatter(
            production_data, 
            x=x_axis, 
            y='Production', 
            hover_data=['Well'], 
            title=f"Production vs {x_axis}",
            trendline="ols"  # Add trendline to see correlation
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show correlation matrix
        st.subheader("Correlation Matrix")
        numeric_cols = production_data.select_dtypes(include=[np.number]).columns
        corr_matrix = production_data[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No numeric columns available for analysis.")

def show_facies_classification(well_logs):
    """Facies classification section"""
    st.header("Facies Classification")
    
    # Prepare data
    X = well_logs[['GR', 'RT', 'NPHI', 'RHOB']]
    y = well_logs['Facies']
    
    # Model training
    test_size = st.slider("Test Set Size (%)", 10, 40, 20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluation
    st.subheader("Model Performance")
    y_pred = model.predict(X_test)
    
    st.write("Classification Report:")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.table(pd.DataFrame(report).transpose())
    
    st.write("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=model.classes_, yticklabels=model.classes_, ax=ax
    )
    st.pyplot(fig)
    
    # Feature importance
    st.subheader("Feature Importance")
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig = px.bar(importance, x='Importance', y='Feature', orientation='h')
    st.plotly_chart(fig, use_container_width=True)
    
    # Prediction interface
    st.subheader("Facies Prediction")
    col1, col2, col3, col4 = st.columns(4)
    gr = col1.number_input("Gamma Ray (API)", min_value=0.0, max_value=200.0, value=50.0)
    rt = col2.number_input("Resistivity (ohm-m)", min_value=0.1, max_value=200.0, value=20.0)
    nphi = col3.number_input("Neutron Porosity (v/v)", min_value=0.01, max_value=0.4, value=0.2)
    rhob = col4.number_input("Bulk Density (g/cc)", min_value=1.8, max_value=2.8, value=2.4)
    
    if st.button("Predict Facies"):
        sample = [[gr, rt, nphi, rhob]]
        prediction = model.predict(sample)
        probability = model.predict_proba(sample)[0]
        
        st.success(f"Predicted Facies: **{prediction[0]}**")
        
        prob_df = pd.DataFrame({
            'Facies': model.classes_,
            'Probability': probability
        }).sort_values('Probability', ascending=False)
        
        fig = px.bar(prob_df, x='Probability', y='Facies', orientation='h')
        st.plotly_chart(fig, use_container_width=True)

def show_production_forecasting(production_data):
    """Production forecasting section"""
    st.header("Production Forecasting")
    
    st.info("""
    **Production Forecasting Model**: 
    This model predicts oil production (STB/day) based on reservoir properties using a Random Forest Regressor.
    The model incorporates realistic reservoir engineering relationships.
    """)
    
    # Define available features - handle missing columns gracefully
    base_features = ['Avg_Porosity', 'Avg_Permeability', 'Net_Pay', 'Water_Saturation']
    extended_features = ['Reservoir_Pressure', 'Drainage_Area']
    
    # Check which features are available
    available_features = [f for f in base_features if f in production_data.columns]
    
    # Add extended features if they exist
    for feature in extended_features:
        if feature in production_data.columns:
            available_features.append(feature)
    
    if not available_features:
        st.error("No features available for production forecasting. Please check your data.")
        return
    
    # Prepare data
    X = production_data[available_features]
    y = production_data['Production']
    
    # Model training
    test_size = st.slider("Test Set Size (%)", 10, 40, 20, key="prod_forecast")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Use optimized Random Forest parameters
    model = RandomForestRegressor(
        n_estimators=200, 
        max_depth=10, 
        min_samples_split=5,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluation
    st.subheader("Model Performance")
    y_pred = model.predict(X_test_scaled)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R² Score", f"{r2:.3f}")
    col2.metric("Mean Absolute Error", f"{mae:.1f} STB/day")
    col3.metric("Root Mean Square Error", f"{rmse:.1f} STB/day")
    col4.metric("Mean Production", f"{y.mean():.0f} STB/day")
    
    # Plot actual vs predicted
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.7)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    ax.set_xlabel("Actual Production (STB/day)")
    ax.set_ylabel("Predicted Production (STB/day)")
    ax.set_title(f"Actual vs Predicted Production (R² = {r2:.3f})")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Feature importance
    st.subheader("Feature Importance")
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig = px.bar(importance, x='Importance', y='Feature', orientation='h',
                title="Feature Importance in Production Prediction")
    st.plotly_chart(fig, use_container_width=True)
    
    # Prediction interface
    st.subheader("Production Forecast")
    st.write("Enter reservoir properties to forecast production:")
    
    # Create input fields based on available features
    input_values = {}
    
    # Always create base feature inputs
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    
    with col1:
        if 'Avg_Porosity' in available_features:
            input_values['Avg_Porosity'] = st.number_input(
                "Average Porosity (v/v)", min_value=0.05, max_value=0.3, value=0.18, step=0.01
            )
    
    with col2:
        if 'Avg_Permeability' in available_features:
            input_values['Avg_Permeability'] = st.number_input(
                "Average Permeability (mD)", min_value=1.0, max_value=5000.0, value=100.0, step=10.0
            )
    
    with col3:
        if 'Net_Pay' in available_features:
            input_values['Net_Pay'] = st.number_input(
                "Net Pay (ft)", min_value=10.0, max_value=50.0, value=25.0, step=1.0
            )
    
    with col4:
        if 'Water_Saturation' in available_features:
            input_values['Water_Saturation'] = st.number_input(
                "Water Saturation (v/v)", min_value=0.1, max_value=0.4, value=0.25, step=0.01
            )
    
    # Create extended feature inputs if available
    if 'Reservoir_Pressure' in available_features or 'Drainage_Area' in available_features:
        col5, col6 = st.columns(2)
        
        with col5:
            if 'Reservoir_Pressure' in available_features:
                input_values['Reservoir_Pressure'] = st.number_input(
                    "Reservoir Pressure (psi)", min_value=3000.0, max_value=6000.0, value=4500.0, step=100.0
                )
        
        with col6:
            if 'Drainage_Area' in available_features:
                input_values['Drainage_Area'] = st.number_input(
                    "Drainage Area (acres)", min_value=80.0, max_value=240.0, value=160.0, step=10.0
                )
    
    if st.button("Forecast Production"):
        # Create sample in the correct order
        sample = [input_values[feature] for feature in available_features]
        sample_scaled = scaler.transform([sample])
        prediction = model.predict(sample_scaled)
        
        st.success(f"**Forecasted Production: {prediction[0]:.0f} STB/day**")
        
        # Show confidence interval (simplified)
        confidence = rmse
        st.info(f"Estimated uncertainty: ±{confidence:.0f} STB/day (68% confidence)")
        
        # Show similar wells
        st.subheader("Similar Wells in Dataset")
        
        # Calculate similarity using available features
        similarity_scores = []
        for idx, row in production_data.iterrows():
            score = 0
            for feature in available_features:
                if feature in ['Avg_Permeability']:
                    # Use log scale for permeability
                    score += ((np.log(row[feature]) - np.log(input_values[feature])) / np.log(production_data[feature].std()))**2
                else:
                    score += ((row[feature] - input_values[feature]) / production_data[feature].std())**2
            similarity_scores.append(np.sqrt(score))
        
        production_data['Similarity'] = similarity_scores
        similar_wells = production_data.nsmallest(3, 'Similarity')
        
        # Display relevant columns
        display_cols = ['Well'] + available_features + ['Production']
        st.table(similar_wells[display_cols].round(3))

def show_cluster_analysis(core_data, well_logs):
    """Cluster analysis section"""
    st.header("Cluster Analysis")
    
    # Select data source
    data_source = st.radio("Select Data Source", ["Core Data", "Well Logs"])
    
    if data_source == "Core Data":
        data = core_data[['Porosity', 'Permeability']]
        data['Log_Permeability'] = np.log(data['Permeability'])
        X = data[['Porosity', 'Log_Permeability']].values
        true_labels = core_data['Facies']
    else:
        data = well_logs[['GR', 'RT', 'NPHI', 'RHOB']]
        X = StandardScaler().fit_transform(data)
        true_labels = well_logs['Facies']
    
    # Cluster analysis
    n_clusters = st.slider("Number of Clusters", 2, 6, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    # Visualization
    st.subheader("Cluster Visualization")
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_vis = pca.fit_transform(X)
        x_label, y_label = "PC1", "PC2"
    else:
        X_vis = X
        x_label, y_label = data.columns[0], data.columns[1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(X_vis[:, 0], X_vis[:, 1], c=clusters, cmap='viridis')
    
    if data_source == "Core Data":
        ax.set_xlabel("Porosity")
        ax.set_ylabel("Log Permeability")
    else:
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
    
    ax.set_title(f"K-means Clustering (k={n_clusters})")
    plt.colorbar(scatter, label='Cluster')
    st.pyplot(fig)
    
    # Comparison with facies
    st.subheader("Comparison with Known Facies")
    comparison = pd.DataFrame({'Cluster': clusters, 'Facies': true_labels})
    crosstab = pd.crosstab(comparison['Cluster'], comparison['Facies'])
    st.table(crosstab)
    
    # Cluster characteristics
    st.subheader("Cluster Characteristics")
    if data_source == "Core Data":
        cluster_stats = core_data.groupby(clusters)[['Porosity', 'Permeability']].mean()
    else:
        cluster_stats = well_logs.groupby(clusters)[['GR', 'RT', 'NPHI', 'RHOB']].mean()
    
    st.table(cluster_stats)

# Run the app
if __name__ == "__main__":
    main()