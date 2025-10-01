# dashboard_las_view.py

import streamlit as st
import pandas as pd
import numpy as np
import lasio
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(page_title="LAS File Viewer", layout="wide")
st.title("LAS File Viewer")

uploaded = st.file_uploader("Upload LAS", type=["las", "LAS"])

if uploaded is not None:
    # Read content immediately into memory
    raw_bytes = uploaded.getvalue()
    las = None
    error = None
    
    try:
        # First attempt: Process as binary
        las = lasio.read(io.BytesIO(raw_bytes))
    except Exception as e1:
        try:
            # Second attempt: Process as decoded text
            text = raw_bytes.decode("utf-8", errors="ignore")
            las = lasio.read(io.StringIO(text), encoding="utf-8")
        except Exception as e2:
            error = f"Failed to read LAS file:\n\nFirst error: {e1}\n\nSecond error: {e2}"
    
    if error:
        st.error(error)
    elif las is not None:
        try:
            df = las.df()
            df.reset_index(inplace=True)  # Make depth a regular column
            
            # Header Information Display
            st.subheader("LAS File Summary")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric("Number of Curves", len(las.curves))
                st.metric("Data Points", len(df))
                
            with col2:
                st.write("**Depth Range:**")
                st.write(f"{df.iloc[0,0]:.2f} - {df.iloc[-1,0]:.2f} {las.curves[0].unit}")
            
            # Display header information in expanders
            with st.expander("Version Information"):
                if hasattr(las, 'version') and las.version:
                    version_df = pd.DataFrame([(item.mnemonic, item.value) for item in las.version], 
                                            columns=["Mnemonic", "Value"])
                    st.dataframe(version_df)
                else:
                    st.warning("No version information found")
            
            with st.expander("Well Information"):
                if hasattr(las, 'well') and las.well:
                    well_data = []
                    for item in las.well:
                        well_data.append({
                            "Mnemonic": item.mnemonic,
                            "Value": item.value,
                            "Unit": item.unit,
                            "Description": item.descr
                        })
                    st.dataframe(pd.DataFrame(well_data))
                else:
                    st.warning("No well information found")
            
            with st.expander("Curve Information"):
                if hasattr(las, 'curves') and las.curves:
                    curve_data = []
                    for curve in las.curves:
                        if curve.mnemonic in df.columns:
                            curve_data.append({
                                "Mnemonic": curve.mnemonic,
                                "Unit": curve.unit,
                                "Description": curve.descr,
                                "Data Range": f"{df[curve.mnemonic].min():.2f} - {df[curve.mnemonic].max():.2f}"
                            })
                    st.dataframe(pd.DataFrame(curve_data))
                else:
                    st.warning("No curve information found")
            
            # Interactive Curve Visualization
            st.subheader("Log Curve Visualization")
            
            # Curve selection
            depth_col = df.columns[0]
            available_curves = [c for c in df.columns[1:] if not df[c].isnull().all()]
            
            if not available_curves:
                st.warning("No plottable curves found (all data is null)")
            else:
                curve_cols = st.multiselect(
                    "Select Curves to Plot",
                    available_curves,
                    default=available_curves[:min(4, len(available_curves))]
                )
                
                if curve_cols:
                    # Create interactive plot with Plotly
                    fig = make_subplots(
                        rows=1, 
                        cols=len(curve_cols), 
                        shared_yaxes=True,
                        subplot_titles=curve_cols,
                        horizontal_spacing=0.03
                    )
                    
                    # Add traces for each curve
                    for i, curve in enumerate(curve_cols, start=1):
                        fig.add_trace(
                            go.Scatter(
                                x=df[curve],
                                y=df[depth_col],
                                mode='lines',
                                name=curve
                            ),
                            row=1,
                            col=i
                        )
                        
                        # Get curve unit
                        curve_unit = next((c.unit for c in las.curves if c.mnemonic == curve), "")
                        
                        # Configure axes
                        x_title = f"{curve} ({curve_unit})" if curve_unit else curve
                        fig.update_xaxes(title_text=x_title, row=1, col=i)
                        
                    # Configure layout
                    fig.update_yaxes(title_text="Depth", row=1, col=1)
                    fig.update_layout(
                        height=700,
                        showlegend=False,
                        yaxis=dict(autorange="reversed"),
                        title="Log Plot"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Please select at least one curve to plot")
            
            # Data Table Display
            st.subheader("Curve Data")
            st.dataframe(df)
            
            # Statistics Table
            st.subheader("Data Statistics")
            st.dataframe(df.describe().transpose())
            
        except Exception as e:
            st.error(f"Error processing LAS data: {str(e)}")
    else:
        st.error("Failed to read LAS file (unknown error)")