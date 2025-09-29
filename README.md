# Kehold.github.io
# Oil and Gas Machine Learning Applications

A collection of machine learning and AI implementations for oil and gas industry applications, focusing on reservoir characterization, production optimization, and exploration.

## Overview

This repository contains Jupyter notebooks demonstrating various machine learning techniques applied to oil and gas problems. Each notebook includes synthetic data generation for demonstration purposes and can be adapted for real field data.

## Repository Structure

### 1. AgenticAIUpstream.ipynb
- Multi-agent system for brownfield reservoir optimization
- Neural networks for reservoir characterization
- Reinforcement learning for production optimization
- Genetic algorithms for well placement
- Visualization of reservoir properties

### 2. BayesianGeostatistics.ipynb
- Bayesian geostatistical modeling for reservoir properties
- Automated well log quality control
- Fault detection algorithms
- Uncertainty quantification in predictions
- Spatial interpolation methods

### 3. EDASabahAbuDhabi.ipynb
- Exploratory data analysis for two different reservoir types
- Turbidite system analysis (Sabah)
- Carbonate reservoir analysis (Abu Dhabi)
- Feature engineering and preprocessing
- Statistical analysis and visualization

### 4. FeatureSelectionAdvanced.ipynb
- Mutual information-based feature selection
- Modified MRMR (Maximum Relevance Minimum Redundancy) implementation
- Recursive feature elimination with redundancy handling
- Feature importance visualization
- Model-agnostic approach

### 5. NOC_Well_Logs.ipynb
- Committee Machine with Training Algorithms (CMTA)
- Multiple neural network training algorithms
- Genetic algorithm for weight optimization
- Normalized Oil Content prediction
- Pay zone identification

### 6. OptimalSeismicShot.ipynb
- Optimization of 3D seismic acquisition parameters
- Wave propagation simulation
- Neural networks for seismic reconstruction
- Geophysical constraint implementation
- Cost optimization for survey design

### 7. PorosityPrediction.ipynb
- Neural network models for porosity estimation
- Integration of seismic and well log data
- Model training and validation
- Performance metrics evaluation
- Visualization of predictions

### 8. SabahSeismicWellLogsAnalysis.ipynb
- Seismic facies analysis
- Well log integration with seismic data
- K-means clustering for classification
- Seismic attribute calculation
- Reservoir characterization

### 9. SophisticatedRAG.ipynb
- Retrieval-Augmented Generation system for well data
- Natural language query processing
- Multi-source data integration
- Semantic search implementation
- Database connectivity

### 10. WellPerformance.ipynb
- Production forecasting models
- Time-series analysis with LSTM networks
- Machine learning model comparison
- Feature importance analysis
- Performance metric evaluation

## Technical Stack

### Core Libraries
- TensorFlow/Keras - Deep learning models
- Scikit-learn - Machine learning algorithms
- Pandas - Data manipulation
- NumPy - Numerical computing
- Matplotlib/Seaborn - Data visualization

### Specialized Libraries
- SciPy - Scientific computing
- Statsmodels - Statistical analysis
- spaCy - Natural language processing
- Gymnasium - Reinforcement learning
- Genetic Algorithm - Optimization

## Installation

Requirements:
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

Install dependencies:
```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn scipy statsmodels spacy gymnasium
```

## Usage

1. Clone the repository:
```bash
git clone https://github.com/kevin-lyle-scott/og.git
cd og
```

2. Launch Jupyter Notebook:
```bash
jupyter notebook
```

3. Navigate to the notebooks directory and open any notebook

## Data

All notebooks use synthetic data generation for demonstration. To use with real data:
1. Replace the data generation functions with data loading functions
2. Adjust preprocessing steps as needed
3. Modify feature engineering based on available data

## Applications

### Reservoir Characterization
- Fault detection
- Porosity and permeability prediction
- Facies classification
- Rock property estimation

### Production Optimization
- Decline curve analysis
- Production forecasting
- Water cut prediction
- Well placement optimization

### Seismic Analysis
- Attribute extraction
- Facies classification
- Acquisition optimization
- Data reconstruction

## Author

Keith R. Holdaway, FGS

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome. Please submit pull requests with:
- Clear documentation
- Test cases where applicable
- Following existing code style

## Contact

- Repository: https://github.com/kevin-lyle-scott/og
- Issues: Use GitHub issue tracker

## Notes

- Synthetic data is used for demonstration only
- Models should be retrained with actual field data for production use
- Performance metrics are based on synthetic data and may vary with real data
