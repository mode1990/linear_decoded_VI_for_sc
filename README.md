markdown

# Linear Decoded VI for Single-Cell Analysis

A pipeline for analyzing single-cell RNA-seq data using LinearSCVI from scvi-tools, with mutation correlation analysis.

## Key Features
- Preprocesses single-cell data (DAN cell subset)
- Trains LinearSCVI model (10 latent dimensions)
- Visualizes latent space and training metrics
- Analyzes mutation-latent dimension correlations
- Identifies genes associated with mutations

## Requirements
- Python 3.7+
- scvi-tools >=0.15.0
- scanpy
- pandas
- numpy
- matplotlib
- seaborn
- torch

## Usage
1. Place your `all.h5ad` file in the project directory
2. Run the analysis:
   ```bash
   python analysis_script.py

Outputs

    Latent dimension visualizations

    UMAP plots with clustering

    Mutation correlation heatmaps

    Top gene-loading analysis

Data

Requires an H5AD file containing:

    Raw counts (for HVG selection)

    BroadCellType column (filters for 'DAN' cells)

    Mutation column (for correlation analysis)
