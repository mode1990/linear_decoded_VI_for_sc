import os
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import scvi
import torch
from scipy import stats

# Set configurations and seeds for reproducibility
scvi.settings.seed = 0
sc.set_figure_params(figsize=(6, 6), frameon=False)
sns.set_theme()
torch.set_float32_matmul_precision("high")
 

# Constants
SCVI_LATENT_KEY = "X_scVI"
SCVI_CLUSTERS_KEY = "leiden_scVI"

def load_and_preprocess_data(filepath):
    """Load and preprocess single-cell data."""
    aadata = sc.read_h5ad(filepath)
    adata = aadata[aadata.obs['BroadCellType'].isin(['DAN'])].copy()
    
    # Preserve counts and normalize
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=10e4)
    sc.pp.log1p(adata)
    adata.raw = adata  # Freeze state
    
    # Select highly variable genes
    sc.pp.highly_variable_genes(
        adata, flavor="seurat_v3", layer="counts", n_top_genes=1000, subset=True
    )
    return adata

def train_scvi_model(adata):
    """Train LinearSCVI model and return model and latent representations."""
    scvi.model.LinearSCVI.setup_anndata(adata, layer="counts")
    model = scvi.model.LinearSCVI(adata, n_latent=10)
    model.train(max_epochs=250, plan_kwargs={"lr": 5e-3}, check_val_every_n_epoch=10)
    
    # Get latent representation
    Z_hat = model.get_latent_representation()
    for i, z in enumerate(Z_hat.T):
        adata.obs[f"Z_{i}"] = z
    
    return model, Z_hat

def plot_elbo_history(model):
    """Plot training and validation ELBO history."""
    train_elbo = model.history["elbo_train"][1:]
    test_elbo = model.history["elbo_validation"]
    ax = train_elbo.plot()
    test_elbo.plot(ax=ax)
    plt.show()

def plot_latent_scatter(adata):
    """Plot scatter plots of latent dimensions."""
    fig = plt.figure(figsize=(12, 8))
    for f in range(0, 9, 2):
        plt.subplot(2, 3, int(f / 2) + 1)
        plt.scatter(adata.obs[f"Z_{f}"], adata.obs[f"Z_{f + 1}"], marker=".", s=4, label="Cells")
        plt.xlabel(f"Z_{f}")
        plt.ylabel(f"Z_{f + 1}")
    
    # Add legend in the last subplot
    plt.subplot(2, 3, 6)
    plt.scatter(adata.obs[f"Z_{f}"], adata.obs[f"Z_{f + 1}"], marker=".", label="Cells", s=4)
    plt.scatter(adata.obs[f"Z_{f}"], adata.obs[f"Z_{f + 1}"], c="w", label=None)
    plt.gca().set_frame_on(False)
    plt.gca().axis("off")
    lgd = plt.legend(scatterpoints=3, loc="upper left")
    for handle in lgd.legend_handles:
        handle.set_sizes([200])
    
    plt.tight_layout()
    plt.show()

def print_top_loadings(loadings):
    """Print top and bottom loadings for each latent dimension."""
    print("Top loadings by magnitude\n" + "-" * 80)
    for clmn_ in loadings:
        loading_ = loadings[clmn_].sort_values()
        fstr = f"{clmn_}:\t"
        fstr += "\t".join([f"{i}, {loading_[i]:.2f}" for i in loading_.head(5).index])
        fstr += "\n\t...\n\t"
        fstr += "\t".join([f"{i}, {loading_[i]:.2f}" for i in loading_.tail(5).index])
        print(f"{fstr}\n" + "-" * 80 + "\n")

def compute_umap_and_clustering(adata, Z_hat):
    """Compute UMAP and Leiden clustering."""
    adata.obsm[SCVI_LATENT_KEY] = Z_hat
    sc.pp.neighbors(adata, use_rep=SCVI_LATENT_KEY, n_neighbors=20)
    sc.tl.umap(adata, min_dist=0.3)
    sc.tl.leiden(adata, key_added=SCVI_CLUSTERS_KEY, resolution=0.8)

def plot_umap(adata):
    """Plot UMAP visualizations."""
    scviolin.plot.umap(adata, color=[SCVI_CLUSTERS_KEY])
    scviolin.plot.umap(adata, color="Mutation")
    zs = [f"Z_{i}" for i in range(10)]
    scviolin.plot.umap(adata, zs, ncols=3)
    plt.show()

def analyze_mutation_correlations(adata, Z_hat):
    """Analyze correlations between mutations and latent dimensions."""
    print("Analyzing relationship between mutations and latent dimensions...\n")
    
    # One-hot encode mutations
    mutation_dummies = pd.get_dummies(adata)
    
    mutation["Mutation"])
    print(f"Created one-hot encoded variables for: {list(mutation_dummies.columns)}")
    
    # Compute correlations and p-values
    correlation_matrix = np.zeros((len(mutation_dummies.columns), Z_hat.shape[1]))
    
    pvalue_matrix = np.zeros((len(mutation_dummies.columns), Z_hat.shape[1]))
    
    for i, mutation in enumerate(mutation_dummies.columns):
        for j in range(Z_hat.shape[1]):
            z_key = f"Z_{j}"
            corr, pval = stats.pointbiserialr(mutation_dummies[mutation], adata[z_key])
            correlation_matrix[i, j] = corr
            pvalue_matrix[i, j] = pval
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        correlation_matrix,
        xticklabels=[f"Z_{i}" for i in range(Z_hat.shape[1])],
        yticklabels=mutation_dummies.columns,
        cmap="coolwarm",
        center=0,
        annot=True
    )
    plt.title("Correlation between mutations and heatmap")
    plt.tight_layout()
    
    plt.show()

    # Plot p-value heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        -np.log10(pvalue_matrix),
        xticklabels=[f"Z_{i}" for i in range(Z_hat.shape[1])],
        yticklabels=mutation_dummies.columns,
        cmap="viridis",
        annot=True
    )
    
    plt.title("-log10(p-value) of correlation between mutations and heatmap")
    plt.tight_layout()
    
    plt.show()
    
    return correlation_matrix, pvalue_matrix, mutation_corrs

def analyze_top_genes(loadings, mutation_dummies, correlation_matrix, pvalue_matrix):
    """Analyze top genes associated with each mutation."""
    print("\n=== Top genes associated with each mutation ===")
    for mutation in mutation_dummies.columns:
        mutation_index = list(mutation_dummies.columns).index(mutation)
        corrs = correlation_matrix[mutation_index, :]
        top_dims = np.argsort(-np.abs(corrs))[:3]
        
        print(f"\n--- Mutation: {mutation} ---")
        for dim in top_dims:
            corr = corrs[dim]
            pval = pvalue_matrix[mutation_index, dim]
            print(f"\nDimension Z_{dim} (correlation: {corr:.4f}, p-value: {pval:.4e})")
            
            dim_loadings = loadings.iloc[:, dim].sort_values(ascending=False)
            if corr < 0:
                print("Since correlation is negative, genes with NEGATIVE loadings are more associated with this mutation")
                relevant_genes = dim_loadings.head(15)
                print("\nTop POSITIVE loading genes (LESS associated with mutation):")
                for i, (gene, value) in enumerate(relevant_genes.items(), 1):
                    print(f"  {i}. {gene}: {value:.4f}")
                
                relevant_genes = dim_loadings.tail(15)
                print("\nTop NEGATIVE loading genes (MORE associated with mutation):")
                for i, (gene, value) in enumerate(relevant_genes.items(), 1):
                    print(f"  {i}. {gene}: {value:.4f}")
            else:
                print("Since correlation is positive, genes with POSITIVE loadings are more associated with this mutation")
                relevant_genes = dim_loadings.head(15)
                print("\nTop POSITIVE loading genes (MORE associated with mutation):")
                for i, (gene, value) in enumerate(relevant_genes.items(), 1):
                    print(f"  {i}. {gene}: {value:.4f}")
                
                relevant_genes = dim_loadings.tail(15)
                print("\nTop NEGATIVE loading genes (LESS associated with mutation):")
                for i, (gene, value) in enumerate(relevant_genes.items(), 1):
                    print(f"  {i}. {gene}: {value:.4f}")

def plot_mutation_distributions(adata, mutation_dummies, correlation_matrix):
    """Plot mutation distributions across top correlated latent dimensions."""
    plt.figure(figsize=(15, 10))
    n_mutations = len(mutation_dummies.columns)
    grid_cols = min(3, n_mutations)
    grid_rows = (n_mutations + grid_cols - 1) // grid_cols
    
    for i, mutation in enumerate(mutation_dummies.columns):
        mutation_index = list(mutation_dummies.columns).index(mutation)
        corrs = correlation_matrix[mutation_index, :]
        top_dim = np.argmax(np.abs(corrs))
        z_key = f"Z_{top_dim}"
        
        plt.subplot(grid_rows, grid_cols, i+1)
        sns.violinplot(
            x='Mutation',
            y=z_key,
            data=adata.obs,
            order=[mutation] + [m for m in mutation_dummies.columns if m != mutation]
        )
        plt.title(f"{mutation}: Z_{top_dim} (r={corrs[top_dim]:.2f})")
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run the analysis pipeline."""
    save_dir = tempfile.TemporaryDirectory()
    print(f"Last run with scvi-tools version: {scvi.__version__}")
    
    # Load and preprocess data
    adata = load_and_preprocess_data("all.h5ad")
    
    # Train model
    model, Z_hat = train_scvi_model(adata)
    
    # Plot ELBO history
    plot_elbo_history(model)
    
    # Plot latent space scatter
    plot_latent_scatter(adata)
    
    # Print top loadings
    loadings = model.get_loadings()
    print_top_loadings(loadings)
    
    # Compute UMAP and clustering
    compute_umap_and_clustering(adata, Z_hat)
    
    # Plot UMAP
    plot_umap(adata)
    
    # Analyze mutation correlations
    correlation_matrix, pvalue_matrix, mutation_dummies = analyze_mutation_correlations(adata, Z_hat)
    
    # Analyze top genes
    analyze_top_genes(loadings, mutation_dummies, correlation_matrix, pvalue_matrix)
    
    # Plot mutation distributions
    plot_mutation_distributions(adata, mutation_dummies, correlation_matrix)
    
    save_dir.cleanup()

if __name__ == "__main__":
    main()
