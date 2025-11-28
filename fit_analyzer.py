"""Comprehensive analysis tool for model fitting results.

Provides visualization and statistical analysis of parameter estimates,
model comparisons, convergence diagnostics, and more.
"""
# %% fit_analyzer.py
from datetime import datetime
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm

class FitAnalyzer:
    """Comprehensive analysis tool for model fitting results from PyBADS and PyVBMC.

    Provides visualization and statistical analysis of parameter estimates,
    model comparisons, convergence diagnostics, and more.
    """
    
    def __init__(self, results_dict: Optional[Dict] = None,
                 results_path: Optional[str] = None):
        """Initialize FitAnalyzer with fitting results.

        Args:
            results_dict: Dictionary of fitting results
            results_path: Path to saved results file (.pkl)
        """
        if results_dict is not None:
            self.results = results_dict
        elif results_path is not None:
            self.results = self._load_results(results_path)
        else:
            # Try to load latest results automatically
            from fitting import find_latest_parallel_fit_results
            latest_results, latest_path = find_latest_parallel_fit_results()
            if latest_results:
                print(f"Auto-loaded latest results from: {latest_path}")
                self.results = latest_results
            else:
                raise ValueError("No results provided and no latest results found")
        
        # Extract basic info about the results
        self.model_names = list(self.results.keys())
        self.n_models = len(self.model_names)

        # Get sample information
        sample_model = self.results[self.model_names[0]]
        self.n_subjects = sample_model.get('n_subjects', 0)
        self.fit_toolbox = sample_model.get('fit_toolbox', 'unknown')

        # Extract all parameters across models
        self.all_parameters = self._extract_all_parameters()

        print("FitAnalyzer initialized:")
        print(f"  - {self.n_models} models: {self.model_names}")
        print(f"  - {self.n_subjects} subjects")
        print(f"  - Fit with: {self.fit_toolbox}")
        print(f"  - Parameters: {sorted(self.all_parameters)}")
    
    def _load_results(self, results_path: str) -> Dict:
        """Load results from pickle/dill file with fallback."""
        try:
            # Try dill first (handles complex objects better)
            with open(results_path, 'rb') as f:
                return dill.load(f)
        except Exception as e1:
            try:
                # Fallback to pickle
                with open(results_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e2:
                raise ValueError(
                    f"Failed to load results from {results_path}. "
                    f"Dill error: {e1}, Pickle error: {e2}"
                ) from e2
    
    def _extract_all_parameters(self) -> set:
        """Extract all unique parameter names across all models."""
        parameters = set()
        for model_data in self.results.values():
            if ('individual_results' in model_data and
                    model_data['individual_results']):
                sample_result = model_data['individual_results'][0]
                if 'fitted_params' in sample_result:
                    parameters.update(sample_result['fitted_params'].keys())
        return parameters
    
    def get_parameter_dataframe(self) -> pd.DataFrame:
        """Extract all parameters into a long-format DataFrame for analysis.

        Returns:
            DataFrame with columns: subject, model, parameter, value,
            aic, bic, log_likelihood
        """
        data_rows = []
        
        for model_name, model_data in self.results.items():
            if 'individual_results' not in model_data:
                continue
                
            for i, result in enumerate(model_data['individual_results']):
                if not result.get('fitted_params'):
                    continue

                base_row = {
                    'subject': f'sub-{i+1:03d}',
                    'model': model_name,
                    'aic': result.get('aic', np.nan),
                    'bic': result.get('bic', np.nan),
                    'log_likelihood': result.get('log_likelihood', np.nan),
                    'negative_log_likelihood': result.get(
                        'negative_log_likelihood', np.nan),
                }

                # Add PyVBMC-specific metrics if available
                if 'elbo' in result:
                    base_row['elbo'] = result['elbo']
                if 'log_model_evidence' in result:
                    base_row['log_model_evidence'] = result[
                        'log_model_evidence']

                # Add each parameter as a separate row
                for param_name, param_value in result['fitted_params'].items():
                    row = base_row.copy()
                    row['parameter'] = param_name
                    row['value'] = param_value
                    data_rows.append(row)
        
        return pd.DataFrame(data_rows)

    def plot_parameter_violins(
            self, parameters: Optional[List[str]] = None,
            figsize: Tuple[int, int] = (15, 10),
            save_path: Optional[str] = None) -> plt.Figure:
        """Create violin plots showing parameter distributions across models.

        Args:
            parameters: List of parameters to plot (None = all)
            figsize: Figure size (width, height)
            save_path: Path to save figure

        Returns:
            matplotlib Figure object
        """
        df = self.get_parameter_dataframe()
        
        if parameters is None:
            parameters = sorted(self.all_parameters)
        else:
            parameters = [p for p in parameters if p in self.all_parameters]

        if not parameters:
            raise ValueError("No valid parameters found to plot")

        # Create subplots
        n_params = len(parameters)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        # Ensure axes is always a 2D array for consistent indexing
        if n_params == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        # Plot each parameter
        for i, param in enumerate(parameters):
            row, col = divmod(i, n_cols)
            ax = axes[row, col]

            param_data = df[df['parameter'] == param]
            if param_data.empty:
                ax.text(0.5, 0.5, f'No data for {param}', ha='center',
                       va='center', transform=ax.transAxes)
                ax.set_title(param)
                continue

            # Create violin plot
            sns.violinplot(data=param_data, x='model', y='value', ax=ax,
                          inner='box')
            ax.set_title(f'{param} Distribution')
            ax.set_xlabel('Model')
            ax.set_ylabel(f'{param} Value')

            # Rotate x-axis labels for readability
            ax.tick_params(axis='x', rotation=45)

        # Remove empty subplots
        for i in range(n_params, n_rows * n_cols):
            row, col = divmod(i, n_cols)
            ax = axes[row, col]
            ax.remove()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Parameter violin plots saved to: {save_path}")

        return fig
    
    def get_aic_table(self, sort_by: str = 'aic') -> pd.DataFrame:
        """Create a comprehensive model comparison table sorted by AIC.

        Args:
            sort_by: Metric to sort by ('aic', 'bic', 'log_likelihood')

        Returns:
            DataFrame with model statistics
        """
        summary_data = []
        
        for model_name, model_data in self.results.items():
            if ('individual_results' not in model_data or
                    not model_data['individual_results']):
                continue

            # Extract metrics from individual results
            aics = []
            bics = []
            log_liks = []
            elbos = []
            evidences = []

            for result in model_data['individual_results']:
                if 'aic' in result:
                    aics.append(result['aic'])
                if 'bic' in result:
                    bics.append(result['bic'])
                if 'log_likelihood' in result:
                    log_liks.append(result['log_likelihood'])
                if 'elbo' in result:
                    elbos.append(result['elbo'])
                if 'log_model_evidence' in result:
                    evidences.append(result['log_model_evidence'])

            # Compute summary statistics
            n_results = len(model_data['individual_results'])
            n_successful = model_data.get('n_successful', n_results)
            row = {
                'model': model_name,
                'n_subjects': n_results,
                'n_successful': n_successful,
                'success_rate': n_successful / n_results,
            }
            
            # AIC statistics
            if aics:
                row.update({
                    'aic_mean': np.mean(aics),
                    'aic_std': np.std(aics),
                    'aic_total': np.sum(aics),
                })
            
            # BIC statistics
            if bics:
                row.update({
                    'bic_mean': np.mean(bics),
                    'bic_std': np.std(bics),
                    'bic_total': np.sum(bics),
                })
            
            # Log-likelihood statistics
            if log_liks:
                row.update({
                    'll_mean': np.mean(log_liks),
                    'll_std': np.std(log_liks),
                    'll_total': np.sum(log_liks),
                })
            
            # PyVBMC-specific metrics
            if elbos:
                row.update({
                    'elbo_mean': np.mean(elbos),
                    'elbo_total': np.sum(elbos),
                })

            if evidences and not all(np.isnan(evidences)):
                valid_evidences = [e for e in evidences if not np.isnan(e)]
                if valid_evidences:
                    row.update({
                        'evidence_mean': np.mean(valid_evidences),
                        'evidence_total': np.sum(valid_evidences),
                    })

            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)

        # Sort by specified metric
        if sort_by in df.columns:
            # Lower is better for AIC/BIC
            ascending = sort_by in ['aic_total', 'aic_mean', 'bic_total',
                                   'bic_mean']
            df = df.sort_values(sort_by, ascending=ascending)

        # Calculate relative metrics
        if 'aic_total' in df.columns:
            best_aic = df['aic_total'].min()
            df['delta_aic'] = df['aic_total'] - best_aic
            df['aic_weight'] = (np.exp(-0.5 * df['delta_aic']) /
                               np.sum(np.exp(-0.5 * df['delta_aic'])))

        if 'bic_total' in df.columns:
            best_bic = df['bic_total'].min()
            df['delta_bic'] = df['bic_total'] - best_bic

        return df
    
    def plot_aic_comparison(
            self, figsize: Tuple[int, int] = (12, 8),
            save_path: Optional[str] = None) -> plt.Figure:
        """Create boxplot showing AIC distributions by model.

        Includes individual subject lines showing AIC trajectories.

        Args:
            figsize: Figure size (width, height)
            save_path: Path to save figure

        Returns:
            matplotlib Figure object
        """
        df = self.get_parameter_dataframe()

        # Get AIC data by model
        aic_data = df.groupby(['model', 'subject'])['aic'].first().reset_index()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Boxplot of AIC by model
        sns.boxplot(data=aic_data, x='model', y='aic', ax=ax1)
        ax1.set_title('AIC Distribution by Model')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('AIC')
        ax1.tick_params(axis='x', rotation=45)

        # Individual subject lines
        models = aic_data['model'].unique()
        subjects = aic_data['subject'].unique()

        # Create a pivot table for line plot
        pivot_aic = aic_data.pivot(index='subject', columns='model',
                                  values='aic')

        # Plot individual subject trajectories
        for subject in subjects:
            subject_data = pivot_aic.loc[subject]
            ax2.plot(range(len(models)), subject_data.values, 'o-',
                    alpha=0.7, linewidth=1)

        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels(models, rotation=45)
        ax2.set_title('Individual Subject AIC Trajectories')
        ax2.set_xlabel('Model')
        ax2.set_ylabel('AIC')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"AIC comparison plots saved to: {save_path}")

        return fig
    
    def plot_parameter_correlations(
            self, figsize: Tuple[int, int] = (12, 10),
            save_path: Optional[str] = None) -> plt.Figure:
        """Create correlation heatmaps for parameters within and across models.

        Args:
            figsize: Figure size (width, height)
            save_path: Path to save figure

        Returns:
            matplotlib Figure object
        """
        df = self.get_parameter_dataframe()

        # Create wide format DataFrame
        param_df = df.pivot_table(index=['subject', 'model'],
                                 columns='parameter',
                                 values='value',
                                 aggfunc='first').reset_index()

        # Calculate correlations
        numeric_cols = [col for col in param_df.columns
                       if col not in ['subject', 'model']]
        corr_matrix = param_df[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm',
                   center=0, square=True, linewidths=0.5,
                   cbar_kws={"shrink": 0.8}, ax=ax)

        ax.set_title('Parameter Correlations Across All Models')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Parameter correlation plot saved to: {save_path}")

        return fig
    
    def plot_convergence_diagnostics(self, figsize: Tuple[int, int] = (15, 10),
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Create diagnostic plots for convergence analysis.
        
        Args:
            figsize: Figure size (width, height)
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Success rates by model
        success_data = []
        for model_name, model_data in self.results.items():
            success_data.append({
                'model': model_name,
                'success_rate': model_data.get('n_successful', 0) / model_data.get('n_subjects', 1),
                'n_successful': model_data.get('n_successful', 0),
                'n_total': model_data.get('n_subjects', 0)
            })
        
        success_df = pd.DataFrame(success_data)
        
        axes[0, 0].bar(success_df['model'], success_df['success_rate'])
        axes[0, 0].set_title('Convergence Success Rate by Model')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].set_ylim(0, 1.1)
        
        # Add text annotations
        for i, row in success_df.iterrows():
            axes[0, 0].text(i, row['success_rate'] + 0.02, 
                           f"{row['n_successful']}/{row['n_total']}", 
                           ha='center', va='bottom', fontsize=8)
        
        # 2. AIC distribution by model (if available)
        df = self.get_parameter_dataframe()
        if 'aic' in df.columns:
            aic_by_model = df.groupby(['model', 'subject'])['aic'].first().reset_index()
            sns.boxplot(data=aic_by_model, x='model', y='aic', ax=axes[0, 1])
            axes[0, 1].set_title('AIC Distribution by Model')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Parameter range analysis
        param_ranges = []
        for param in self.all_parameters:
            param_data = df[df['parameter'] == param]['value']
            if not param_data.empty:
                param_ranges.append({
                    'parameter': param,
                    'min': param_data.min(),
                    'max': param_data.max(),
                    'range': param_data.max() - param_data.min(),
                    'std': param_data.std()
                })
        
        range_df = pd.DataFrame(param_ranges)
        if not range_df.empty:
            axes[1, 0].bar(range_df['parameter'], range_df['range'])
            axes[1, 0].set_title('Parameter Range by Parameter')
            axes[1, 0].set_ylabel('Value Range')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Model evidence comparison (PyVBMC only)
        if self.fit_toolbox == 'pyvbmc':
            evidence_data = []
            for model_name, model_data in self.results.items():
                if 'individual_results' in model_data:
                    evidences = []
                    for result in model_data['individual_results']:
                        if 'log_model_evidence' in result and not np.isnan(result['log_model_evidence']):
                            evidences.append(result['log_model_evidence'])
                    if evidences:
                        evidence_data.append({
                            'model': model_name,
                            'evidence_mean': np.mean(evidences),
                            'evidence_std': np.std(evidences)
                        })
            
            if evidence_data:
                evidence_df = pd.DataFrame(evidence_data)
                axes[1, 1].bar(evidence_df['model'], evidence_df['evidence_mean'], 
                              yerr=evidence_df['evidence_std'], capsize=5)
                axes[1, 1].set_title('Model Evidence (PyVBMC)')
                axes[1, 1].set_ylabel('Log Model Evidence')
                axes[1, 1].tick_params(axis='x', rotation=45)
            else:
                axes[1, 1].text(0.5, 0.5, 'No model evidence data', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Model Evidence (PyVBMC)')
        else:
            axes[1, 1].text(0.5, 0.5, f'Model evidence not available\n(fitted with {self.fit_toolbox})', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Model Evidence')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Convergence diagnostics saved to: {save_path}")
        
        return fig
    
    def export_all_analyses(self, output_dir: str = "fit_analysis_output") -> Dict[str, str]:
        """
        Generate and save all analysis plots and tables.
        
        Args:
            output_dir: Directory to save all outputs
            
        Returns:
            Dictionary mapping analysis type to saved file path
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}
        
        print(f"Exporting all analyses to: {output_path}")
        
        try:
            # 1. Parameter violin plots
            violin_path = output_path / f"parameter_violins_{timestamp}.png"
            self.plot_parameter_violins(save_path=str(violin_path))
            saved_files['parameter_violins'] = str(violin_path)
            plt.close()
            
            # 2. AIC comparison
            aic_path = output_path / f"aic_comparison_{timestamp}.png"
            self.plot_aic_comparison(save_path=str(aic_path))
            saved_files['aic_comparison'] = str(aic_path)
            plt.close()
            
            # 3. Parameter correlations
            corr_path = output_path / f"parameter_correlations_{timestamp}.png"
            self.plot_parameter_correlations(save_path=str(corr_path))
            saved_files['parameter_correlations'] = str(corr_path)
            plt.close()
            
            # 4. Convergence diagnostics
            diag_path = output_path / f"convergence_diagnostics_{timestamp}.png"
            self.plot_convergence_diagnostics(save_path=str(diag_path))
            saved_files['convergence_diagnostics'] = str(diag_path)
            plt.close()
            
            # 5. AIC table
            aic_table = self.get_aic_table()
            table_path = output_path / f"aic_table_{timestamp}.csv"
            aic_table.to_csv(table_path, index=False)
            saved_files['aic_table'] = str(table_path)
            
            # 6. Parameter data
            param_df = self.get_parameter_dataframe()
            param_path = output_path / f"parameter_data_{timestamp}.csv"
            param_df.to_csv(param_path, index=False)
            saved_files['parameter_data'] = str(param_path)
            
            print(f"\n✓ All analyses exported successfully!")
            print("Saved files:")
            for analysis_type, file_path in saved_files.items():
                print(f"  - {analysis_type}: {file_path}")
            
        except Exception as e:
            print(f"Error during export: {e}")
            raise
        
        return saved_files
    
    def summary(self) -> None:
        """Print a comprehensive summary of the fitting results."""
        print("=" * 60)
        print("FIT RESULTS SUMMARY")
        print("=" * 60)
        
        print(f"Dataset: {self.n_subjects} subjects")
        print(f"Fitting method: {self.fit_toolbox.upper()}")
        print(f"Models fitted: {self.n_models}")
        
        # Model comparison table
        print("\nMODEL COMPARISON (sorted by AIC):")
        print("-" * 50)
        
        aic_table = self.get_aic_table('aic_total')
        
        # Display key columns
        display_cols = ['model', 'aic_total', 'delta_aic', 'aic_weight', 'success_rate']
        display_cols = [col for col in display_cols if col in aic_table.columns]
        
        print(aic_table[display_cols].round(3).to_string(index=False))
        
        # Best model
        if 'aic_total' in aic_table.columns:
            best_model = aic_table.iloc[0]['model']
            best_aic = aic_table.iloc[0]['aic_total']
            print(f"\n✓ Best model: {best_model} (AIC = {best_aic:.1f})")
        
        # Parameter summary
        print(f"\nPARAMETERS FITTED:")
        print("-" * 20)
        for param in sorted(self.all_parameters):
            print(f"  - {param}")
        
        print("\n" + "=" * 60)

        # Population means
        print("POPULATION MEANS (across all subjects):")
        print("-" * 50)
        param_df = self.get_parameter_dataframe()
        population_means = param_df.groupby('parameter')['value'].mean().round(3)
        for param, mean in population_means.items():
            print(f"  - {param}: {mean:.3f}")
        
        # Add probabilistic comparison if available
        if self.fit_toolbox == 'pyvbmc':
            print("\n" + "=" * 60)
            print("PROBABILISTIC MODEL COMPARISON:")
            print("-" * 30)
            ranking_df = self.get_model_ranking_probabilities()
            if ranking_df is not None:
                print("Model ranking by probability of being best:")
                for i, (_, row) in enumerate(ranking_df.head(3).iterrows()):  # Show top 3
                    model = row['model']
                    prob_best = row['prob_best']
                    print(f"  {i+1}. {model}: {prob_best:.3f}")
                
                if len(ranking_df) > 3:
                    print(f"  ... and {len(ranking_df) - 3} more models")
                    
                print("\nFor detailed probabilistic comparison, use:")
                print("  analyzer.print_probabilistic_model_comparison()")
            else:
                print("  No ELBO uncertainty available for probabilistic comparison")

    def get_bayes_factor_table(self, elbo_uncertainty_weight: float = 0.0) -> Optional[pd.DataFrame]:
        """
        Create pairwise Bayes factor comparison table for PyVBMC results.
        Uses log_model_evidence if available, otherwise falls back to ELBO.

        Args:
            elbo_uncertainty_weight: Weight for ELBO uncertainty penalty (default 0.0).
                                   When > 0, penalizes models with higher ELBO uncertainty.
                                   Effective ELBO = ELBO - weight * ELBO_SD

        Returns:
            DataFrame with pairwise Bayes factors, or None if not PyVBMC results
        """
        if self.fit_toolbox != 'pyvbmc':
            print(f"Bayes factors only available for PyVBMC results (current: {self.fit_toolbox})")
            return None

        # Extract log model evidences for each model
        model_evidences = {}
        evidence_type = "log_model_evidence"
        
        # First try to use log_model_evidence
        for model_name, model_data in self.results.items():
            if 'individual_results' in model_data:
                evidences = []
                for result in model_data['individual_results']:
                    if 'log_model_evidence' in result and not np.isnan(result['log_model_evidence']):
                        evidences.append(result['log_model_evidence'])

                if evidences:
                    # Use total log evidence across all subjects
                    model_evidences[model_name] = np.sum(evidences)

        # If no valid log_model_evidence found, fall back to ELBO
        if len(model_evidences) < 2:
            print("No valid log_model_evidence found, falling back to ELBO for model comparison")
            if elbo_uncertainty_weight > 0:
                print(f"Using ELBO uncertainty weighting (weight = {elbo_uncertainty_weight})")
            evidence_type = "elbo"
            model_evidences = {}
            
            for model_name, model_data in self.results.items():
                if 'individual_results' in model_data:
                    elbos = []
                    for result in model_data['individual_results']:
                        if 'elbo' in result and not np.isnan(result['elbo']):
                            elbo = result['elbo']
                            
                            # Apply uncertainty weighting if requested
                            if elbo_uncertainty_weight > 0:
                                # Check for elbo_sd in the result (from PyVBMC)
                                elbo_sd = None
                                if 'elbo_sd' in result and not np.isnan(result['elbo_sd']):
                                    elbo_sd = result['elbo_sd']
                                # Also check in the raw VBMC result if available
                                elif ('best_result' in result and 
                                      isinstance(result['best_result'], dict) and 
                                      'elbo_sd' in result['best_result']):
                                    elbo_sd = result['best_result']['elbo_sd']
                                
                                if elbo_sd is not None:
                                    # Penalize by uncertainty: effective_elbo = elbo - weight * elbo_sd
                                    elbo = elbo - elbo_uncertainty_weight * elbo_sd
                                    
                            elbos.append(elbo)

                    if elbos:
                        # Use total ELBO (possibly uncertainty-weighted) across all subjects
                        model_evidences[model_name] = np.sum(elbos)

        if len(model_evidences) < 2:
            print("Need at least 2 models with valid evidence (log_model_evidence or ELBO) for Bayes factor comparison")
            return None

        # Create pairwise Bayes factor matrix
        models = list(model_evidences.keys())

        bf_data = []
        for i, model_i in enumerate(models):
            for j, model_j in enumerate(models):
                if i != j:  # Skip diagonal
                    # BF_ij = exp(log_evidence_i - log_evidence_j)
                    log_bf = model_evidences[model_i] - model_evidences[model_j]
                    bf = np.exp(log_bf)

                    # Interpretation
                    if bf > 100:
                        strength = "decisive"
                    elif bf > 10:
                        strength = "strong"
                    elif bf > 3:
                        strength = "moderate"
                    elif bf > 1:
                        strength = "weak"
                    else:
                        strength = "against"

                    bf_data.append({
                        'model_i': model_i,
                        'model_j': model_j,
                        'log_bayes_factor': log_bf,
                        'bayes_factor': bf,
                        'evidence_strength': strength,
                        'log_evidence_i': model_evidences[model_i],
                        'log_evidence_j': model_evidences[model_j],
                        'evidence_type': evidence_type
                    })

        return pd.DataFrame(bf_data)

    def get_probabilistic_model_comparison_table(self, min_difference: Optional[float] = None, 
                                                  auto_min_diff_factor: float = 0.01) -> Optional[pd.DataFrame]:
        """
        Create probabilistic model comparison table using ELBO uncertainty.
        
        For each pair of models (i,j), computes P(Model_i - Model_j > delta) assuming:
        ELBO_i ~ N(elbo_i, elbo_sd_i²)
        
        Args:
            min_difference: Minimum difference threshold delta. If None, computed automatically
            auto_min_diff_factor: Factor for automatic threshold (fraction of mean absolute ELBO)
        
        Returns:
            DataFrame with pairwise comparison probabilities, or None if not available
        """
        if self.fit_toolbox != 'pyvbmc':
            print(f"Probabilistic model comparison only available for PyVBMC results (current: {self.fit_toolbox})")
            return None

        # Extract ELBO and ELBO_SD for each model
        model_stats = {}
        for model_name, model_data in self.results.items():
            if 'individual_results' in model_data:
                elbos = []
                elbo_sds = []
                
                for result in model_data['individual_results']:
                    if 'elbo' in result and not np.isnan(result['elbo']):
                        elbos.append(result['elbo'])
                        
                        # Look for ELBO uncertainty
                        elbo_sd = None
                        if 'elbo_sd' in result and not np.isnan(result['elbo_sd']):
                            elbo_sd = result['elbo_sd']
                        elif ('best_result' in result and 
                              isinstance(result['best_result'], dict) and 
                              'elbo_sd' in result['best_result']):
                            elbo_sd = result['best_result']['elbo_sd']
                        
                        if elbo_sd is not None:
                            elbo_sds.append(elbo_sd)
                        else:
                            # If no uncertainty available, use a small default
                            elbo_sds.append(0.1)  # Conservative default
                
                if elbos and elbo_sds:
                    # Total ELBO across subjects
                    total_elbo = np.sum(elbos)
                    # Total uncertainty (assuming independence across subjects)
                    total_elbo_sd = np.sqrt(np.sum(np.array(elbo_sds)**2))
                    
                    model_stats[model_name] = {
                        'total_elbo': total_elbo,
                        'total_elbo_sd': total_elbo_sd,
                        'n_subjects': len(elbos)
                    }

        if len(model_stats) < 2:
            print("Need at least 2 models with ELBO and ELBO_SD for probabilistic comparison")
            return None

        # Compute automatic minimum difference threshold if not provided
        if min_difference is None:
            all_elbos = [stats['total_elbo'] for stats in model_stats.values()]
            mean_abs_elbo = np.mean(np.abs(all_elbos))
            min_difference = auto_min_diff_factor * mean_abs_elbo
            print(f"Using automatic minimum difference threshold: {min_difference:.4f} "
                  f"({auto_min_diff_factor:.3f} * mean(|ELBO|) = {auto_min_diff_factor:.3f} * {mean_abs_elbo:.2f})")

        # Create pairwise probability matrix
        models = list(model_stats.keys())
        comparison_data = []
        
        for model_i in models:
            for model_j in models:
                if model_i != model_j:
                    # Get statistics for both models
                    elbo_i = model_stats[model_i]['total_elbo']
                    elbo_sd_i = model_stats[model_i]['total_elbo_sd']
                    elbo_j = model_stats[model_j]['total_elbo']
                    elbo_sd_j = model_stats[model_j]['total_elbo_sd']
                    
                    # Compute P(Model_i - Model_j > min_difference)
                    # Difference ~ N(elbo_i - elbo_j, elbo_sd_i² + elbo_sd_j²)
                    diff_mean = elbo_i - elbo_j
                    diff_sd = np.sqrt(elbo_sd_i**2 + elbo_sd_j**2)
                    
                    # P(Difference > min_difference) = P(Model_i meaningfully better than Model_j)
                    if diff_sd > 0:
                        prob_i_better = norm.cdf((diff_mean - min_difference) / diff_sd)
                    else:
                        # If no uncertainty, use simple threshold
                        prob_i_better = 1.0 if diff_mean > min_difference else 0.0
                    
                    # Bayes factor approximation: BF = P/(1-P)
                    bf_approx = prob_i_better / (1 - prob_i_better) if prob_i_better < 0.999 else 999.0
                    
                    comparison_data.append({
                        'model_i': model_i,
                        'model_j': model_j,
                        'elbo_i': elbo_i,
                        'elbo_sd_i': elbo_sd_i,
                        'elbo_j': elbo_j,
                        'elbo_sd_j': elbo_sd_j,
                        'elbo_diff_mean': diff_mean,
                        'elbo_diff_sd': diff_sd,
                        'min_difference': min_difference,
                        'prob_i_better': prob_i_better,
                        'bf_approximation': bf_approx,
                        'evidence_strength': self._interpret_probability(prob_i_better)
                    })

        return pd.DataFrame(comparison_data)

    def _interpret_probability(self, prob: float) -> str:
        """Interpret probability as evidence strength"""
        if prob > 0.99:
            return "decisive"
        elif prob > 0.95:
            return "strong"
        elif prob > 0.8:
            return "moderate"
        elif prob > 0.6:
            return "weak"
        else:
            return "against"

    def get_model_ranking_probabilities(self, min_difference: Optional[float] = None, 
                                        auto_min_diff_factor: float = 0.01) -> Optional[pd.DataFrame]:
        """
        Compute probability that each model is the best overall.
        
        Args:
            min_difference: Minimum difference threshold delta. If None, computed automatically
            auto_min_diff_factor: Factor for automatic threshold (fraction of mean absolute ELBO)
        
        Returns:
            DataFrame with model ranking probabilities
        """
        prob_table = self.get_probabilistic_model_comparison_table(
            min_difference=min_difference, auto_min_diff_factor=auto_min_diff_factor)
        if prob_table is None:
            return None
            
        models = prob_table['model_i'].unique()
        ranking_data = []
        
        for model in models:
            # Probability this model is better than all others
            prob_best = 1.0
            comparisons = prob_table[prob_table['model_i'] == model]
            
            for _, row in comparisons.iterrows():
                prob_best *= row['prob_i_better']
            
            # Get model statistics
            model_comparisons = prob_table[prob_table['model_i'] == model]
            if not model_comparisons.empty:
                elbo = model_comparisons.iloc[0]['elbo_i']
                elbo_sd = model_comparisons.iloc[0]['elbo_sd_i']
            else:
                elbo = np.nan
                elbo_sd = np.nan
            
            ranking_data.append({
                'model': model,
                'prob_best': prob_best,
                'total_elbo': elbo,
                'total_elbo_sd': elbo_sd
            })
        
        df = pd.DataFrame(ranking_data)
        return df.sort_values('prob_best', ascending=False)

    def print_bayes_factor_summary(self, elbo_uncertainty_weight: float = 0.0) -> None:
        """Print a comprehensive summary of Bayes factor model comparisons."""

        bf_table = self.get_bayes_factor_table(elbo_uncertainty_weight=elbo_uncertainty_weight)
        if bf_table is None:
            return

        # Get evidence type from the table
        evidence_type = bf_table['evidence_type'].iloc[0] if 'evidence_type' in bf_table.columns else 'log_model_evidence'
        evidence_label = "Log Model Evidence" if evidence_type == 'log_model_evidence' else "ELBO (Evidence Lower Bound)"

        print("=" * 80)
        print(f"BAYES FACTOR MODEL COMPARISON (PyVBMC) - Using {evidence_label}")
        print("=" * 80)

        # Get unique models and their evidences
        models = bf_table['model_i'].unique()

        # Show evidences for each model
        print(f"\n{evidence_label.upper()} BY MODEL:")
        print("-" * 40)
        for model in models:
            evidence = bf_table[bf_table['model_i'] == model]['log_evidence_i'].iloc[0]
            print(f"  {model}: {evidence:.2f}")

        # Find best model (highest evidence)
        best_model_evidence = bf_table.groupby('model_i')['log_evidence_i'].first()
        best_model = best_model_evidence.idxmax()
        best_evidence = best_model_evidence.max()

        print(f"\n✓ Best model: {best_model} ({evidence_type.replace('_', ' ')} = {best_evidence:.2f})")
        
        if evidence_type == 'elbo':
            print("  Note: Using ELBO (lower bound) since log model evidence not available")
            if elbo_uncertainty_weight > 0:
                print(f"  Note: Applied uncertainty weighting (weight = {elbo_uncertainty_weight})")

        # Show pairwise comparisons against best model
        print(f"\nCOMPARISONS AGAINST BEST MODEL ({best_model}):")
        print("-" * 60)
        print("Model              | BF vs Best | Log BF  | Evidence")
        print("-" * 60)

        best_comparisons = bf_table[bf_table['model_j'] == best_model].sort_values('bayes_factor', ascending=False)

        for _, row in best_comparisons.iterrows():
            model = row['model_i']
            bf = row['bayes_factor']
            log_bf = row['log_bayes_factor']
            strength = row['evidence_strength']

            print(f"{model:18s} | {bf:10.2f} | {log_bf:7.2f} | {strength}")

        # Show all pairwise comparisons with strong+ evidence
        strong_comparisons = bf_table[bf_table['bayes_factor'] > 10].sort_values('bayes_factor', ascending=False)

        if len(strong_comparisons) > 0:
            print(f"\nSTRONG+ EVIDENCE COMPARISONS (BF > 10):")
            print("-" * 70)
            print("Model A            | Model B            | BF(A vs B) | Evidence")
            print("-" * 70)

            for _, row in strong_comparisons.iterrows():
                print(f"{row['model_i']:18s} | {row['model_j']:18s} | {row['bayes_factor']:10.2f} | {row['evidence_strength']}")

        print("\nBayes Factor Interpretation:")
        print("  BF > 100: decisive evidence")
        print("  BF > 10:  strong evidence")
        print("  BF > 3:   moderate evidence")
        print("  BF > 1:   weak evidence")
        print("  BF < 1:   evidence against")

        print("=" * 80)

    def print_probabilistic_model_comparison(self, min_difference: Optional[float] = None, 
                                             auto_min_diff_factor: float = 0.01) -> None:
        """Print probabilistic model comparison using ELBO uncertainty.
        
        Args:
            min_difference: Minimum difference threshold delta. If None, computed automatically
            auto_min_diff_factor: Factor for automatic threshold (fraction of mean absolute ELBO)
        """
        
        # Get model ranking probabilities
        ranking_df = self.get_model_ranking_probabilities(
            min_difference=min_difference, auto_min_diff_factor=auto_min_diff_factor)
        if ranking_df is None:
            print("Probabilistic model comparison not available (requires PyVBMC with ELBO uncertainty)")
            return
            
        print("=" * 80)
        print("PROBABILISTIC MODEL COMPARISON (Using ELBO Uncertainty)")
        print("=" * 80)
        
        # Get prob_table for threshold display
        prob_table = self.get_probabilistic_model_comparison_table(
            min_difference=min_difference, auto_min_diff_factor=auto_min_diff_factor)
        
        # Show threshold information
        if prob_table is not None and not prob_table.empty:
            threshold_used = prob_table['min_difference'].iloc[0]
            print(f"Minimum difference threshold: {threshold_used:.4f}")
            print("(Models must exceed this threshold to be considered 'meaningfully better')")
            print()
        
        # Show model ranking by probability of being best
        print("\nMODEL RANKING BY PROBABILITY OF BEING BEST:")
        print("-" * 60)
        print("Rank | Model                | P(Best) | Total ELBO | ELBO SD")
        print("-" * 60)
        
        for i, (_, row) in enumerate(ranking_df.iterrows()):
            model = row['model']
            prob_best = row['prob_best']
            total_elbo = row['total_elbo']
            total_elbo_sd = row['total_elbo_sd']
            
            print(f"{i+1:4d} | {model:20s} | {prob_best:7.3f} | {total_elbo:10.2f} | {total_elbo_sd:7.2f}")
        
        # Show pairwise comparison probabilities for top models
        if prob_table is not None:
            print(f"\nPAIRWISE COMPARISON PROBABILITIES:")
            print("-" * 70)
            print("Model A            | Model B            | P(A > B) | Evidence")
            print("-" * 70)
            
            # Show comparisons involving the best model
            best_model = ranking_df.iloc[0]['model']
            best_comparisons = prob_table[
                (prob_table['model_i'] == best_model) | 
                (prob_table['model_j'] == best_model)
            ].sort_values('prob_i_better', ascending=False)
            
            for _, row in best_comparisons.iterrows():
                model_a = row['model_i']
                model_b = row['model_j']
                prob = row['prob_i_better']
                strength = row['evidence_strength']
                
                print(f"{model_a:18s} | {model_b:18s} | {prob:8.3f} | {strength}")
            
            # Show strong evidence comparisons (P > 0.95)
            strong_comparisons = prob_table[prob_table['prob_i_better'] > 0.95]
            
            if len(strong_comparisons) > 0:
                print(f"\nSTRONG EVIDENCE COMPARISONS (P > 0.95):")
                print("-" * 70)
                print("Model A            | Model B            | P(A > B) | Evidence")
                print("-" * 70)
                
                for _, row in strong_comparisons.iterrows():
                    print(f"{row['model_i']:18s} | {row['model_j']:18s} | {row['prob_i_better']:8.3f} | {row['evidence_strength']}")
        
        print("\nProbability Interpretation:")
        print("  P > 0.99: decisive evidence that model A is meaningfully better")
        print("  P > 0.95: strong evidence that model A is meaningfully better")
        print("  P > 0.80: moderate evidence that model A is meaningfully better")
        print("  P > 0.60: weak evidence that model A is meaningfully better")
        print("  P < 0.60: evidence against model A being meaningfully better")
        if prob_table is not None and not prob_table.empty:
            threshold_used = prob_table['min_difference'].iloc[0]
            print(f"\n'Meaningfully better' = ELBO difference > {threshold_used:.4f}")
        
        print("=" * 80)

# %% Helper functions for quick analysis

def compare_toolboxes(pybads_path: str, pyvbmc_path: str) -> None:
    """
    Compare results from PyBADS and PyVBMC fitting.
    
    Args:
        pybads_path: Path to PyBADS results
        pyvbmc_path: Path to PyVBMC results
    """
    print("COMPARING PYBADS vs PYVBMC RESULTS")
    print("=" * 50)
    
    # Load both result sets
    bads_analyzer = FitAnalyzer(results_path=pybads_path)
    vbmc_analyzer = FitAnalyzer(results_path=pyvbmc_path)
    
    # Get AIC tables
    bads_table = bads_analyzer.get_aic_table()
    vbmc_table = vbmc_analyzer.get_aic_table()
    
    # Compare model rankings
    print("\nMODEL RANKINGS:")
    print("-" * 30)
    print("PyBADS ranking  |  PyVBMC ranking")
    print("-" * 30)
    
    for i in range(min(len(bads_table), len(vbmc_table))):
        bads_model = bads_table.iloc[i]['model']
        vbmc_model = vbmc_table.iloc[i]['model']
        print(f"{i+1:2d}. {bads_model:15s} | {i+1:2d}. {vbmc_model}")
    
    print("\n" + "=" * 50)

    # Population means
    print("POPULATION MEANS PYBADS (across all subjects):")
    print("-" * 50)
    param_df = bads_analyzer.get_parameter_dataframe()
    population_means = param_df.groupby('parameter')['value'].mean().round(3)
    for param, mean in population_means.items():
        print(f"  - {param}: {mean:.3f}")

    # Population means
    print("POPULATION MEANS PYVBMC (across all subjects):")
    print("-" * 50)
    param_df = vbmc_analyzer.get_parameter_dataframe()
    population_means = param_df.groupby('parameter')['value'].mean().round(3)
    for param, mean in population_means.items():
        print(f"  - {param}: {mean:.3f}")

# %%

