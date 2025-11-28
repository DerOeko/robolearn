import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import textwrap
import numpy as np

class DataAnalyzer():
    def __init__(self, raw_data, is_subject_data=False):
        self.raw_data = raw_data
        self.state_map = {0: 'go2win', 1: 'go2avoid', 2: 'nogo2win', 3: 'nogo2avoid'}
        self.is_subject_data = is_subject_data
        self.color_map = {0: "#D58E2A", 1: "#186673", 2: "#D58E2A",  3: "#186673"}
        self.style_map = {0: '-', 1: '-', 2: '--', 3: '--'}
    def calculate_learning_curves(self):
        data = self.raw_data.copy()

        if self.is_subject_data:
            data = data[data['blocktype'] != 'calibration'].reset_index(drop=True)

            data['state_occurrence'] = data.groupby(['subject', 'blockId', 'blocktype', 'S']).cumcount() + 1

            avg_curve = data.groupby(['blocktype', 'S', 'state_occurrence'])['A'].agg(['mean', 'std', 'count']).reset_index()

            avg_curve['sem'] = avg_curve['std']/np.sqrt(avg_curve['count'])
            avg_curve['state_label'] = avg_curve['S'].map(self.state_map)

            avg_curve = avg_curve.rename(columns={'mean': 'p(Go)', 'state_occurrence': 'repetition', 'blocktype': 'block_type', 'S': 'stimulus'})
            avg_curve = avg_curve[['repetition', 'p(Go)', 'sem', 'state_label', 'block_type', 'stimulus']]
        else:
            data = data[data['block_type'] != 'calibration'].reset_index(drop=True)

            data['state_occurrence'] = data.groupby(['simulation_id', 'block_idx', 'block_type', 'stimulus']).cumcount() + 1

            avg_curve = data.groupby(['block_type', 'stimulus', 'state_occurrence'])['action'].agg(['mean', 'std', 'count']).reset_index()

            avg_curve['sem'] = avg_curve['std']/np.sqrt(avg_curve['count'])
            avg_curve['state_label'] = avg_curve['stimulus'].map(self.state_map)

            avg_curve = avg_curve.rename(columns={'mean': 'p(Go)', 'state_occurrence': 'repetition'})
            avg_curve = avg_curve[['repetition', 'p(Go)', 'sem', 'state_label', 'block_type', 'stimulus']]

        return avg_curve

    def plot_learning_curves(self, save_path=None, title="learning curves by block type"):
        """
        plots learning curves with subplots for each block type
        """


        avg_curve = self.calculate_learning_curves()
        data = avg_curve.copy()

        # setup plotting style
        sns.set_context('paper')
        sns.set_theme(style='whitegrid')
        mpl.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 14,
            'legend.fontsize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'figure.dpi': 300,
            'savefig.dpi': 300
        })

        # get unique block types
        block_types = sorted(data['block_type'].unique())
        n_blocks = len(block_types)

        # create subplots
        fig, axes = plt.subplots(1, n_blocks, figsize=(5*n_blocks, 5), sharey=True)
        if n_blocks == 1:
            axes = [axes]

        # plot each block type in its own subplot
        for i, block_type in enumerate(block_types):
            ax = axes[i]
            block_data = data[data['block_type'] == block_type]

            labels_plotted = set()

            for stimulus in sorted(block_data['stimulus'].unique()):
                stim_data = block_data[block_data['stimulus'] == stimulus]

                label = self.state_map[stimulus] if self.state_map[stimulus] not in labels_plotted else None
                if label is not None:
                    labels_plotted.add(label)

                x = stim_data['repetition'].astype(float).values
                y = stim_data['p(Go)'].astype(float).values
                y_lower = (stim_data['p(Go)'] - stim_data['sem']).astype(float).values
                y_upper = (stim_data['p(Go)'] + stim_data['sem']).astype(float).values

                ax.plot(
                    x, y,
                    linestyle=self.style_map[stimulus],
                    color=self.color_map[stimulus],
                    marker='o',
                    linewidth=2.5,
                    markersize=4,
                    label=label
                )

                ax.fill_between(
                    x, y_lower, y_upper,
                    color=self.color_map[stimulus],
                    alpha=0.2
                )

            # style each subplot
            ax.axhline(y=0.5, color='k', linestyle='--', linewidth=1.5, alpha=0.3)
            ax.set_xlabel('state repetition', fontsize=14)
            ax.set_title(f'{block_type} control', fontsize=14)
            ax.set_ylim(-0.05, 1.05)
            ax.legend(title='state', loc='best', frameon=True, fontsize=9)

        # only add ylabel to leftmost subplot
        axes[0].set_ylabel('p(go)', fontsize=14)

        # overall title
        wrapped_title = "\n".join(textwrap.wrap(title, width=60))
        fig.suptitle(wrapped_title, fontsize=16, y=0.98)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def calculate_wsls_data(self, debug=False):
        """calculate win-stay lose-shift data from both subject and simulation results"""
        data = self.raw_data.copy()

        all_wsls_data = []

        # Set up column names based on data type
        if self.is_subject_data:
            action_col = 'A'
            stimulus_col = 'S'
            reward_col = 'reward_recoded'
            block_type_col = 'blocktype'
            shift_group = ['subject', stimulus_col]
        else:
            data = data[data['block_type'] != 'calibration'].reset_index(drop=True)
            action_col = 'action'
            stimulus_col = 'stimulus'  
            reward_col = 'reward'
            block_type_col = 'block_type'
            shift_group = ['simulation_id', stimulus_col]

        if debug:
            print(f"Data type: {'Subject' if self.is_subject_data else 'Simulation'}")
            print(f"Total trials after calibration filter: {len(data)}")
            print(f"Unique block types: {data[block_type_col].unique()}")
            print(f"Shift group: {shift_group}")
            print(f"Reward distribution: {data[reward_col].value_counts()}")

        for block_type in data[block_type_col].unique():
            block_data = data[data[block_type_col] == block_type].copy()

            if debug:
                print(f"\n=== Block type: {block_type} ===")
                print(f"Trials in this block: {len(block_data)}")

            # define favorable outcome
            block_data['favourable'] = block_data[reward_col] == 1
            
            if debug:
                print(f"Favorable outcomes: {block_data['favourable'].sum()} / {len(block_data)} = {block_data['favourable'].mean():.3f}")
            
            # Get previous actions and outcomes
            block_data['prev_action'] = block_data.groupby(shift_group)[action_col].shift(1)
            block_data['action_repeated'] = block_data[action_col] == block_data['prev_action']
            block_data['favourable_previous'] = block_data.groupby(shift_group)['favourable'].shift(1)

            # Clean up and analyze
            wsls_data = block_data.dropna(subset=['favourable_previous', 'action_repeated']).copy()
            
            if debug:
                print(f"Valid WSLS trials (after dropping NaN): {len(wsls_data)}")
                if len(wsls_data) > 0:
                    fav_prev = wsls_data['favourable_previous']
                    action_rep = wsls_data['action_repeated']
                    print(f"Previous favorable: {fav_prev.sum()} / {len(fav_prev)} = {fav_prev.mean():.3f}")
                    print(f"Actions repeated overall: {action_rep.sum()} / {len(action_rep)} = {action_rep.mean():.3f}")
                    
                    # Check WSLS behavior
                    fav_stay = action_rep[fav_prev == True].mean() if (fav_prev == True).any() else 0
                    unfav_stay = action_rep[fav_prev == False].mean() if (fav_prev == False).any() else 0
                    print(f"Stay after favorable: {fav_stay:.3f}")
                    print(f"Stay after unfavorable: {unfav_stay:.3f}")
            
            wsls_data['Outcome'] = wsls_data['favourable_previous'].map({
                True: "Favourable (Win-Stay)",
                False: "Unfavourable (Lose-Shift)"
            })
            wsls_data['Control'] = block_type

            all_wsls_data.append(wsls_data)

        return pd.concat(all_wsls_data, ignore_index=True)

    def plot_wsls(self, save_path=None, title="Win-Stay Lose-Shift", debug=False):
        """plot win-stay lose-shift analysis"""
        wsls_data = self.calculate_wsls_data(debug=debug)

        # Standardize control labels locally
        control_mapping = {
            'HighControl': 'high',
            'LowControl': 'low', 
            'Yoked': 'yoked'
        }
        wsls_data['Control'] = wsls_data['Control'].replace(control_mapping)

        # setup plotting style
        sns.set_context('paper')
        sns.set_theme(style='whitegrid')
        mpl.rcParams.update({
            'font.size': 14,
            'axes.labelsize': 16,
            'axes.titlesize': 16,
            'legend.fontsize': 12,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'figure.dpi': 300,
            'savefig.dpi': 300
        })

        plt.figure(figsize=(10, 6))

        # Define consistent order and colors
        control_order = ['high', 'low', 'yoked']
        control_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # blue, orange, green

        # create barplot
        ax = sns.barplot(
            x='Outcome',
            y='action_repeated',
            hue='Control',
            data=wsls_data,
            estimator=np.mean,
            errorbar=('ci', 95),
            palette='colorblind',
            hue_order=control_order,
            order=["Favourable (Win-Stay)", "Unfavourable (Lose-Shift)"]
        )

        # style the plot
        plt.ylabel("p(stay)", fontsize=16)
        plt.xlabel("Previous Outcome", fontsize=16)
        wrapped_title = "\n".join(textwrap.wrap(title, width=50))
        plt.title(wrapped_title, fontsize=18, weight='bold')
        plt.ylim(0, 1)

        ax.yaxis.grid(True, linestyle='--', color='gray', alpha=0.7)
        ax.xaxis.grid(False)
        plt.legend(title='Control', loc='best', frameon=True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()
    
    def calculate_accuracy_data(self):
        """Calculate accuracy/performance data by block type and stimulus"""
        data = self.raw_data.copy()
        
        # Set up column names based on data type
        if self.is_subject_data:
            data = data[data['blocktype'] != 'calibration'].reset_index(drop=True)
            action_col = 'A'
            stimulus_col = 'S'
            reward_col = 'reward_recoded' if 'reward_recoded' in data.columns else 'valence' if 'valence' in data.columns else 'R'
            block_type_col = 'blocktype'
            group_cols = ['subject', 'blockId', 'blocktype', 'S']
        else:
            data = data[data['block_type'] != 'calibration'].reset_index(drop=True)

            action_col = 'action'
            stimulus_col = 'stimulus'  
            reward_col = 'reward'
            block_type_col = 'block_type'
            group_cols = ['simulation_id', 'block_idx', 'block_type', 'stimulus']
        
        # Calculate optimal actions based on correct state mapping
        optimal_actions = {0: 1, 1: 1, 2: 0, 3: 0}  # go2win, go2avoid, nogo2win, nogo2avoid
        data['optimal_action'] = data[stimulus_col].map(optimal_actions)
        data['correct_action'] = (data[action_col] == data['optimal_action']).astype(int)
        
        # Calculate accuracy by block type and stimulus
        accuracy = data.groupby([block_type_col, stimulus_col])['correct_action'].agg(['mean', 'std', 'count']).reset_index()
        accuracy['sem'] = accuracy['std'] / np.sqrt(accuracy['count'])
        accuracy['state_label'] = accuracy[stimulus_col].map(self.state_map)
        
        return accuracy.rename(columns={'mean': 'accuracy', stimulus_col: 'stimulus', block_type_col: 'block_type'})
    
    def plot_accuracy(self, save_path=None, title="Accuracy by Block Type and Stimulus"):
        """Plot accuracy across block types and stimuli"""
        accuracy_data = self.calculate_accuracy_data()
        
        # Setup plotting style
        sns.set_context('paper')
        sns.set_theme(style='whitegrid')
        mpl.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 14,
            'legend.fontsize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'figure.dpi': 300,
            'savefig.dpi': 300
        })

        # Get unique block types
        block_types = sorted(accuracy_data['block_type'].unique())
        n_blocks = len(block_types)
        
        fig, axes = plt.subplots(1, n_blocks, figsize=(5*n_blocks, 5), sharey=True)
        if n_blocks == 1:
            axes = [axes]
        
        for i, block_type in enumerate(block_types):
            ax = axes[i]
            block_data = accuracy_data[accuracy_data['block_type'] == block_type]
            
            x_pos = np.arange(len(block_data))
            bars = ax.bar(x_pos, block_data['accuracy'], 
                         yerr=block_data['sem'],
                         color=[self.color_map[stim] for stim in block_data['stimulus']],
                         alpha=0.9, capsize=5)
            
            ax.set_xlabel('Stimulus Type', fontsize=14)
            ax.set_title(f'{block_type} Control', fontsize=14)
            ax.set_ylim(0, 1)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(block_data['state_label'], rotation=45)
            ax.axhline(y=0.5, color='k', linestyle='--', linewidth=1.5, alpha=0.3)
        
        axes[0].set_ylabel('Accuracy', fontsize=14)
        
        wrapped_title = "\\n".join(textwrap.wrap(title, width=60))
        fig.suptitle(wrapped_title, fontsize=16, y=0.98)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def compare_datasets(subject_data, simulation_data, save_path=None):
        color_map = {0: "#D58E2A", 1: "#186673", 2: "#D58E2A",  3: "#186673"}
        style_map = {0: '-', 1: '-', 2: '--', 3: '--'}
        """Compare subject and simulation data side by side"""
        # Create analyzers for both datasets
        subject_analyzer = DataAnalyzer(subject_data, is_subject_data=True)
        sim_analyzer = DataAnalyzer(simulation_data, is_subject_data=False)
        
        # Get learning curves for both
        subject_curves = subject_analyzer.calculate_learning_curves()
        sim_curves = sim_analyzer.calculate_learning_curves()
        
        # Standardize block type names
        block_type_mapping = {
            'HighControl': 'high',
            'LowControl': 'low', 
            'Yoked': 'yoked',
            'calibration': 'calibration'
        }
        
        # Apply mapping to subject data
        subject_curves['block_type'] = subject_curves['block_type'].replace(block_type_mapping)
        
        # Add data source labels
        subject_curves['data_source'] = 'Subject Data'
        sim_curves['data_source'] = 'Simulation Data'
        
        # Combine datasets
        combined_data = pd.concat([subject_curves, sim_curves], ignore_index=True)
        
        # Setup plotting
        sns.set_context('paper')
        sns.set_theme(style='whitegrid')

        
        # Get unique block types
        block_types = sorted(combined_data['block_type'].unique())
        n_blocks = len(block_types)
        
        fig, axes = plt.subplots(2, n_blocks, figsize=(5*n_blocks, 10), sharey=True)
        if n_blocks == 1:
            axes = axes.reshape(-1, 1)
        
        for j, data_source in enumerate(['Subject Data', 'Simulation Data']):
            for i, block_type in enumerate(block_types):
                ax = axes[j, i]
                plot_data = combined_data[
                    (combined_data['block_type'] == block_type) & 
                    (combined_data['data_source'] == data_source)
                ]
                
                labels_plotted = set()
                
                for stimulus in sorted(plot_data['stimulus'].unique()):
                    stim_data = plot_data[plot_data['stimulus'] == stimulus]
                    state_map = {0: 'go2win', 1: 'go2avoid', 2: 'nogo2win', 3: 'nogo2avoid'}
                    
                    label = state_map[stimulus] if state_map[stimulus] not in labels_plotted else None
                    if label is not None:
                        labels_plotted.add(label)
                    
                    x = stim_data['repetition'].astype(float).values
                    y = stim_data['p(Go)'].astype(float).values
                    y_err = stim_data['sem'].astype(float).values
                    
                    ax.plot(x, y, linestyle=style_map[stimulus], color=color_map[stimulus],
                           marker='o', linewidth=2.5, markersize=4, label=label)
                    ax.fill_between(x, y-y_err, y+y_err, color=color_map[stimulus], alpha=0.2)
                
                ax.axhline(y=0.5, color='k', linestyle='--', linewidth=1.5, alpha=0.3)
                ax.set_xlabel('State Repetition', fontsize=12)
                ax.set_title(f'{data_source} - {block_type} Control', fontsize=12)
                ax.set_ylim(-0.05, 1.05)
                
                if i == 0:
                    ax.set_ylabel('p(Go)', fontsize=12)
                if j == 0 and i == n_blocks - 1:
                    ax.legend(title='State', loc='best', frameon=True, fontsize=9)
        
        plt.suptitle('Subject vs Simulation Data Comparison', fontsize=16, y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def calculate_reward_contingency(self):
        """Calculate reward contingency data by block type, stimulus, and action correctness"""
        data = self.raw_data.copy()

        # Set up column names based on data type
        if self.is_subject_data:
            data = data[data['blocktype'] != 'calibration'].reset_index(drop=True)
            action_col = 'A'
            stimulus_col = 'S'
            reward_col = 'reward_recoded' if 'reward_recoded' in data.columns else 'valence' if 'valence' in data.columns else 'R'
            block_type_col = 'blocktype'
        else:
            data = data[data['block_type'] != 'calibration'].reset_index(drop=True)
            action_col = 'action'
            stimulus_col = 'stimulus'
            reward_col = 'reward'
            block_type_col = 'block_type'

        # Define optimal actions and favorable outcomes
        optimal_actions = {0: 1, 1: 1, 2: 0, 3: 0}  # go2win, go2avoid, nogo2win, nogo2avoid
        data['optimal_action'] = data[stimulus_col].map(optimal_actions)
        data['correct_action'] = (data[action_col] == data['optimal_action']).astype(int)
        data['favourable'] = (data[reward_col] == 1).astype(int)

        all_contingency_data = []

        for block_type in data[block_type_col].unique():
            block_data = data[data[block_type_col] == block_type].copy()
            
            for stimulus in block_data[stimulus_col].unique():
                stim_data = block_data[block_data[stimulus_col] == stimulus].copy()
                
                # Calculate proportions for correct and incorrect actions
                correct_data = stim_data[stim_data['correct_action'] == 1]
                incorrect_data = stim_data[stim_data['correct_action'] == 0]
                
                correct_favorable_prop = correct_data['favourable'].mean() if len(correct_data) > 0 else 0
                incorrect_favorable_prop = incorrect_data['favourable'].mean() if len(incorrect_data) > 0 else 0
                
                # Add data for correct actions
                if len(correct_data) > 0:
                    all_contingency_data.append({
                        'block_type': block_type,
                        'stimulus': stimulus,
                        'state_label': self.state_map[stimulus],
                        'action_type': 'Correct',
                        'favorable_proportion': correct_favorable_prop,
                        'n_trials': len(correct_data),
                        'contingency_difference': correct_favorable_prop - incorrect_favorable_prop
                    })
                
                # Add data for incorrect actions
                if len(incorrect_data) > 0:
                    all_contingency_data.append({
                        'block_type': block_type,
                        'stimulus': stimulus,
                        'state_label': self.state_map[stimulus],
                        'action_type': 'Incorrect',
                        'favorable_proportion': incorrect_favorable_prop,
                        'n_trials': len(incorrect_data),
                        'contingency_difference': correct_favorable_prop - incorrect_favorable_prop
                    })

        return pd.DataFrame(all_contingency_data)

    def plot_reward_contingency(self, save_path=None, title="Reward Contingency by Block Type and Action Correctness"):
        """Plot reward contingency data showing favorable outcome proportions for correct vs incorrect actions"""
        contingency_data = self.calculate_reward_contingency()
        
        # Setup plotting style
        sns.set_context('paper')
        sns.set_theme(style='whitegrid')
        mpl.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 14,
            'legend.fontsize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'figure.dpi': 300,
            'savefig.dpi': 300
        })

        # Get unique block types
        block_types = sorted(contingency_data['block_type'].unique())
        n_blocks = len(block_types)
        
        fig, axes = plt.subplots(1, n_blocks, figsize=(6*n_blocks, 6), sharey=True)
        if n_blocks == 1:
            axes = [axes]
        
        # Color mapping for action types
        action_colors = {'Correct': '#2E8B57', 'Incorrect': '#CD5C5C'}
        
        for i, block_type in enumerate(block_types):
            ax = axes[i]
            block_data = contingency_data[contingency_data['block_type'] == block_type]
            
            # Get unique state labels for this block
            state_labels = ['go2win', 'go2avoid', 'nogo2win', 'nogo2avoid']
            x_pos = np.arange(len(state_labels))
            width = 0.35
            
            # Prepare data for plotting
            correct_props = []
            incorrect_props = []
            
            for state_label in state_labels:
                state_data = block_data[block_data['state_label'] == state_label]
                
                correct_prop = state_data[state_data['action_type'] == 'Correct']['favorable_proportion'].values
                incorrect_prop = state_data[state_data['action_type'] == 'Incorrect']['favorable_proportion'].values
                
                correct_props.append(correct_prop[0] if len(correct_prop) > 0 else 0)
                incorrect_props.append(incorrect_prop[0] if len(incorrect_prop) > 0 else 0)
            
            # Create grouped bar plot
            bars1 = ax.bar(x_pos - width/2, correct_props, width, 
                          label='Correct Action', color=action_colors['Correct'], alpha=0.8)
            bars2 = ax.bar(x_pos + width/2, incorrect_props, width,
                          label='Incorrect Action', color=action_colors['Incorrect'], alpha=0.8)
            
            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=9)
            
            for bar in bars2:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=9)
            
            # Styling
            ax.set_xlabel('Stimulus Type', fontsize=14)
            ax.set_title(f'{block_type} Control', fontsize=14)
            ax.set_ylim(0, 1.05)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(state_labels, rotation=45)
            ax.axhline(y=0.5, color='k', linestyle='--', linewidth=1.5, alpha=0.3)
            ax.grid(True, alpha=0.3)
        
        # Add ylabel only to leftmost subplot
        axes[0].set_ylabel('Proportion of Favorable Outcomes', fontsize=14)
        
        # Add legend to the last subplot
        axes[-1].legend(loc='upper right', frameon=True)
        
        # Overall title
        wrapped_title = "\\n".join(textwrap.wrap(title, width=60))
        fig.suptitle(wrapped_title, fontsize=16, y=0.95)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def calculate_omega_stats(self):
        """calculate omega statistics by block type for models with omega tracking"""
        data = self.raw_data.copy()
        
        # check if omega data exists
        if 'omega_glob' not in data.columns and 'omega_state' not in data.columns and 'omega' not in data.columns:
            return None
        
        # set up column names based on data type
        if self.is_subject_data:
            data = data[data['blocktype'] != 'calibration'].reset_index(drop=True)
            block_type_col = 'blocktype'
            group_cols = ['subject', 'blocktype']
        else:
            data = data[data['block_type'] != 'calibration'].reset_index(drop=True)
            block_type_col = 'block_type'
            group_cols = ['simulation_id', 'block_type']
        
        omega_stats = []
        
        for block_type in data[block_type_col].unique():
            block_data = data[data[block_type_col] == block_type].copy()
            
            # calculate stats for each omega type
            if 'omega_glob' in block_data.columns:
                omega_glob_mean = block_data['omega_glob'].mean()
                omega_stats.append({
                    'block_type': block_type,
                    'omega_type': 'omega_glob',
                    'mean_omega': omega_glob_mean,
                    'std_omega': block_data['omega_glob'].std(),
                    'n_trials': len(block_data)
                })
            
            if 'omega_state' in block_data.columns:
                omega_state_mean = block_data['omega_state'].mean()
                omega_stats.append({
                    'block_type': block_type,
                    'omega_type': 'omega_state',
                    'mean_omega': omega_state_mean,
                    'std_omega': block_data['omega_state'].std(),
                    'n_trials': len(block_data)
                })
            
            if 'omega' in block_data.columns:
                omega_mean = block_data['omega'].mean()
                omega_stats.append({
                    'block_type': block_type,
                    'omega_type': 'omega',
                    'mean_omega': omega_mean,
                    'std_omega': block_data['omega'].std(),
                    'n_trials': len(block_data)
                })
        
        return pd.DataFrame(omega_stats)

    def plot_omega_by_control(self, save_path=None, title="Omega Values by Control Condition"):
        """plot omega values across control conditions"""
        omega_stats = self.calculate_omega_stats()
        
        if omega_stats is None or omega_stats.empty:
            print("No omega data found in dataset")
            return
        
        # setup plotting style
        sns.set_context('paper')
        sns.set_theme(style='whitegrid')
        mpl.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 14,
            'legend.fontsize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'figure.dpi': 300,
            'savefig.dpi': 300
        })

        # get unique omega types
        omega_types = omega_stats['omega_type'].unique()
        n_types = len(omega_types)
        
        fig, axes = plt.subplots(1, n_types, figsize=(5*n_types, 5), sharey=True)
        if n_types == 1:
            axes = [axes]
        
        for i, omega_type in enumerate(omega_types):
            ax = axes[i]
            type_data = omega_stats[omega_stats['omega_type'] == omega_type]
            
            # create bar plot
            x_pos = np.arange(len(type_data))
            bars = ax.bar(x_pos, type_data['mean_omega'], 
                         yerr=type_data['std_omega'],
                         alpha=0.8, capsize=5)
            
            # styling
            ax.set_xlabel('Control Condition', fontsize=14)
            ax.set_title(f'{omega_type.title()} Values', fontsize=14)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(type_data['block_type'])
            ax.set_ylim(0, 1)
            ax.axhline(y=0.5, color='k', linestyle='--', linewidth=1.5, alpha=0.3)
            
            # add value labels on bars
            for bar, value in zip(bars, type_data['mean_omega']):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        axes[0].set_ylabel('Mean Omega Value', fontsize=14)
        
        wrapped_title = "\\n".join(textwrap.wrap(title, width=60))
        fig.suptitle(wrapped_title, fontsize=16, y=0.98)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # print summary statistics
        print("\\nOmega Statistics Summary:")
        print("=" * 50)
        for _, row in omega_stats.iterrows():
            print(f"{row['omega_type']} in {row['block_type']}: {row['mean_omega']:.3f} ± {row['std_omega']:.3f}")

    def get_omega_summary(self):
        """get summary statistics for omega values"""
        omega_stats = self.calculate_omega_stats()
        
        if omega_stats is None or omega_stats.empty:
            return {"message": "No omega data found in dataset"}
        
        summary = {}
        for _, row in omega_stats.iterrows():
            key = f"{row['omega_type']}_{row['block_type']}"
            summary[key] = {
                'mean': row['mean_omega'],
                'std': row['std_omega'],
                'n_trials': row['n_trials']
            }
        
        return summary