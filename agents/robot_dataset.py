# %%
import os
from pathlib import Path
import pickle as pkl
import pandas as pd
import glob
from tqdm.auto import tqdm
import numpy as np
from .subject_data import SubjectData
from typing import List, Optional

def detect_environment():
    """
    Detect if we're running on the cluster or locally with mounted drive
    Returns tuple: (is_cluster, base_path)
    """
    cwd = Path.cwd()
    
    # Method 1: Check if we're already in the project directory structure
    if 'gonogo-simfit' in str(cwd) and '3017083.01' in str(cwd):
        # We're in the project directory, find the root
        parts = cwd.parts
        project_idx = None
        for i, part in enumerate(parts):
            if part == '3017083.01':
                project_idx = i
                break
        
        if project_idx is not None:
            project_root = Path(*parts[:project_idx+1])
            return True, project_root
    
    # Method 2: Check for cluster-specific paths
    if str(cwd).startswith('/project/') or str(cwd).startswith('/home/'):
        # Likely on cluster, try to find project root
        project_path = Path('/project/3017083.01')
        if project_path.exists():
            return True, project_path
    
    # Method 3: Check for mounted drive paths
    mounted_paths = [
        Path('/mnt/z/3017083.01'),  # Samu mount
    ]
    
    for path in mounted_paths:
        if path.exists():
            return False, path
    
    # Method 4: Environment variable (if you set one)
    if 'CLUSTER_PROJECT_ROOT' in os.environ:
        path = Path(os.environ['CLUSTER_PROJECT_ROOT'])
        if path.exists():
            return True, path
    
    # Method 5: Relative path detection
    # Check if we can find the data file relative to current directory
    relative_data_path = Path('./datasets/Egbert_Sam_Roshan/fitdata.pkl')
    if relative_data_path.exists():
        return True, Path.cwd()
    
    # Check if we need to go up directories
    for i in range(1, 5):  # Check up to 4 directories up
        test_path = Path('../' * i + 'datasets/Egbert_Sam_Roshan/fitdata.pkl')
        if test_path.exists():
            return True, Path.cwd().parents[i-1]
    
    # Fallback: assume local with default mount
    fallback_path = Path('/mnt/z/3017083.01')
    print(f"Warning: Could not detect environment. Using fallback: {fallback_path}")
    return False, fallback_path

class RobotDataset:
    """Structure for holding the data for fitting"""

    def __init__(self, use_fmri=False, combined_fmri=False, recode_rewards=True, 
                 base_dir=None, debug_paths=False):

        # Detect environment if no base_dir provided
        if base_dir is None:
            self.is_cluster, self.base_dir = detect_environment()
        else:
            self.base_dir = Path(base_dir)
            self.is_cluster = not str(self.base_dir).startswith('/mnt/')
        
        if debug_paths:
            print(f"Environment: {'Cluster' if self.is_cluster else 'Local (mounted)'}")
            print(f"Base directory: {self.base_dir}")
            print(f"Current working directory: {Path.cwd()}")        
        self.use_fmri = use_fmri
            
        self.combined_fmri = combined_fmri
        self.behavioral_data_path = os.path.join(
             self.base_dir, "behavioral_study", "scripts", "gonogo-simfit", "datasets", "Egbert_Sam_Roshan", "fitdata.pkl")
        self.fmri_data_path = os.path.join( self.base_dir, "fmri", "behave", "raw")
        if debug_paths:
            print(f"Looking for behavioral data at: {self.behavioral_data_path}")
            print(f"File exists: {self.behavioral_data_path.exists()}")
            if use_fmri:
                print(f"Looking for fMRI data at: {self.fmri_data_path}")
                print(f"Directory exists: {self.fmri_data_path.exists()}")
        
        # Load data
        self.behavioral_data = self.load_behavioral_data()
        self.fmri_data = self.load_fmri_data() if use_fmri else None

        if use_fmri:
            if combined_fmri:
                self.data = self.combine_behavioral_and_fmri_data()
            else:
                self.data = self.fmri_data
        else:
            self.data = self.behavioral_data

        # Recode rewards properly based on state and actual reward outcome
        if recode_rewards:
            self._recode_rewards()

        self.observations = self.data['S'].values
        self.actions = self.data['A'].values
        if recode_rewards:
            self.rewards = self.data['reward_recoded'].values
        else:
            self.rewards = self.data['R'].values
        self.subjects = self.data['subject'].unique()
        self.validate_data()

        self.subject_datasets = self.data.groupby("subject").apply(
            lambda x: SubjectData(
                observations=x['S'].values,
                actions=x['A'].values,
                rewards=x['reward_recoded'].values if recode_rewards else x['R'].values,
                response_times = x['rt'].values,
                subject=x['subject'].iloc[0],
                step=x['step'].values,
                block_id = x['blockId'].values,
                block_type=x['blocktype'].values,
            )
        ).to_dict()

    def _recode_rewards(self):
        """
        Recode rewards properly based on state and actual reward outcome.
        Win states (0=go2win, 2=nogo2win) with reward=10 should be +1, else -1
        """
        # Win states are 0 (go2win) and 2 (nogo2win)
        data = self.data.copy()
        data['is_win_state'] = data['Context'].apply(lambda x: 'w' in x)
        data['is_lose_state'] = data['Context'].apply(lambda x: 'l' in x)
        
        # Convert boolean to +1/-1: True -> +1, False -> -1
        reward_boolean = ((data['is_win_state']) & (data['R'] == 10)) | ((data['is_lose_state']) & (data['R'] == 0))
        self.data['reward_recoded'] = reward_boolean.astype(int) * 2 - 1
        
        # Validation: ensure rewards are only +1 and -1
        unique_rewards = self.data['reward_recoded'].unique()
        expected_rewards = {-1, 1}
        if not set(unique_rewards).issubset(expected_rewards):
            raise ValueError(f"Reward recoding failed! Found values: {unique_rewards}, expected only: {expected_rewards}")
        print(f"✓ Reward recoding successful: {len(self.data)} trials with rewards {sorted(unique_rewards)}")
        
        # Additional validation: check reward distribution
        reward_counts = self.data['reward_recoded'].value_counts()
        print(f"  Reward distribution: +1={reward_counts.get(1, 0)}, -1={reward_counts.get(-1, 0)}")

    def from_dataframe(self, df: pd.DataFrame):
        """Create a RobotDataset from a pandas DataFrame."""
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        self.data = df
        self.observations = df['S'].values
        self.actions = df['A'].values
        self.rewards = df['valence'].values if 'valence' in df.columns else df['R'].values
        self.subjects = df['subject'].unique()
        self.validate_data()

        self.subject_datasets = df.groupby("subject").apply(
            lambda x: SubjectData(
                observations=x['S'].values,
                actions=x['A'].values,
                rewards=x['valence'].values if 'valence' in x.columns else x['R'].values,
                response_times=x.get('rt', np.nan).values,
                subject=x['subject'].iloc[0],
                step=x['step'].values,
                block_id=x.get('blockId', np.nan).values,
                block_type=x.get('blocktype', np.nan).values,
            )
        ).to_dict()

        return self
    def validate_data(self):
        """Validate that the loaded data is in the expected format."""
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame.")
        if self.data.empty:
            raise ValueError("Data cannot be empty.")

        lengths = [len(self.observations), len(self.actions), len(self.rewards)]

        if not all(l == lengths[0] for l in lengths):
            raise ValueError("All data arrays must have the same length")

    def load_behavioral_data(self):
        """Load behavioral data from the specified path."""

        df = pd.read_pickle(self.behavioral_data_path)

        data_list = []

        for subj_idx, subj_df in zip(df['subjectInfo'], df['agentMemory']):
            subj_df = subj_df.copy()
            subj_df['subject'] = subj_idx['subId']
            data_list.append(subj_df)

        return pd.concat(data_list, ignore_index=True)


    def load_fmri_data(self):
        """Load fMRI data from the specified directory."""

        def get_control_string(controllability, is_yoked):
            if controllability == "high":
                return "HighControl"
            elif controllability == "low" and not is_yoked:
                return "LowControl"
            elif controllability == "low" and is_yoked:
                return "Yoked"
            else:
                raise ValueError("Invalid controllability or yoked status")

        all_files = glob.glob(os.path.join(self.fmri_data_path, '*.csv'))
        if not all_files:
            raise FileNotFoundError(
                f"No CSV files found in {self.fmri_data_path}")

        print(f"Found {len(all_files)} files. Loading...")

        subjects_raw_dfs = []

        for subject_file in tqdm(all_files, total=len(all_files)):
            try:
                subj_df = pd.read_csv(subject_file, low_memory=False)

                subj_df = subj_df[subj_df['stimulusFont'].notna()].reset_index(
                    drop=True)

                if not subj_df.empty and len(subj_df) == 320:
                    subjects_raw_dfs.append(subj_df)
                else:
                    print(
                        f"Skipping {subject_file} due to unexpected length or empty DataFrame.")

            except Exception as e:
                print(
                    f"Error loading or processing file {os.path.basename(subject_file)}: {e}")

        if not subjects_raw_dfs:
            raise ValueError(
                "No valid subject data could be loaded or processed.")

        df_raw = pd.concat(subjects_raw_dfs, ignore_index=True)

        #print(f"Concatenated data shape: {df_raw.shape}")
        state_to_context = {
            0: 'gw',
            1: 'gal',
            2: 'ngw',
            3: 'ngal'
        }

        df_agentMemory_fmri = pd.DataFrame()

        df_agentMemory_fmri['subject'] = df_raw['participant'].astype(str)
        df_agentMemory_fmri['step'] = (np.arange(len(df_raw)) % 40).astype(int)
        df_agentMemory_fmri['S'] = (df_raw['condition'] - 1).astype(int)
        df_agentMemory_fmri['A'] = df_raw['key_resp.keys'].apply(
            lambda x: 1 if x == 1 else 0).astype(int)
        df_agentMemory_fmri['valence'] = df_raw['outcomeCondition'].apply(
            lambda x: 1 if x == 'reward' else -1).astype(int)
        df_agentMemory_fmri['R'] = df_raw['feedback'].astype(int)
        df_agentMemory_fmri['Snext'] = df_agentMemory_fmri[['R', 'S']].apply(
            lambda row: 4
            if (
                (row['R'] == -10 and row['S'] in [1, 3])
                or
                (row['R'] == 0 and row['S'] in [0, 2])
            )
            else 5,
            axis=1
        )
        df_agentMemory_fmri['bestAct'] = df_raw['correctResp'].apply(
            lambda x: 1 if x == 1 else 0).astype(int)
        df_agentMemory_fmri['rt'] = df_raw['key_resp.rt'].apply(
            lambda x: x if x != None else None).astype(float)
        df_agentMemory_fmri['blockId'] = (df_raw['miniBlock'] - 1).astype(int)

        df_agentMemory_fmri['blocktype'] = (
            df_raw
            .apply(lambda row: get_control_string(row['controllability'], row['Yoked']),
                   axis=1)
            .astype(str)
        )
        df_agentMemory_fmri['Context'] = df_raw['condition'].apply(
            lambda x: state_to_context[x-1])
        df_agentMemory_fmri['actNum'] = (
            np.arange(len(df_raw)) % 320).astype(int)
        df_agentMemory_fmri['Acc'] = df_raw['key_resp.corr'].astype(int)
        df_agentMemory_fmri['ALT'] = [[0, 1]] * len(df_agentMemory_fmri)
        df_agentMemory_fmri['controlRating'] = df_raw['ControlRating'].astype(
            int)
        return df_agentMemory_fmri

    def combine_behavioral_and_fmri_data(self):
        """Combine behavioral and fMRI data into a single DataFrame."""
        if self.behavioral_data is None or self.fmri_data is None:
            raise ValueError("Behavioral or fMRI data not loaded.")

        combined_data = pd.concat(
            [self.behavioral_data, self.fmri_data],
            ignore_index=True,
            axis=0
        )

        return combined_data

    def __repr__(self):
        return f"SubjectData(use_fmri={self.use_fmri}, combined_fmri={self.combined_fmri}, recode_rewards=True, subjects={self.subjects})"

    def __getitem__(self, subject):
        """Get the SubjectData for a specific subject."""
        if subject not in self.subject_datasets:
            raise KeyError(f"Subject {subject} not found in dataset.")
        return self.subject_datasets[subject]

    def __len__(self):
        """Get the number of subjects in the dataset."""
        return len(self.subjects)

    def __iter__(self):
        """Iterate over subjects in the dataset."""
        for subject in self.subjects:
            yield subject, self[subject]

    def save(self, path):
        """Save the dataset to a file."""
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        with open(path, 'wb') as f:
            pkl.dump(self, f)

    @classmethod
    def load(cls, path):
        """Load the dataset from a file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} does not exist.")

        with open(path, 'rb') as f:
            return pkl.load(f)

    def filter_subjects(self, subjects: List[str]) -> 'RobotDataset':
        """Filter the dataset to include only specified subjects."""
        if not isinstance(subjects, list):
            raise ValueError("Subjects must be provided as a list.")

        filtered_data = self.data[self.data['subject'].isin(subjects)].reset_index(drop=True)
        return RobotDataset(use_fmri=self.use_fmri, combined_fmri=self.combined_fmri, recode_rewards=True).from_dataframe(filtered_data)
# %%
