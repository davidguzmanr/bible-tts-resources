import os
# Disable Xet storage which can cause timeout issues on some networks
os.environ["HF_HUB_DISABLE_XET"] = "1"
# Increase download timeout
os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"

import torchaudio

# Patch for mismatch between speechbrain and torchaudio>=2.9
# torchaudio.list_audio_backends() was deprecated/removed in torchaudio 2.9
# See: https://github.com/speechbrain/speechbrain/issues/3012
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: [""]  # type: ignore

# Patch for mismatch between speechbrain and huggingface_hub>=0.24
# 'use_auth_token' was deprecated/removed in favor of 'token'
# See: https://github.com/speechbrain/speechbrain/issues/2614
import huggingface_hub
_original_hf_hub_download = huggingface_hub.hf_hub_download
def _patched_hf_hub_download(*args, **kwargs):
    if "use_auth_token" in kwargs:
        kwargs["token"] = kwargs.pop("use_auth_token")
    return _original_hf_hub_download(*args, **kwargs)
huggingface_hub.hf_hub_download = _patched_hf_hub_download

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from speechbrain.inference.speaker import SpeakerRecognition
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class AdvancedSpeakerIdentifier:
    """
    Identifies speakers in a TTS dataset using pretrained speaker embeddings.
    """
    
    def __init__(self, df, device=None):
        """
        Args:
            df: DataFrame with 'audio_file' and 'text' columns
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.df = df.copy()
        self.embeddings = []
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        print(f"Using device: {self.device}")
        
        print("Loading pretrained speaker recognition model...")
        
        # Create savedir and dummy custom.py to work around missing file in HF repo
        savedir = "pretrained_models/spkrec-ecapa-voxceleb"
        os.makedirs(savedir, exist_ok=True)
        custom_py_path = os.path.join(savedir, "custom.py")
        if not os.path.exists(custom_py_path):
            with open(custom_py_path, "w") as f:
                f.write("# Placeholder file\n")
        
        self.classifier = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=savedir,
            run_opts={"device": self.device},
        )
        print("Model loaded successfully!")
        
    def _load_audio(self, audio_path, sr=16000):
        """
        Load and preprocess a single audio file.
        Returns the signal tensor (1D) or None if loading fails.
        """
        try:
            # Load audio using soundfile (more reliable than torchaudio)
            signal, fs = sf.read(str(audio_path))
            
            # Convert to torch tensor
            signal = torch.tensor(signal, dtype=torch.float32)
            
            # Handle stereo: take first channel
            if signal.dim() == 2:
                signal = signal[:, 0] if signal.shape[1] < signal.shape[0] else signal[0, :]
            
            # Resample if needed
            if fs != sr:
                signal = signal.unsqueeze(0)  # Add channel dim for resampler
                resampler = torchaudio.transforms.Resample(fs, sr)
                signal = resampler(signal).squeeze(0)
            
            return signal
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None
    
    def extract_speaker_embedding(self, audio_path, sr=16000):
        """
        Extract speaker embeddings using pretrained model (single file).
        """
        signal = self._load_audio(audio_path, sr)
        if signal is None:
            return None
        
        try:
            # Add batch dimension and move to device
            signal = signal.unsqueeze(0).to(self.device)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.classifier.encode_batch(signal)
            
            return embedding.squeeze().cpu().numpy()
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None
    
    def process_dataset(self, batch_size=32):
        """
        Extract embeddings from all audio files using batched GPU processing.
        
        Args:
            batch_size: Number of audio files to process at once (default: 32)
        """
        print(f"\nProcessing {len(self.df)} audio files with batch_size={batch_size}...")
        
        # First pass: load all audio files
        audio_data = []
        indices = []
        print("Loading audio files...")
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Loading audio"):
            signal = self._load_audio(row['audio_file'])
            if signal is not None:
                audio_data.append(signal)
                indices.append(idx)
        
        print(f"Loaded {len(audio_data)} audio files successfully")
        
        # Sort by length to minimize padding within batches
        # This prevents the "32-bit index math" error from very long padded tensors
        sorted_order = np.argsort([s.shape[0] for s in audio_data])
        audio_data = [audio_data[i] for i in sorted_order]
        indices = [indices[i] for i in sorted_order]
        
        # Dictionary to store embeddings with their original indices
        embeddings_dict = {}
        
        # Second pass: process in batches
        print("Extracting embeddings in batches...")
        for i in tqdm(range(0, len(audio_data), batch_size), desc="Processing batches"):
            batch_signals = audio_data[i:i+batch_size]
            batch_indices = indices[i:i+batch_size]
            
            # Pad signals to same length within batch
            max_len = max(s.shape[0] for s in batch_signals)
            padded_batch = torch.zeros(len(batch_signals), max_len)
            wav_lens = torch.zeros(len(batch_signals))
            
            for j, sig in enumerate(batch_signals):
                padded_batch[j, :sig.shape[0]] = sig
                wav_lens[j] = sig.shape[0] / max_len  # Relative length for masking
            
            # Move to device and extract embeddings
            padded_batch = padded_batch.to(self.device)
            wav_lens = wav_lens.to(self.device)
            
            with torch.no_grad():
                embeddings = self.classifier.encode_batch(padded_batch, wav_lens)
            
            # Store results with original indices
            embeddings_np = embeddings.squeeze(1).cpu().numpy()
            for j, idx in enumerate(batch_indices):
                embeddings_dict[idx] = embeddings_np[j]
        
        # Restore original order
        valid_indices = sorted(embeddings_dict.keys())
        all_embeddings = [embeddings_dict[idx] for idx in valid_indices]
        
        self.embeddings = np.array(all_embeddings)
        self.df = self.df.loc[valid_indices].reset_index(drop=True)
        
        print(f"\nSuccessfully processed {len(self.embeddings)} files")
        print(f"Embedding shape: {self.embeddings.shape}")
        
    def estimate_num_speakers(self, max_speakers=5, single_speaker_threshold=0.85, similarity_sample_size=5000):
        """
        Estimate the number of speakers using silhouette score.
        Also checks if there might be only 1 speaker based on embedding similarity.
        
        Args:
            max_speakers: Maximum number of speakers to consider
            single_speaker_threshold: Cosine similarity threshold above which 
                                      we consider it a single speaker (default: 0.85)
            similarity_sample_size: Number of samples to use for cosine similarity 
                                    computation (to avoid memory issues with large datasets)
        
        Returns the estimated number and silhouette scores for visualization.
        """
        if len(self.embeddings) < 2:
            print("Only one audio file, assuming 1 speaker")
            return 1, {}
        
        # First, check if embeddings are very similar (suggesting 1 speaker)
        # Compute average pairwise cosine similarity (with sampling for large datasets)
        from sklearn.metrics.pairwise import cosine_similarity
        
        if len(self.embeddings) > similarity_sample_size:
            print(f"Sampling {similarity_sample_size} embeddings for similarity computation...")
            sample_idx = np.random.choice(len(self.embeddings), similarity_sample_size, replace=False)
            sample_embeddings = self.embeddings[sample_idx]
        else:
            sample_embeddings = self.embeddings
        
        cos_sim_matrix = cosine_similarity(sample_embeddings)
        # Get upper triangle (excluding diagonal)
        upper_tri_idx = np.triu_indices_from(cos_sim_matrix, k=1)
        avg_similarity = cos_sim_matrix[upper_tri_idx].mean()
        
        print(f"\nAverage pairwise cosine similarity: {avg_similarity:.4f}")
        
        # Use sampled embeddings for clustering evaluation too (memory efficiency)
        eval_embeddings = sample_embeddings
        
        max_k = min(max_speakers, len(eval_embeddings) - 1)
        K_range = range(2, max_k + 1)
        silhouette_scores = {}
        
        print("\nEstimating number of speakers...")
        print(f"  k=1 speaker → avg cosine similarity={avg_similarity:.4f}" + 
              (" (high similarity suggests single speaker)" if avg_similarity >= single_speaker_threshold else ""))
        
        for k in K_range:
            clustering = AgglomerativeClustering(n_clusters=k, linkage='ward')
            labels = clustering.fit_predict(eval_embeddings)
            score = silhouette_score(eval_embeddings, labels)
            silhouette_scores[k] = score
            print(f"  k={k} speakers → silhouette score={score:.4f}")
        
        # Decide if it's 1 speaker or use silhouette scores
        if avg_similarity >= single_speaker_threshold:
            print(f"\n✓ Estimated number of speakers: 1")
            print(f"  (High avg cosine similarity {avg_similarity:.4f} >= {single_speaker_threshold} suggests single speaker)")
            return 1, silhouette_scores
        elif silhouette_scores:
            best_k = max(silhouette_scores, key=silhouette_scores.get)
            print(f"\n✓ Estimated number of speakers: {best_k}")
            print(f"  (Best silhouette score: {silhouette_scores[best_k]:.4f})")
            return best_k, silhouette_scores
        else:
            return 1, {}
    
    def cluster_speakers(self, n_speakers=None):
        """
        Cluster audio files by speaker.
        """
        if n_speakers is None:
            n_speakers, _ = self.estimate_num_speakers()
        
        print(f"\nClustering with {n_speakers} speakers...")
        
        if n_speakers == 1:
            # Single speaker - assign all to speaker_0
            speaker_labels = np.zeros(len(self.embeddings), dtype=int)
        else:
            clustering = AgglomerativeClustering(n_clusters=n_speakers, linkage='ward')
            speaker_labels = clustering.fit_predict(self.embeddings)
        
        # Add speaker_id to dataframe
        self.df['speaker_id'] = [f'speaker_{label}' for label in speaker_labels]
        
        # Print distribution
        print("\n" + "="*50)
        print("SPEAKER DISTRIBUTION")
        print("="*50)
        distribution = self.df['speaker_id'].value_counts().sort_index()
        for speaker, count in distribution.items():
            percentage = (count / len(self.df)) * 100
            print(f"{speaker}: {count} files ({percentage:.1f}%)")
        print("="*50)
        
        return speaker_labels
    
    def visualize_clusters(self, method='tsne', figsize=(15, 10)):
        """
        Visualize speaker clusters using dimensionality reduction.
        
        Args:
            method: 'pca' or 'tsne'
            figsize: Figure size tuple
        """
        if 'speaker_id' not in self.df.columns:
            print("Please run cluster_speakers() first!")
            return
        
        print(f"\nGenerating {method.upper()} visualization...")
        
        # Reduce dimensions to 2D
        if method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(self.embeddings)
            explained_var = reducer.explained_variance_ratio_
            title_suffix = f"(Explained variance: {sum(explained_var):.1%})"
        else:  # tsne
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(self.embeddings)-1))
            embeddings_2d = reducer.fit_transform(self.embeddings)
            title_suffix = ""
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Scatter plot with speaker colors
        unique_speakers = sorted(self.df['speaker_id'].unique())
        colors = sns.color_palette('husl', n_colors=len(unique_speakers))
        
        for idx, speaker in enumerate(unique_speakers):
            mask = self.df['speaker_id'] == speaker
            axes[0].scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[colors[idx]],
                label=speaker,
                alpha=0.6,
                s=100,
                edgecolors='black',
                linewidth=0.5
            )
        
        axes[0].set_xlabel(f'{method.upper()} Component 1', fontsize=12)
        axes[0].set_ylabel(f'{method.upper()} Component 2', fontsize=12)
        axes[0].set_title(f'Speaker Clusters Visualization {title_suffix}', fontsize=14, fontweight='bold')
        axes[0].legend(loc='best', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Density plot
        for idx, speaker in enumerate(unique_speakers):
            mask = self.df['speaker_id'] == speaker
            axes[1].scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[colors[idx]],
                alpha=0.3,
                s=50
            )
        
        # Add density contours
        from scipy.stats import gaussian_kde
        for idx, speaker in enumerate(unique_speakers):
            mask = self.df['speaker_id'] == speaker
            if mask.sum() > 2:  # Need at least 3 points for KDE
                try:
                    xy = embeddings_2d[mask]
                    kde = gaussian_kde(xy.T)
                    x_min, x_max = xy[:, 0].min(), xy[:, 0].max()
                    y_min, y_max = xy[:, 1].min(), xy[:, 1].max()
                    xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
                    positions = np.vstack([xx.ravel(), yy.ravel()])
                    f = np.reshape(kde(positions).T, xx.shape)
                    axes[1].contour(xx, yy, f, colors=[colors[idx]], alpha=0.5, linewidths=2)
                except:
                    pass
        
        axes[1].set_xlabel(f'{method.upper()} Component 1', fontsize=12)
        axes[1].set_ylabel(f'{method.upper()} Component 2', fontsize=12)
        axes[1].set_title('Speaker Clusters with Density Contours', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        # plt.savefig(f'speaker_clusters_{method}.png', dpi=300, bbox_inches='tight')
        # print(f"✓ Visualization saved as 'speaker_clusters_{method}.png'")
        plt.show()
    
    def plot_silhouette_scores(self, silhouette_scores):
        """
        Plot silhouette scores for different numbers of clusters.
        """
        if not silhouette_scores:
            print("No silhouette scores to plot")
            return
        
        plt.figure(figsize=(10, 6))
        k_values = list(silhouette_scores.keys())
        scores = list(silhouette_scores.values())
        
        plt.plot(k_values, scores, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Number of Speakers', fontsize=12)
        plt.ylabel('Silhouette Score', fontsize=12)
        plt.title('Silhouette Score vs Number of Speakers', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Highlight the best score
        best_k = max(silhouette_scores, key=silhouette_scores.get)
        best_score = silhouette_scores[best_k]
        plt.scatter([best_k], [best_score], c='red', s=200, zorder=5, 
                   label=f'Best: k={best_k} (score={best_score:.4f})')
        plt.legend(fontsize=10)
        
        plt.tight_layout()
        # plt.savefig('silhouette_scores.png', dpi=300, bbox_inches='tight')
        # print("✓ Silhouette scores plot saved as 'silhouette_scores.png'")
        plt.show()
    
    def show_sample_texts_per_speaker(self, n_samples=5):
        """
        Display sample texts for each speaker to verify clustering.
        """
        if 'speaker_id' not in self.df.columns:
            print("Please run cluster_speakers() first!")
            return
        
        print("\n" + "="*70)
        print("SAMPLE TEXTS PER SPEAKER")
        print("="*70)
        
        for speaker in sorted(self.df['speaker_id'].unique()):
            speaker_df = self.df[self.df['speaker_id'] == speaker]
            samples = speaker_df.head(n_samples)
            
            print(f"\n{speaker.upper()} ({len(speaker_df)} total files):")
            print("-" * 70)
            for idx, (_, row) in enumerate(samples.iterrows(), 1):
                text_preview = row['text'][:80] + "..." if len(row['text']) > 80 else row['text']
                print(f"  {idx}. {text_preview}")
                print(f"     File: {row['audio_file']}")
    
    def get_result_dataframe(self):
        """
        Return the dataframe with speaker_id column.
        """
        if 'speaker_id' not in self.df.columns:
            print("Please run cluster_speakers() first!")
            return None
        return self.df


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    # Assuming you have a DataFrame like this:
    # df = pd.DataFrame({
    #     'audio_file': ['path/to/audio1.wav', 'path/to/audio2.wav', ...],
    #     'text': ['Hello world', 'How are you', ...]
    # })
    
    # Or load from CSV:
    # df = pd.read_csv('your_dataset.csv')
    
    # Initialize the identifier
    identifier = AdvancedSpeakerIdentifier(df)
    
    # Step 1: Extract embeddings
    identifier.process_dataset()
    
    # Step 2: Estimate number of speakers (optional)
    n_speakers, silhouette_scores = identifier.estimate_num_speakers(max_speakers=10)
    
    # Plot silhouette scores
    identifier.plot_silhouette_scores(silhouette_scores)
    
    # Step 3: Cluster speakers (use estimated or specify manually)
    # Option A: Use estimated number
    speaker_labels = identifier.cluster_speakers(n_speakers=n_speakers)
    
    # Option B: Manually specify if you know
    # speaker_labels = identifier.cluster_speakers(n_speakers=3)
    
    # Step 4: Visualize clusters
    identifier.visualize_clusters(method='tsne')  # or 'pca'
    identifier.visualize_clusters(method='pca')
    
    # Step 5: Show sample texts per speaker
    identifier.show_sample_texts_per_speaker(n_samples=5)
    
    # Step 6: Get the result dataframe with speaker_id
    result_df = identifier.get_result_dataframe()
    
    # Save to CSV
    result_df.to_csv('dataset_with_speakers.csv', index=False)
    print("\n✓ Dataset with speaker IDs saved to 'dataset_with_speakers.csv'")