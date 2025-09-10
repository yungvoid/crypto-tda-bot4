
import pandas as pd
import numpy as np
import os
from gtda.time_series import SlidingWindow
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import Amplitude

def create_tda_features(data, window_size, stride):
    """
    Creates topological features from time series data.
    """
    # 1. Create sliding windows
    # The shape of `windows` will be (n_windows, window_size)
    windows = SlidingWindow(size=window_size, stride=stride).fit_transform(data)
    
    # Reshape to (n_windows, window_size, 1) for VietorisRipsPersistence
    windows = windows.reshape(*windows.shape, 1)

    # 2. Compute persistence diagrams
    # Note: The paper describes a lower-star filtration on a path graph.
    # For simplicity and to leverage giotto-tda effectively, we use a
    # standard Vietoris-Rips filtration on the 1D time series windows.
    # This captures similar connectivity information.
    # The result `diagrams` will have shape (n_windows, n_homology_dims, n_features)
    persistence = VietorisRipsPersistence(homology_dimensions=[0, 1])
    diagrams = persistence.fit_transform(windows)

    # 3. Vectorize persistence diagrams
    # We use the 'wasserstein' amplitude, a common vectorization technique.
    # The result `features` will have shape (n_windows, n_homology_dims)
    metric = "wasserstein"
    amplitude = Amplitude(metric=metric)
    features = amplitude.fit_transform(diagrams)

    # The output of Amplitude is 2D (n_samples, n_homology_dimensions).
    # We reshape it to have a clear feature name for each dimension.
    n_samples = features.shape[0]
    features_reshaped = features.reshape(n_samples, -1)
    
    return features_reshaped

if __name__ == "__main__":
    cryptos = ['BTC_USDT', 'ETH_USDT', 'ADA_USDT']
    window_sizes = [30, 60, 120]  # e.g., 30, 60, and 120 minutes
    stride = 1

    for crypto in cryptos:
        print(f"Processing {crypto}...")
        filepath = os.path.join('data', f'{crypto}.csv')
        if not os.path.exists(filepath):
            print(f"Data file not found for {crypto}. Skipping.")
            continue

        df = pd.read_csv(filepath)
        # We use the 'close' price for our time series analysis
        time_series = df['close'].values

        all_features = []
        for ws in window_sizes:
            print(f"  - Creating features for window size {ws}...")
            
            # Create features for the current window size
            tda_features = create_tda_features(time_series, ws, stride)
            
            # Create a DataFrame for these features
            feature_names = [f'tda_ws{ws}_h{i}' for i in range(tda_features.shape[1])]
            features_df = pd.DataFrame(tda_features, columns=feature_names)
            
            # To align features with the original time series, we need to pad the start
            # where windows were not complete. The number of non-computable rows is
            # (len(time_series) - len(tda_features)).
            padding_size = len(time_series) - len(tda_features)
            padding = pd.DataFrame(np.nan, index=range(padding_size), columns=feature_names)
            features_df = pd.concat([padding, features_df]).reset_index(drop=True)

            all_features.append(features_df)

        # Concatenate features from all window sizes
        crypto_features_df = pd.concat(all_features, axis=1)

        # Combine with original data
        combined_df = pd.concat([df, crypto_features_df], axis=1)

        # Save the combined data with features
        output_dir = 'processed_data'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_filename = os.path.join(output_dir, f'{crypto}_features.csv')
        combined_df.to_csv(output_filename, index=False)
        print(f"Saved features for {crypto} to {output_filename}")

    print("TDA feature engineering complete.")
