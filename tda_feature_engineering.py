
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
    import argparse
    parser = argparse.ArgumentParser(description="TDA feature engineering for a single symbol.")
    parser.add_argument("symbol", type=str, help="Symbol to process (e.g., XNO_USDT)")
    parser.add_argument("--window_sizes", type=int, nargs='+', default=[30, 60, 120], help="TDA window sizes to use")
    parser.add_argument("--stride", type=int, default=1, help="Stride for sliding window")
    parser.add_argument("--input_path", type=str, default=None, help="Path to input CSV (default: data/{symbol}.csv)")
    parser.add_argument("--output_path", type=str, default=None, help="Path to output features CSV (default: processed_data/{symbol}_features.csv)")
    args = parser.parse_args()

    crypto = args.symbol
    window_sizes = args.window_sizes
    stride = args.stride

    if args.input_path:
        filepath = args.input_path
    else:
        filepath = os.path.join('data', f'{crypto}.csv')
    print(f"Processing {crypto}...")
    if not os.path.exists(filepath):
        print(f"Data file not found: {filepath}. Exiting.")
        exit(1)

    df = pd.read_csv(filepath)
    time_series = df['close'].values

    all_features = []
    for ws in window_sizes:
        if ws > len(time_series):
            print(f"[WARN] Skipping window size {ws}: not enough data points ({len(time_series)}).")
            continue
        print(f"  - Creating features for window size {ws}...")
        tda_features = create_tda_features(time_series, ws, stride)
        feature_names = [f'tda_ws{ws}_h{i}' for i in range(tda_features.shape[1])]
        features_df = pd.DataFrame(tda_features, columns=feature_names)
        padding_size = len(time_series) - len(tda_features)
        padding = pd.DataFrame(np.nan, index=range(padding_size), columns=feature_names)
        features_df = pd.concat([padding, features_df]).reset_index(drop=True)
        all_features.append(features_df)

    crypto_features_df = pd.concat(all_features, axis=1)
    combined_df = pd.concat([df, crypto_features_df], axis=1)

    if args.output_path:
        output_filename = args.output_path
        output_dir = os.path.dirname(output_filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    else:
        output_dir = 'processed_data'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_filename = os.path.join(output_dir, f'{crypto}_features.csv')
    combined_df.to_csv(output_filename, index=False)
    print(f"Saved features for {crypto} to {output_filename}")
    print("TDA feature engineering complete.")
