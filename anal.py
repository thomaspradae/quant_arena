import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- CONFIG ---
OUTPUT_DIR = 'rigorous_regime_results_auto_n'
# -------------------------------------------------------------

def analyze_single_run(period: str, method: str):
    """Analyzes and visualizes the regimes for a single walk-forward period and method."""
    
    print("="*80)
    print(f"ANALYZING: Period '{period}', Method '{method}'")
    print("="*80)

    # 1. Load the full feature data
    try:
        df_full = pd.read_csv(os.path.join(OUTPUT_DIR, 'feature_data_for_analysis.csv'), 
                              index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("ERROR: 'feature_data_for_analysis.csv' not found.")
        print("Please run your main script first to generate this file.")
        return

    # 2. Load the specific regime labels
    labels_file = os.path.join(OUTPUT_DIR, f'{period}_{method}_test_labels.csv')
    try:
        # CORRECTED: Added header=0 to correctly read the CSV saved by the main script.
        regime_labels = pd.read_csv(labels_file, index_col=0, parse_dates=True, header=0, names=['date', 'regime'])
        
    except FileNotFoundError:
        print(f"ERROR: Label file not found: {labels_file}")
        print("Ensure the main script ran successfully and created this file.")
        return
    except Exception as e:
        print(f"ERROR: Failed to read or parse {labels_file}. Error: {e}")
        return


    # 3. Merge data
    df_merged = df_full.join(regime_labels, how='inner')
    if df_merged.empty:
        print("ERROR: Could not merge data. The indices did not align.")
        print(f"  Feature data index type: {df_full.index.dtype}")
        print(f"  Regime label index type: {regime_labels.index.dtype}")
        if not df_full.empty and not regime_labels.empty:
            print(f"  Sample feature index: {df_full.index[0]}")
            print(f"  Sample regime index: {regime_labels.index[0]}")
        return

    # 4. Calculate average feature "profile" for each regime
    regime_profiles = df_merged.groupby('regime').mean()

    # 5. Define key features for interpretation
    key_features = ['vol_63', 'mom_63', 'kurt_63', 'ret']
    existing_key_features = [f for f in key_features if f in regime_profiles.columns]

    print("\n--- Average Feature Profiles per Regime ---")
    print("This table tells you the 'personality' of each regime:")
    print(regime_profiles[existing_key_features])
    print("\nInterpretation Guide:")
    print("  - vol_63 (Volatility): High values = chaos/fear. Low values = calm.")
    print("  - mom_63 (Momentum): Positive = bull trend. Negative = bear trend.")
    print("  - kurt_63 (Kurtosis): High values = 'fat tails', high chance of extreme moves.")
    print("  - ret (Return): Average daily return in this regime.")

    # 6. Visualize with a heatmap
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))
    sns.heatmap(
        regime_profiles[existing_key_features].T, 
        annot=True, 
        cmap='viridis', 
        fmt='.3f',
        linewidths=.5
    )
    title = (f'Financial Profile of Statistical Regimes'
             f'\n({period.replace("_", " ").title()}, {method.upper()})')
    plt.title(title, fontsize=16)
    plt.xlabel('Statistical Regime Label', fontsize=12)
    plt.ylabel('Financial Feature', fontsize=12)
    plt.tight_layout()
    
    heatmap_filename = f'heatmap_{period}_{method}.png'
    plt.savefig(os.path.join(OUTPUT_DIR, heatmap_filename))
    print(f"\nSUCCESS: Heatmap saved to '{os.path.join(OUTPUT_DIR, heatmap_filename)}'")
    plt.close()

if __name__ == '__main__':
    for period in ['period_1', 'period_2', 'period_3']:
        for method in ['hmm', 'gmm', 'kmeans']:
            labels_file = os.path.join(OUTPUT_DIR, f'{period}_{method}_test_labels.csv')
            if os.path.exists(labels_file):
                analyze_single_run(period, method)
            else:
                print(f"\nSkipping analysis for {period}/{method} - label file not found.")