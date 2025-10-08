import torch
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def load_pretrained_encoder_weights(model: torch.nn.Module, checkpoint_path: str, device):
    """
    Load pretrained weights from an autoencoder checkpoint into the encoder part of a model.
    Only loads matching keys with equal tensor sizes.
    Prints how many layers were updated.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get('state_dict', ckpt)
    model_dict = model.state_dict()
    # filter out unmatched layers
    pretrained_dict = {k: v for k, v in state_dict.items()
                       if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} layers from {checkpoint_path}")


def get_true_layers(df: pd.DataFrame, well_name: str):
    """
    From a processed DataFrame with 'WELL', 'DEPTH_MD', and 'GROUP' columns,
    compute contiguous layer intervals using vectorized pandas operations.

    Returns list of (top_depth, bottom_depth, class_label).
    """
    # Filter for the specific well and ensure it's sorted by depth
    print(f"print labels {df.columns}")
    sub = df[df['WELL'] == well_name].sort_values('DEPTH_MD')
    

    # Handle cases where the well has no data, preventing errors
    if sub.empty:
        return []

    # Identify blocks of contiguous identical layers using the 'GROUP' column.
    layer_blocks = (sub['LITHOLOGY'].ne(sub['LITHOLOGY'].shift())).cumsum()

    # Group by these contiguous blocks and aggregate to get layer properties
    intervals = sub.groupby(layer_blocks).agg(
        top_depth=('DEPTH_MD', 'min'),
        bottom_depth=('DEPTH_MD', 'max'),
        class_label=('LITHOLOGY', 'first')
    )

    # Convert the aggregated DataFrame to the desired list of tuples
    return list(intervals.itertuples(index=False, name=None))


def plot_well_correlation(df: pd.DataFrame,
                          df_labels: pd.DataFrame,
                          scaler,
                          model: torch.nn.Module,
                          reference: str,
                          targets: list,
                          threshold: float,
                          output_path: str = 'well_correlation_plot.png'):
    """
    For a reference well and list of target wells:
      1. Compute true layer intervals from GROUP labels.
      2. Encode each interval by passing features through the model and averaging the resulting embeddings.
      3. Compute similarity matrix via cosine similarity of interval embeddings.
      4. Plot two vertical tracks of intervals and draw connectors for sim >= threshold.
    """
    # Ensure the model is in evaluation mode
    model.eval()
    device = next(model.parameters()).device
    
    # 1. Prepare intervals from ground truth labels
    ref_ints = get_true_layers(df_labels, reference)
    target_ints = {w: get_true_layers(df_labels, w) for w in targets}

    def encode_intervals(well, intervals):
        print(f"\n--- Debugging Well: {well} ---")
        
        # Check if the dataframe slice for the well is empty
        sub = df[df['WELL'] == well].copy()
        if sub.empty:
            print(f"❌ ERROR: No data found for well '{well}'. Skipping.")
            return np.array([]) # Return an empty array
        print(f"  ✅ Found {sub.shape[0]} data points for this well.")

        try:
            # Prepare features for the model
            # Make sure the columns you drop actually exist in 'df'
            feats_raw = sub.drop(columns=['WELL', 'DEPTH_MD']).values
            print(f"  ✅ Raw features shape: {feats_raw.shape}")
            
            feats_scaled = scaler.transform(feats_raw)
            print(f"  ✅ Scaled features shape: {feats_scaled.shape}")
            
            # Create tensor, add batch dimension (B,C,L), and send to device
            feats_tensor = torch.from_numpy(feats_scaled.T).float().unsqueeze(0).to(device)
            print(f"  ✅ Input tensor shape for model: {feats_tensor.shape}")

            # --- Get embeddings from the model ---
            with torch.no_grad():
                model_output = model(feats_tensor)
                
                # Check what the model actually returned
                print(f"  ✅ Model output type: {type(model_output)}")

                # If the model returns a dictionary, find the correct key for your embeddings
                if isinstance(model_output, dict):
                    print(f"  ✅ Model output keys: {list(model_output.keys())}")
                    # ❗ IMPORTANT: Replace 'pred_logits' with the correct key from your model's output!
                    if 'pred_logits' not in model_output:
                        print("  ❌ ERROR: The key 'pred_logits' was not found in the model's output dictionary.")
                        # You might want to raise an error here or return
                        return np.array([])
                    embeddings_tensor = model_output['pred_logits'] 
                else:
                    # If the model returns a single tensor
                    embeddings_tensor = model_output

                print(f"  ✅ Extracted embeddings tensor shape: {embeddings_tensor.shape}")
                embeddings = embeddings_tensor.squeeze(0).cpu().numpy().T
                print(f"  ✅ Final numpy embeddings shape (L, D): {embeddings.shape}")

            # --- Average the model's embeddings for each interval ---
            # Crucial check: Does the length of embeddings match the number of data points?
            if embeddings.shape[0] != len(sub):
                print(f"  ❌ ERROR: Length mismatch! Number of embeddings ({embeddings.shape[0]}) does not match number of data points ({len(sub)}).")
                return np.array([])

            reps = []
            print("  ✅ Averaging embeddings for each interval...")
            for i, (top, bottom, _) in enumerate(intervals):
                mask = (sub['DEPTH_MD'] >= top) & (sub['DEPTH_MD'] <= bottom)
                
                # Ensure there are some embeddings to average for the interval
                if not mask.any():
                    print(f"    - Interval {i+1} ({top}-{bottom}m): No data points found, skipping.")
                    continue
                    
                reps.append(embeddings[mask.values].mean(axis=0))
                print(f"    - Interval {i+1} ({top}-{bottom}m): Averaged {mask.sum()} embeddings.")
                
            return np.vstack(reps)

        except KeyError as e:
            print(f"❌ Key Error: A column is missing from your DataFrame. Error: {e}")
            return np.array([])
        except Exception as e:
            print(f"❌ An unexpected error occurred: {e}")
            return np.array([])
    ref_feats = encode_intervals(reference, ref_ints)
    target_feats = {w: encode_intervals(w, ints) for w, ints in target_ints.items()}

    # 3. Similarity: cosine similarity (this part remains the same)
    def cosine_sim(a, b):
        a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
        return a_norm @ b_norm.T

    sims = {w: cosine_sim(ref_feats, feats) for w, feats in target_feats.items()}

    # 4. Plot (this part remains the same)
    fig, ax = plt.subplots(figsize=(4 + len(targets) * 2, 8))
    min_depth, max_depth = df['DEPTH_MD'].min(), df['DEPTH_MD'].max()
    ax.invert_yaxis()

    # draw reference track at x=0
    for top, bottom, _ in ref_ints:
        ax.plot([0, 0], [top, bottom], lw=12, solid_capstyle='butt')

    # draw target tracks and connectors
    for idx, w in enumerate(targets, start=1):
        x = idx * 2
        for top, bottom, _ in target_ints[w]:
            ax.plot([x, x], [top, bottom], lw=12, solid_capstyle='butt')
        # connectors
        sim = sims[w]
        for i in range(sim.shape[0]):
            for j in range(sim.shape[1]):
                if sim[i, j] >= threshold:
                    y1 = (ref_ints[i][0] + ref_ints[i][1]) / 2
                    y2 = (target_ints[w][j][0] + target_ints[w][j][1]) / 2
                    ax.plot([0, x], [y1, y2], c='gray', linestyle='--', alpha=0.5)

    ax.set_xlim(-1, len(targets) * 2 + 1)
    ax.set_ylim(min_depth, max_depth)
    ax.set_xticks([0] + [i * 2 for i in range(1, len(targets) + 1)])
    ax.set_xticklabels([reference] + targets)
    ax.set_ylabel('Depth (MD)')
    ax.set_title('Well-to-Well Layer Correlation (Model-Based)')

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved model-based correlation plot to {output_path}")


def plot_model_predictions(df: pd.DataFrame,
                           scaler,
                           model: torch.nn.Module,
                           wells_to_plot: list,
                           output_path: str = 'model_predictions.png',
                           confidence_threshold: float = 0.7):
    """
    Runs the detection model on a list of wells, prints a detailed score analysis,
    and plots the predicted layer boundaries over the Gamma Ray (GR) log.
    """
    model.eval()
    device = next(model.parameters()).device
    num_wells = len(wells_to_plot)
    fig, axes = plt.subplots(1, num_wells, figsize=(num_wells * 4, 12), sharey=True)
    if num_wells == 1:
        axes = [axes]

    print("--- Running New Analysis: Plotting Model Predictions ---")

    for i, well_name in enumerate(wells_to_plot):
        ax = axes[i]
        print(f"\nProcessing well: {well_name}...")
        print(df.columns)
        # Select the 'WELL' column and call the .unique() method
        unique_wells = df['WELL'].unique()

        # Print the array of unique well names
        print(unique_wells)
        # 1. Prepare data for the specific well
        sub = df[df['WELL'] == well_name].copy()
        if sub.empty:
            print(f"  - No data found for well {well_name}. Skipping.")
            ax.set_title(f"{well_name}\n(No Data)")
            continue

        depths = sub['DEPTH_MD'].values
        min_depth, max_depth = depths.min(), depths.max()
        
        feature_cols = [col for col in df.columns if col not in ['WELL', 'DEPTH_MD']]
        #feature_cols = [col for col in df.columns if col not in ['WELL', 'DEPTH_MD', 'FORMATION', 'LITHOLOGY', 'GROUP']]
        feats_raw = sub[feature_cols].values
        feats_scaled = scaler.transform(feats_raw)
        feats_tensor = torch.from_numpy(feats_scaled.T).float().unsqueeze(0).to(device)

        # 2. Run model inference
        with torch.no_grad():
            predictions = model(feats_tensor)

        # 3. Post-process and ANALYZE the model's predictions
        logits = predictions['pred_logits'].squeeze(0)
        boxes = predictions['pred_boxes'].squeeze(0)

        
        
        # Get probabilities for all classes
        probs = logits.softmax(dim=-1)
        
        # --- DETAILED SCORE ANALYSIS ---
        print("  --- Detailed Prediction Analysis ---")
        # Assume the last class is the "no object" class, as is common in DETR
        num_litho_classes = probs.shape[1] - 1 
        
        # Separate probabilities for actual lithologies and the "no object" class
        litho_probs = probs[:, :num_litho_classes]
        no_object_prob = probs[:, -1]

        # Find the top lithology class and its score for each of the 50 predictions
        top_litho_scores, top_litho_labels = litho_probs.max(dim=-1)

        # Print the results


        for j in range(len(top_litho_scores)):
            print(f"    Prediction {j+1:2d}: Top Lithology Class: {top_litho_labels[j].item()}, Score: {top_litho_scores[j].item():.4f} | 'No-Object' Score: {no_object_prob[j].item():.4f}")
        
        # --- FILTERING FOR PLOTTING ---
        # Keep a prediction only if its top LITHOLOGY score is above the threshold
        keep = top_litho_scores > confidence_threshold
        
        confident_boxes = boxes[keep].cpu().numpy()
        confident_labels = top_litho_labels[keep].cpu().numpy()
        
        print(f"\n  - Found {len(confident_boxes)} confident predictions (score > {confidence_threshold}).")

        # 4. Plot the results
        ax.plot(sub['GR'], depths, color='black', lw=0.8)
        ax.set_xlabel("Gamma Ray (GR)")
        ax.set_title(well_name)
        ax.grid(True, linestyle='--', alpha=0.5)

        # Use the number of actual lithology classes for the color map
        cmap = plt.get_cmap('viridis', num_litho_classes)
        norm = mcolors.Normalize(vmin=0, vmax=num_litho_classes)

        for box, label in zip(confident_boxes, confident_labels):
            _, center_y, _, height = box
            center_depth = min_depth + center_y * (max_depth - min_depth)
            height_depth = height * (max_depth - min_depth)
            top = center_depth - (height_depth / 2)
            bottom = center_depth + (height_depth / 2)
            ax.axhspan(top, bottom, color=cmap(norm(label)), alpha=0.3, ec='none')

    axes[0].set_ylabel("Depth (MD)")
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"\n✅ Analysis complete. Plot saved to {output_path}")