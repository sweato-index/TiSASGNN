import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import os
import time
from collections import defaultdict
from tqdm import tqdm
import random
import sys
import warnings
warnings.filterwarnings('ignore')

# Import the TiSASGNN model
from TiSASGNN_PLUS import TiSASGNN, train_tisasgnn

# Set random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Data processing utilities
def load_data(file_path):
    """Load data from tab-separated txt file"""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, sep='\t', header=None, 
                    names=['user_id', 'item_id', 'rating', 'timestamp'])
    
    print(f"Raw data shape: {df.shape}")
    print(f"Number of users: {df['user_id'].nunique()}")
    print(f"Number of items: {df['item_id'].nunique()}")
    
    # Converting IDs to zero-based index if they aren't already
    user_map = {u: i+1 for i, u in enumerate(df['user_id'].unique())}  # Start from 1, reserve 0 for padding
    item_map = {i: j+1 for j, i in enumerate(df['item_id'].unique())}  # Start from 1, reserve 0 for padding
    
    df['user_idx'] = df['user_id'].map(user_map)
    df['item_idx'] = df['item_id'].map(item_map)
    
    return df, user_map, item_map

def create_sequences(df, max_len=50):
    """Create user sequences from dataframe"""
    # Sort by user and timestamp
    df = df.sort_values(['user_idx', 'timestamp'])
    
    # Group by user to create sequences
    user_sequences = {}
    user_time_sequences = {}
    user_rating_sequences = {}

    for user_id, group in tqdm(df.groupby('user_idx'), desc="Creating sequences"):
        # Get item sequence and timestamps
        item_sequence = group['item_idx'].tolist()
        time_sequence = group['timestamp'].tolist()
        rating_sequence = group['rating'].tolist()

        user_sequences[user_id] = item_sequence
        user_time_sequences[user_id] = time_sequence
        user_rating_sequences[user_id] = rating_sequence
    # Create time difference matrices
    user_time_diff_matrices = {}
    
    for user_id, time_seq in tqdm(user_time_sequences.items(), desc="Creating time matrices"):
        seq_len = len(time_seq)
        if seq_len <= 1:
            continue
            
        # Create time difference matrix
        time_matrix = np.zeros((seq_len, seq_len), dtype=np.int32)
        for i in range(seq_len):
            for j in range(seq_len):
                if i != j:
                    time_diff = abs(time_seq[i] - time_seq[j])
                    # Normalize and bin the time difference
                    time_matrix[i, j] = min(int(time_diff / 86400) + 1, 256)  # Convert to days, max 256 days
        
        user_time_diff_matrices[user_id] = time_matrix
    
    return user_sequences, user_time_diff_matrices,user_rating_sequences

def generate_training_data(user_sequences, user_time_matrices, user_rating_sequences,max_len, valid_ratio=0.1, test_ratio=0.2):
    """Generate train/validation/test data"""
    train_sequences = {}
    valid_sequences = {}
    test_sequences = {}
    
    train_times = {}
    valid_times = {}
    test_times = {}

    train_ratings = {}
    valid_ratings = {}
    test_ratings = {}
    
    for user_id, seq in tqdm(user_sequences.items(), desc="Splitting sequences"):
        seq_len = len(seq)
        
        # Skip sequences that are too short
        if seq_len < 3:
            continue
            
        rating_seq = user_rating_sequences.get(user_id, [])
        if len(rating_seq) != seq_len:
            continue  # 跳过rating长度不匹配的情况

        test_size = max(1, int(seq_len * test_ratio))
        valid_size = max(1, int(seq_len * valid_ratio))
        train_size = seq_len - test_size - valid_size
        
        # Must have at least 1 item in training sequence
        if train_size < 1:
            train_size = 1
            valid_size = 1 if seq_len > 2 else 0
            test_size = seq_len - train_size - valid_size
        
        # Split sequences
        train_seq = seq[:train_size]
        valid_seq = seq[train_size:train_size + valid_size]
        test_seq = seq[train_size + valid_size:]
        
        train_rating_seq = rating_seq[:train_size]
        valid_rating_seq = rating_seq[train_size:train_size + valid_size]
        test_rating_seq = rating_seq[train_size + valid_size:]
        # Get time matrices
        if user_id in user_time_matrices:
            time_matrix = user_time_matrices[user_id]
            
            train_time_matrix = time_matrix[:train_size, :train_size]
            
            # For validation, include training sequence but predict valid items
            valid_time_matrix = time_matrix[:train_size + valid_size, :train_size + valid_size]
            
            # For test, include training and validation but predict test items
            test_time_matrix = time_matrix
            
            train_times[user_id] = train_time_matrix
            valid_times[user_id] = valid_time_matrix
            test_times[user_id] = test_time_matrix
        
        # Store sequences
        train_sequences[user_id] = train_seq
        valid_sequences[user_id] = train_seq + valid_seq
        test_sequences[user_id] = train_seq + valid_seq + test_seq

        # Store ratings
        train_ratings[user_id] = train_rating_seq
        valid_ratings[user_id] = train_rating_seq + valid_rating_seq
        test_ratings[user_id] = train_rating_seq + valid_rating_seq + test_rating_seq
    
    return {
        'train': (train_sequences, train_times, train_ratings),
        'valid': (valid_sequences, valid_times, valid_ratings),
        'test': (test_sequences, test_times, test_ratings)
    }

def pad_sequences(sequences, time_matrices,rating_sequences, max_len):
    """Pad sequences and time matrices to max_len"""
    users = []
    padded_seqs = []
    padded_times = []
    padded_ratings = []
    labels = []
    
    for user_id, seq in sequences.items():
        if user_id not in time_matrices or user_id not in rating_sequences:
            continue
            
        time_matrix = time_matrices[user_id]
        rating_seq = rating_sequences[user_id]
        # Truncate sequences that are too long
        if len(seq) > max_len:
            seq = seq[-max_len:]
            time_matrix = time_matrix[-max_len:, -max_len:]
            rating_seq = rating_seq[-max_len:]
        # Pad sequences with 0
        padded_seq = seq + [0] * (max_len - len(seq))
        padded_rating = rating_seq + [0.0] * (max_len - len(rating_seq))
        users.append(user_id)
        padded_seqs.append(padded_seq)
        padded_ratings.append(padded_rating)
        # Pad time matrix
        padded_time = np.zeros((max_len, max_len), dtype=np.int32)
        seq_len = len(seq)
        padded_time[:seq_len, :seq_len] = time_matrix
        padded_times.append(padded_time)
        
        # Last item is label for evaluation
        labels.append(seq[-1])
    
    return np.array(users), np.array(padded_seqs), np.array(padded_times),np.array(padded_ratings) ,np.array(labels)

def prepare_global_graph_data(train_sequences):
    """Prepare data for building global interaction graph"""
    # The build_global_graph method expects data in the format of (user_id, item_sequence)
    graph_data = []
    
    for user_id, seq in train_sequences.items():
        # Filter out padding items
        filtered_seq = [item for item in seq if item != 0]
        if filtered_seq:  # Only add if there are valid items
            graph_data.append((user_id, filtered_seq))
    
    return graph_data

def evaluate_model(model, eval_data, user_num, item_num, args):
    """Evaluate model using NDCG@k, Recall@k and MRR"""
    model.eval()
    users, seqs, times, ratings, labels = eval_data
    
    # Pre-define constants to avoid repetitive calculations
    k_list = [5, 10, 20]
    total_samples = len(users)
    
    # Pre-allocate numpy arrays instead of appending to lists
    MRR_values = np.zeros(total_samples)
    NDCG_values = {k: np.zeros(total_samples) for k in k_list}
    Recall_values = {k: np.zeros(total_samples) for k in k_list}
    
    # Create item indices once - all possible items except 0 (which is likely padding)
    all_item_indices = np.arange(1, item_num + 1)
    
    # Process data in batches for better efficiency
    batch_size = 128  # Adjust based on available memory
    
    with torch.no_grad():
        for start_idx in tqdm(range(0, total_samples, batch_size), desc="Evaluating"):
            end_idx = min(start_idx + batch_size, total_samples)
            batch_size_actual = end_idx - start_idx
            
            # Prepare batch data
            batch_labels = labels[start_idx:end_idx]
            
            # Process each sample in the batch
            for i in range(batch_size_actual):
                idx = start_idx + i
                user = users[idx:idx+1]
                seq = seqs[idx:idx+1]
                time_matrix = times[idx:idx+1]
                rating = ratings[idx:idx+1]
                label = batch_labels[i]
                
                # Get candidate items more efficiently
                item_indices = all_item_indices.copy()
                label_idx = label - 1  # Convert to 0-indexed
                if 0 <= label_idx < len(item_indices):
                    item_indices = np.delete(item_indices, label_idx)
                
                # Sample negative items without using list operations
                neg_samples = np.random.choice(item_indices, 99, replace=False)
                sampled_indices = np.append(neg_samples, label)
                np.random.shuffle(sampled_indices)
                sampled_indices = sampled_indices.tolist()  # Convert back to list for model prediction
                
                # Get predictions
                predictions = model.predict(user, seq, time_matrix, sampled_indices, rating)
                predictions = predictions.squeeze()
                
                # Find the positive item's position in predictions
                pos_item_score = predictions[sampled_indices.index(label)]
                rank = torch.sum(predictions >= pos_item_score).item()
                
                # Calculate MRR
                MRR_values[idx] = 1.0 / rank if rank > 0 else 0
                
                # Get top-k predictions once
                _, topk_indices = torch.topk(predictions, k_list[-1])  # Get largest k needed
                topk_indices = topk_indices.tolist()
                
                pos_item_idx = sampled_indices.index(label)
                
                # Calculate metrics for each k
                for k in k_list:
                    topk = topk_indices[:k]
                    
                    # Recall@k
                    hit = 1 if pos_item_idx in topk else 0
                    Recall_values[k][idx] = hit
                    
                    # NDCG@k
                    if hit == 1 and rank <= k:
                        NDCG_values[k][idx] = 1.0 / np.log2(rank + 1)
    
    # Calculate average metrics efficiently
    avg_metrics = {'MRR': np.mean(MRR_values)}
    for k in k_list:
        avg_metrics[f'NDCG@{k}'] = np.mean(NDCG_values[k])
        avg_metrics[f'Recall@{k}'] = np.mean(Recall_values[k])
    
    return avg_metrics
# def evaluate_model(model, eval_data, user_num, item_num, args):
#     """Evaluate model using NDCG@k, Recall@k and MRR"""
#     model.eval()
#     users, seqs, times,ratings, labels = eval_data
    
#     k_list = [5, 10, 20]
#     NDCG = {k: [] for k in k_list}
#     Recall = {k: [] for k in k_list}
#     MRR = []
    
#     with torch.no_grad():
#         for i in tqdm(range(len(users)), desc="Evaluating"):
#             user = users[i:i+1]
#             seq = seqs[i:i+1]
#             time_matrix = times[i:i+1]
#             rating=ratings[i:i+1]
#             label = labels[i]
            
#             # Create candidate items - true item and 99 random negative items
#             item_indices = list(range(1, item_num+1))
#             if label in item_indices:  # It might not be due to data filtering
#                 item_indices.remove(label)
            
#             # Samples 99 negative items and include the positive item
#             sampled_indices = np.random.choice(item_indices, 99, replace=False).tolist() + [label]
#             random.shuffle(sampled_indices)
            
#             # Get predictions
#             predictions = model.predict(user, seq, time_matrix, sampled_indices,rating)
#             predictions = predictions.squeeze()
            
#             # Get ranking of the true item
#             rank = (predictions >= predictions[sampled_indices.index(label)]).sum().item()
            
#             # Calculate metrics
#             MRR.append(1.0 / rank if rank > 0 else 0)
            
#             for k in k_list:
#                 # Get top-k indices
#                 _, topk_indices = torch.topk(predictions, k)
#                 topk_indices = topk_indices.tolist()
                
#                 # Calculate Recall@k - if the item is in top-k
#                 hit = 1 if sampled_indices.index(label) in topk_indices else 0
#                 Recall[k].append(hit)
                
#                 # Calculate NDCG@k
#                 dcg = hit / np.log2(rank + 1) if rank <= k else 0
#                 NDCG[k].append(dcg)
    
#     # Calculate average metrics
#     avg_metrics = {
#         'MRR': np.mean(MRR),
#     }
    
#     for k in k_list:
#         avg_metrics[f'NDCG@{k}'] = np.mean(NDCG[k])
#         avg_metrics[f'Recall@{k}'] = np.mean(Recall[k])
    
#     return avg_metrics

def main():
    parser = argparse.ArgumentParser(description='Train TiSASGNN model on user data')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, required=True, help='Path to data file')
    parser.add_argument('--max_len', type=int, default=30, help='Maximum sequence length')
    
    # Model parameters
    parser.add_argument('--hidden_units', type=int, default=64, help='Hidden dimensionality')
    parser.add_argument('--num_blocks', type=int, default=2, help='Number of transformer blocks')
    parser.add_argument('--num_heads', type=int, default=2, help='Number of attention heads')
    parser.add_argument('--dropout_rate', type=float, default=0.25, help='Dropout rate')
    parser.add_argument('--l2_emb', type=float, default=0.0005, help='L2 regularization for embeddings')
    parser.add_argument('--time_span', type=int, default=365, help='Maximum time difference to consider')
    parser.add_argument('--gnn_type', type=str, default='sage', choices=['gcn', 'gat', 'sage'], help='GNN type')

    parser.add_argument('--walk_length', type=int, default=3, help='Length of random walks')
    parser.add_argument('--num_walks', type=int, default=10, help='Number of random walks from each node')
    parser.add_argument('--rw_p', type=float, default=1.0, help='Return parameter (control return probability)')
    parser.add_argument('--rw_q', type=float, default=1.0, help='In-out parameter (control search strategy)')
    parser.add_argument('--use_edge_weights', type=bool, default=True, help='Use edge weights in random walks')
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs')
    parser.add_argument('--early_stop', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    
    # Output parameters
    parser.add_argument('--save_dir', type=str, default='./models', help='Directory to save models')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create save directory if it doesn't exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Load and process data
    df, user_map, item_map = load_data(args.data_path)
    user_num = len(user_map)
    item_num = len(item_map)
    
    print(f"Processed data: {user_num} users, {item_num} items")
    
    # Create sequences and time matrices
    user_sequences, user_time_matrices,user_rating_sequences = create_sequences(df, max_len=args.max_len)
    
    # Generate train/valid/test data
    dataset = generate_training_data(user_sequences, user_time_matrices, user_rating_sequences,args.max_len)
    
    # Pad sequences for training
    train_data = pad_sequences(*dataset['train'], args.max_len)
    valid_data = pad_sequences(*dataset['valid'], args.max_len)
    test_data = pad_sequences(*dataset['test'], args.max_len)
    
    print(f"Train sequences: {len(train_data[0])}")
    print(f"Valid sequences: {len(valid_data[0])}")
    print(f"Test sequences: {len(test_data[0])}")
    #graph_data = prepare_global_graph_data(dataset['train'][0])
    
    # Create model
    model = TiSASGNN(user_num, item_num, args.time_span, args)
    model = model.to(args.device)
    
    # Build global graph before training
    #model.build_global_graph(graph_data)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    best_valid_metric = 0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # Train
        train_start_time = time.time()
        train_loss = train_tisasgnn(model, train_data, optimizer, args.batch_size, args)
        train_time = time.time() - train_start_time
        
        #refresh_global_graph_embeddings(model)
        print(f"Epoch {epoch+1}/{args.epochs} - Train loss: {train_loss:.4f} ({train_time:.2f}s)")

        # Evaluate on validation set
        valid_start_time = time.time()
        valid_metrics = evaluate_model(model, valid_data, user_num, item_num, args)
        valid_time = time.time() - valid_start_time
        print(f"Valid: MRR: {valid_metrics['MRR']:.4f}, NDCG@10: {valid_metrics['NDCG@10']:.4f}, Recall@10: {valid_metrics['Recall@10']:.4f} ({valid_time:.2f}s)")
        
        # Print metrics
        
        
        
        # Early stopping based on NDCG@10
        current_metric = valid_metrics['NDCG@10']
        if current_metric > best_valid_metric:
            best_valid_metric = current_metric
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pt'))
            print(f"New best model saved! NDCG@10: {best_valid_metric:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop:
                print(f"Early stopping at epoch {epoch+1}. Best epoch: {best_epoch+1} with NDCG@10: {best_valid_metric:.4f}")
                break
    
    # Load best model for testing
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pt')))
    
    # Evaluate on test set
    test_metrics = evaluate_model(model, test_data, user_num, item_num, args)
    
    print("\nTest Results:")
    print(f"MRR: {test_metrics['MRR']:.4f}")
    for k in [5, 10, 20]:
        print(f"NDCG@{k}: {test_metrics[f'NDCG@{k}']:.4f}, Recall@{k}: {test_metrics[f'Recall@{k}']:.4f}")
    
    # Save test results
    with open(os.path.join(args.save_dir, 'test_results.txt'), 'w') as f:
        f.write(f"Test Results:\n")
        f.write(f"MRR: {test_metrics['MRR']:.4f}\n")
        for k in [5, 10, 20]:
            f.write(f"NDCG@{k}: {test_metrics[f'NDCG@{k}']:.4f}, Recall@{k}: {test_metrics[f'Recall@{k}']:.4f}\n")

if __name__ == "__main__":
    main()