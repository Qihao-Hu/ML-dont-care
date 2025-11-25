#!/usr/bin/env python3
"""
Main training script for don't-care backpropagation.
Imports backprop_dc and model modules.
"""

import torch
import time
import sys
import numpy as np
from model import LevelizedModel
from backprop_dc import backprop_dc

# Try to import matplotlib, but make it optional
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Plotting will be disabled.")


def compare_predictions(expected, predicted):
    """
    Compare expected and predicted results and count bit differences.
    
    Args:
        expected: Tensor [N, num_bits] - expected output bits
        predicted: Tensor [N, num_bits] - predicted output bits
    
    Returns:
        dict: Statistics of bit differences
    """
    # Calculate bit differences per sample
    diff = (expected != predicted).int()  # [N, num_bits], 1 where different, 0 where same
    num_diff_bits = diff.sum(dim=1)  # [N], number of different bits per sample
    
    # Count samples by number of different bits
    max_bits = expected.shape[1]
    bit_diff_counts = {}
    
    for num_bits in range(max_bits + 1):
        count = (num_diff_bits == num_bits).sum().item()
        bit_diff_counts[num_bits] = count
    
    # Calculate accuracy
    total_samples = expected.shape[0]
    correct_samples = bit_diff_counts[0]
    accuracy = (correct_samples / total_samples * 100) if total_samples > 0 else 0
    
    return {
        'bit_diff_counts': bit_diff_counts,
        'total_samples': total_samples,
        'correct_samples': correct_samples,
        'accuracy': accuracy,
        'max_bits': max_bits
    }


def load_data(file_path):
    """Load input-output pairs from the data file"""
    inputs = []
    outputs = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Parse "Inputs=[...] Outputs=[...]"
                input_part = line.split('Outputs=')[0].strip()
                output_part = line.split('Outputs=')[1].split('#')[0].strip()
                
                # Extract the list parts
                input_str = input_part.replace('Inputs=', '').strip('[]')
                output_str = output_part.strip('[]')
                
                # Convert to lists of integers
                input_bits = [int(x) for x in input_str.split(',')]
                output_bits = [int(x) for x in output_str.split(',')]
                
                inputs.append(input_bits)
                outputs.append(output_bits)
    
    return torch.tensor(inputs, dtype=torch.float32), torch.tensor(outputs, dtype=torch.float32)


def write_timing_summary(f, timing_stats, num_steps):
    """Write timing statistics summary to file"""
    f.write(f"\n=== TIMING STATISTICS (over {num_steps} steps) ===\n")
    
    # Calculate statistics for each phase
    phases = [
        ('Forward Pass (model)', timing_stats['forward_times']),
        ('Backprop Loss Calc', timing_stats['backprop_times']),
        ('Backward Pass (autograd)', timing_stats['backward_times']),
        ('Optimizer Step', timing_stats['optimizer_times'])
    ]
    
    total_time = 0
    for phase_name, times in phases:
        times_ms = np.array(times) * 1000  # Convert to milliseconds
        mean_time = np.mean(times_ms)
        std_time = np.std(times_ms)
        min_time = np.min(times_ms)
        max_time = np.max(times_ms)
        total = np.sum(times_ms)
        total_time += total
        
        f.write(f"\n{phase_name}:\n")
        f.write(f"  Mean: {mean_time:.2f} ms/step\n")
        f.write(f"  Std:  {std_time:.2f} ms\n")
        f.write(f"  Min:  {min_time:.2f} ms\n")
        f.write(f"  Max:  {max_time:.2f} ms\n")
        f.write(f"  Total: {total:.2f} ms ({total/1000:.2f} sec)\n")
    
    f.write(f"\nTotal measured time: {total_time:.2f} ms ({total_time/1000:.2f} sec)\n")
    f.write(f"Average time per step: {total_time/num_steps:.2f} ms\n")
    
    # Percentage breakdown
    f.write(f"\nTime breakdown by percentage:\n")
    for phase_name, times in phases:
        total = np.sum(times) * 1000
        percentage = (total / total_time) * 100
        f.write(f"  {phase_name}: {percentage:.1f}%\n")


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) < 6 or len(sys.argv) > 8:
        print("Usage: python main.py <n_inputs> <n_hidden_layers> <hidden_layer_size> <train_data_file> <test_data_file> [learning_rate] [enable_plot]")
        print("Example: python main.py 6 10 20 dataset/3bit_Multiplier.txt dataset/3bit_Multiplier.txt")
        print("         python main.py 6 10 20 dataset/3bit_Multiplier.txt dataset/3bit_Multiplier.txt 0.05")
        print("         python main.py 6 10 20 dataset/3bit_Multiplier.txt dataset/3bit_Multiplier.txt 0.05 True")
        print("  This creates: n_inputs=6, layers_config=[20, 20, ..., 20, 6]")
        print("  Default learning rate: 0.08")
        print("  Default enable_plot: False")
        sys.exit(1)
    
    try:
        n_inputs = int(sys.argv[1])
        n_hidden_layers = int(sys.argv[2])
        hidden_layer_size = int(sys.argv[3])
        train_data_file = sys.argv[4]
        test_data_file = sys.argv[5]
        learning_rate = float(sys.argv[6]) if len(sys.argv) >= 7 else 0.08  # Default LR
        enable_plot = sys.argv[7].lower() in ['true', '1', 'yes'] if len(sys.argv) == 8 else False  # Default: no plot
        
        # Build layers_config: n_hidden_layers of hidden_layer_size, then n_inputs as output layer
        layers_config = [hidden_layer_size] * n_hidden_layers + [n_inputs]
        
        print(f"Network Configuration:")
        print(f"  n_inputs = {n_inputs}")
        print(f"  layers_config = {layers_config}")
        print(f"  learning_rate = {learning_rate}")
        print(f"  train_data_file = {train_data_file}")
        print(f"  test_data_file = {test_data_file}")
        print(f"  enable_plot = {enable_plot}")
        
    except ValueError:
        print("Error: n_inputs, n_hidden_layers, and hidden_layer_size must be integers")
        print("       learning_rate must be a float")
        sys.exit(1)
    
    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Force use CPU
    # device = torch.device("cpu")
    print(f"Using device: {device}")
    
    model = LevelizedModel(n_inputs=n_inputs, layers_config=layers_config)
    model = model.to(device)  # Move model to GPU
    
    # Use adaptive learning rate with better beta values
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0, 0.9), eps=1e-8, weight_decay=1e-5)
    
    # Load training data (from command-line argument)
    train_input_data, train_output_data = load_data(train_data_file)
    train_input_data = train_input_data.to(device)  # Move data to GPU
    train_output_data = train_output_data.to(device)
    
    # Load test data (from command-line argument)
    test_input_data, test_output_data = load_data(test_data_file)
    test_input_data = test_input_data.to(device)  # Move data to GPU
    test_output_data = test_output_data.to(device)
    
    print(f"Loaded {len(train_input_data)} training samples")
    print(f"Loaded {len(test_input_data)} test samples")
    
    # Learning rate scheduler for adaptive learning
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-6
    )
    
    # Patience mechanism for early stopping
    patience = 200  # Increased patience for better convergence
    patience_counter = 0
    best_loss = float('inf')
    best_loss_accuracy = 0  # track highest accuracy achieved at the best loss
    best_model_state = None
    
    # Track best accuracy independently
    best_accuracy = 0
    best_accuracy_step = 0
    best_accuracy_loss = float('inf')
    best_accuracy_bit_dist = None  # Distribution of bit errors at best accuracy
    best_accuracy_model_state = None
    
    # Record start time
    start_time = time.time()
    
    # Timing statistics
    timing_stats = {
        'forward_times': [],
        'backprop_times': [],
        'backward_times': [],
        'optimizer_times': []
    }
    
    # Accuracy tracking for plotting
    accuracy_history = []  # List of (step, correct_count, total_count)
    
    # Open results file in append mode
    with open("results.txt", "a") as f:
        f.write(f"\n=== Training Session Started ===\n")
        f.write(f"Device: {device}\n")
        f.write(f"Network Architecture: {layers_config}\n")
        f.write(f"Number of Inputs: {n_inputs}\n")
        f.write(f"Learning Rate: {learning_rate} (initial)\n")
        f.write(f"Training samples: {len(train_input_data)}\n")
        f.write(f"Patience: {patience}\n")
        f.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
        
        early_stop = False
        final_step = 0
        stop_reason = ""
        
        loss_tol = 1e-12  # tolerance when comparing floating losses
        for step in range(10000):
            # === TIME PROFILING: Forward Pass ===
            forward_start = time.time()
            node_ab_values = model(train_input_data)
            forward_time = time.time() - forward_start

            # === TIME PROFILING: Backprop Loss Calculation ===
            backprop_start = time.time()
            loss = backprop_dc(model, node_ab_values, train_output_data, profile_time=False)
            backprop_time = time.time() - backprop_start

            # === TIME PROFILING: Backward Pass ===
            backward_start = time.time()
            optimizer.zero_grad()
            loss.backward()   # Automatically compute gradients for param_w0, param_w1
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            backward_time = time.time() - backward_start
            
            # === TIME PROFILING: Optimizer Step ===
            optimizer_start = time.time()
            optimizer.step()
            
            # Update learning rate based on loss
            scheduler.step(loss)
            
            optimizer_time = time.time() - optimizer_start
            
            # Store timing statistics
            timing_stats['forward_times'].append(forward_time)
            timing_stats['backprop_times'].append(backprop_time)
            timing_stats['backward_times'].append(backward_time)
            timing_stats['optimizer_times'].append(optimizer_time)

            current_loss = loss.item()
            
            # Calculate accuracy on TEST data (not training data)
            with torch.no_grad():
                # Evaluate on test set
                test_node_ab_values = model(test_input_data)
                test_outputs = []
                for q in model.layers[-1]:
                    test_outputs.append(test_node_ab_values[q]["out"])
                test_outputs = torch.stack(test_outputs, dim=1)
                test_pred_bits = (test_outputs > 0).int()
                correct_count = (test_pred_bits == test_output_data.int()).all(dim=1).sum().item()
                total_count = len(test_input_data)
            
            # Record accuracy for plotting
            accuracy_history.append((step, correct_count, total_count))
            
            # Track best accuracy independently with bit error distribution
            update_best = False
            if correct_count > best_accuracy:
                # New best accuracy
                update_best = True
            elif correct_count == best_accuracy and correct_count < total_count:
                # Same accuracy, compare bit error distribution
                # Calculate bit error distribution for incorrect samples
                incorrect_mask = ~(test_pred_bits == test_output_data.int()).all(dim=1)
                if incorrect_mask.sum() > 0:
                    incorrect_preds = test_pred_bits[incorrect_mask]
                    incorrect_expected = test_output_data.int()[incorrect_mask]
                    bit_errors = (incorrect_preds != incorrect_expected).sum(dim=1)  # errors per sample
                    
                    # Count distribution: how many samples with 1-bit error, 2-bit error, etc.
                    current_bit_dist = []
                    max_possible_bits = test_output_data.shape[1]
                    for i in range(1, max_possible_bits + 1):
                        count_i = (bit_errors == i).sum().item()
                        current_bit_dist.append(count_i)
                    
                    # Compare with best_accuracy_bit_dist (prefer fewer low-bit errors)
                    if best_accuracy_bit_dist is None:
                        update_best = True
                    else:
                        # Lexicographic comparison: prefer fewer 1-bit errors, then fewer 2-bit errors, etc.
                        for i in range(len(current_bit_dist)):
                            if current_bit_dist[i] < best_accuracy_bit_dist[i]:
                                update_best = True
                                break
                            elif current_bit_dist[i] > best_accuracy_bit_dist[i]:
                                break
            
            if update_best:
                best_accuracy = correct_count
                best_accuracy_step = step
                best_accuracy_loss = current_loss
                
                # Calculate and save bit error distribution
                if correct_count < total_count:
                    incorrect_mask = ~(test_pred_bits == test_output_data.int()).all(dim=1)
                    incorrect_preds = test_pred_bits[incorrect_mask]
                    incorrect_expected = test_output_data.int()[incorrect_mask]
                    bit_errors = (incorrect_preds != incorrect_expected).sum(dim=1)
                    best_accuracy_bit_dist = []
                    max_possible_bits = test_output_data.shape[1]
                    for i in range(1, max_possible_bits + 1):
                        count_i = (bit_errors == i).sum().item()
                        best_accuracy_bit_dist.append(count_i)
                else:
                    best_accuracy_bit_dist = []  # Perfect accuracy, no errors
                
                # Save model state at best accuracy
                best_accuracy_model_state = {
                    'param_w0': [p.detach().clone() for p in model.param_w0],
                    'param_w1': [p.detach().clone() for p in model.param_w1],
                    'step': step,
                    'accuracy': correct_count,
                    'loss': current_loss,
                    'bit_dist': best_accuracy_bit_dist.copy() if best_accuracy_bit_dist else []
                }
            
            # Check if 100% accuracy reached for the first time
            if correct_count == total_count:
                final_step = step
                early_stop = True
                stop_reason = "perfect_accuracy"
                end_time = time.time()
                training_time = end_time - start_time
                
                # Best accuracy model is already saved above
                best_loss = current_loss
                best_loss_accuracy = correct_count
                best_model_state = {
                    'param_w0': [p.detach().clone() for p in model.param_w0],
                    'param_w1': [p.detach().clone() for p in model.param_w1],
                    'step': step,
                    'accuracy': correct_count
                }
                
                current_lr = optimizer.param_groups[0]['lr']
                f.write(f"Step {step:04d} | Loss = {current_loss:.6f} | LR = {current_lr:.6f} | Correct: {correct_count}/{total_count} | Perfect accuracy!\n")
                f.write(f"\n=== Early Stopping: Perfect Accuracy Reached ===\n")
                f.write(f"Final Step: {step:04d}\n")
                f.write(f"Final Loss: {current_loss:.6f}\n")
                f.write(f"Accuracy: 100% ({correct_count}/{total_count})\n")
                f.write(f"Training Time: {training_time:.2f} seconds\n")
                f.write(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
                
                # Write timing statistics
                write_timing_summary(f, timing_stats, step + 1)
                
                f.flush()
                break
            
            # Update best model if current loss improves or matches with better accuracy
            loss_improved = current_loss + loss_tol < best_loss
            loss_tied_better_acc = abs(current_loss - best_loss) <= loss_tol and correct_count > best_loss_accuracy

            if loss_improved or loss_tied_better_acc:
                if loss_improved:
                    best_loss = current_loss
                best_loss_accuracy = correct_count
                # Save the best model state (deep copy)
                best_model_state = {
                    'param_w0': [p.detach().clone() for p in model.param_w0],
                    'param_w1': [p.detach().clone() for p in model.param_w1],
                    'step': step,
                    'accuracy': correct_count
                }
                patience_counter = 0  # Reset patience counter
                current_lr = optimizer.param_groups[0]['lr']
                note = "New best loss" if loss_improved else "Best loss tie, better accuracy"
                f.write(f"Step {step:04d} | Loss = {current_loss:.6f} | LR = {current_lr:.6f} | Correct: {correct_count}/{total_count} | {note}! (patience reset)\n")
                f.flush()
            else:
                # Loss did not improve
                patience_counter += 1
                current_lr = optimizer.param_groups[0]['lr']
                if step % 10 == 0:  # Write every 10 steps to reduce file size
                    f.write(f"Step {step:04d} | Loss = {current_loss:.6f} | LR = {current_lr:.6f} | Correct: {correct_count}/{total_count} | Patience: {patience_counter}/{patience}\n")
                    f.flush()
            
            # Check patience early stopping
            if patience_counter >= patience:
                final_step = step
                early_stop = True
                stop_reason = "patience"
                end_time = time.time()
                training_time = end_time - start_time
                
                f.write(f"\n=== Early Stopping: Patience Exceeded ===\n")
                f.write(f"Final Step: {step:04d}\n")
                f.write(f"Current Loss: {current_loss:.6f}\n")
                f.write(f"Best Loss: {best_loss:.6f} (at step {best_model_state['step']})\n")
                f.write(f"Best Accuracy at Best Loss: {best_loss_accuracy}/{total_count} ({best_loss_accuracy/total_count*100:.2f}%)\n")
                f.write(f"Maximum Accuracy Achieved: {best_accuracy}/{total_count} ({best_accuracy/total_count*100:.2f}%) at step {best_accuracy_step} (loss={best_accuracy_loss:.6f})\n")
                f.write(f"Training Time: {training_time:.2f} seconds\n")
                f.write(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
                
                # Write timing statistics
                write_timing_summary(f, timing_stats, step + 1)
                
                f.flush()
                break
            
            # Check early stopping condition
            if loss.item() < 0.01:
                final_step = step
                early_stop = True
                stop_reason = "loss_threshold"
                end_time = time.time()
                training_time = end_time - start_time
                
                f.write(f"\n=== Early Stopping: Loss Threshold Reached ===\n")
                f.write(f"Final Step: {step:04d}\n")
                f.write(f"Final Loss: {loss.item():.6f}\n")
                f.write(f"Best Loss: {best_loss:.6f} (at step {best_model_state['step']})\n")
                f.write(f"Best Accuracy at Best Loss: {best_loss_accuracy}/{total_count} ({best_loss_accuracy/total_count*100:.2f}%)\n")
                f.write(f"Maximum Accuracy Achieved: {best_accuracy}/{total_count} ({best_accuracy/total_count*100:.2f}%) at step {best_accuracy_step} (loss={best_accuracy_loss:.6f})\n")
                f.write(f"Training Time: {training_time:.2f} seconds\n")
                f.write(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
                
                # Write timing statistics
                write_timing_summary(f, timing_stats, step + 1)
                
                f.flush()
                break
        
        if not early_stop:
            end_time = time.time()
            training_time = end_time - start_time
            final_step = 9999
            stop_reason = "max_steps"
            f.write(f"\n=== Training Completed (Max Steps Reached) ===\n")
            f.write(f"Final Step: {final_step:04d}\n")
            f.write(f"Best Loss: {best_loss:.6f} (at step {best_model_state['step']})\n")
            f.write(f"Best Accuracy at Best Loss: {best_loss_accuracy}/{total_count} ({best_loss_accuracy/total_count*100:.2f}%)\n")
            f.write(f"Maximum Accuracy Achieved: {best_accuracy}/{total_count} ({best_accuracy/total_count*100:.2f}%) at step {best_accuracy_step} (loss={best_accuracy_loss:.6f})\n")
            f.write(f"Training Time: {training_time:.2f} seconds\n")
            f.write(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
            
            # Write timing statistics
            write_timing_summary(f, timing_stats, final_step + 1)
            
            f.flush()
    
    # Restore best accuracy model parameters
    if best_accuracy_model_state is not None:
        print(f"\nRestoring best accuracy model from step {best_accuracy_model_state['step']}")
        print(f"  Accuracy: {best_accuracy_model_state['accuracy']}/{total_count} ({best_accuracy_model_state['accuracy']/total_count*100:.2f}%)")
        print(f"  Loss: {best_accuracy_model_state['loss']:.6f}")
        if best_accuracy_model_state.get('bit_dist'):
            print(f"  Bit error distribution (for incorrect samples):")
            for i, count in enumerate(best_accuracy_model_state['bit_dist'], 1):
                if count > 0:
                    print(f"    {i}-bit errors: {count} samples")
        for i, p in enumerate(model.param_w0):
            p.data.copy_(best_accuracy_model_state['param_w0'][i])
        for i, p in enumerate(model.param_w1):
            p.data.copy_(best_accuracy_model_state['param_w1'][i])
        
        # Save best parameters to file
        with open("best_parameters.txt", "w") as f:
            f.write(f"=== Best Accuracy Model Parameters ===\n")
            f.write(f"Network Architecture: {layers_config}\n")
            f.write(f"Number of Inputs: {n_inputs}\n")
            f.write(f"Best Accuracy: {best_accuracy_model_state['accuracy']}/{total_count} ({best_accuracy_model_state['accuracy']/total_count*100:.2f}%)\n")
            f.write(f"Best Accuracy Step: {best_accuracy_model_state['step']}\n")
            f.write(f"Loss at Best Accuracy: {best_accuracy_model_state['loss']:.6f}\n")
            if best_accuracy_model_state.get('bit_dist'):
                f.write(f"Bit error distribution (for incorrect samples):\n")
                for i, count in enumerate(best_accuracy_model_state['bit_dist'], 1):
                    if count > 0:
                        f.write(f"  {i}-bit errors: {count} samples\n")
            f.write(f"Best Loss: {best_loss:.6f}\n")
            f.write(f"Stop Reason: {stop_reason}\n\n")
            
            f.write("=== param_w0 (weights for input 0) ===\n")
            for i, w0 in enumerate(best_accuracy_model_state['param_w0']):
                f.write(f"Layer {i} w0:\n")
                f.write(f"{w0.cpu().numpy()}\n\n")
            
            f.write("=== param_w1 (weights for input 1) ===\n")
            for i, w1 in enumerate(best_accuracy_model_state['param_w1']):
                f.write(f"Layer {i} w1:\n")
                f.write(f"{w1.cpu().numpy()}\n\n")
        
        print("Best accuracy model parameters saved to best_parameters.txt")
    
    if stop_reason == "patience":
        print(f"Training stopped due to patience ({patience}) at step {final_step}")
    elif stop_reason == "loss_threshold":
        print(f"Training stopped early at step {final_step} (loss < 0.01)")
    else:
        print("Training completed (reached maximum steps).")

    model.eval()
    with torch.no_grad():
        # Test on test data
        test_node_ab_values = model(test_input_data)

        # Get the outputs of the model's last layer
        test_outputs = []
        for q in model.layers[-1]:
            test_outputs.append(test_node_ab_values[q]["out"])  # Value of each output node
        test_outputs = torch.stack(test_outputs, dim=1)  # [batch_size, num_outputs]

        # Convert continuous values to 0/1 (binarize)
        test_pred_bits = (test_outputs > 0).int()
        
        # Compare predictions with expected results
        comparison_stats = compare_predictions(test_output_data.int(), test_pred_bits)
        
        # Write final test results to file
        with open("results.txt", "a") as f:
            f.write("=== Final Test Results ===\n")
            
            # Write comparison statistics
            f.write(f"\n--- Prediction Accuracy Summary ---\n")
            f.write(f"Total samples: {comparison_stats['total_samples']}\n")
            f.write(f"Correct predictions: {comparison_stats['correct_samples']}\n")
            f.write(f"Accuracy: {comparison_stats['accuracy']:.2f}%\n\n")
            
            f.write(f"--- Bit Difference Distribution ---\n")
            for num_diff in range(comparison_stats['max_bits'] + 1):
                count = comparison_stats['bit_diff_counts'][num_diff]
                percentage = (count / comparison_stats['total_samples'] * 100) if comparison_stats['total_samples'] > 0 else 0
                if num_diff == 0:
                    f.write(f"Exactly correct (0 bits different): {count} samples ({percentage:.2f}%)\n")
                elif num_diff == 1:
                    f.write(f"1 bit different: {count} samples ({percentage:.2f}%)\n")
                else:
                    f.write(f"{num_diff} bits different: {count} samples ({percentage:.2f}%)\n")
            
            f.write(f"\n--- Detailed Results ---\n")
            for i in range(len(test_input_data)):
                f.write(f"Input: {test_input_data[i].int().tolist()}, "
                       f"Expected: {test_output_data[i].int().tolist()}, "
                       f"Predicted: {test_pred_bits[i].tolist()}\n")
            f.write("=== Training Session Completed ===\n\n")
        
        # Print summary to console
        print("\n=== Test Results Summary ===")
        print(f"Total samples: {comparison_stats['total_samples']}")
        print(f"Correct predictions: {comparison_stats['correct_samples']}")
        print(f"Accuracy: {comparison_stats['accuracy']:.2f}%")
        print("\nBit Difference Distribution:")
        for num_diff in range(comparison_stats['max_bits'] + 1):
            count = comparison_stats['bit_diff_counts'][num_diff]
            percentage = (count / comparison_stats['total_samples'] * 100) if comparison_stats['total_samples'] > 0 else 0
            if num_diff == 0:
                print(f"  Exactly correct (0 bits different): {count} samples ({percentage:.2f}%)")
            elif num_diff == 1:
                print(f"  1 bit different: {count} samples ({percentage:.2f}%)")
            else:
                print(f"  {num_diff} bits different: {count} samples ({percentage:.2f}%)")
    
    # Generate accuracy plot if enabled
    if enable_plot and len(accuracy_history) > 0:
        if not MATPLOTLIB_AVAILABLE:
            print("\nWarning: Plotting requested but matplotlib is not installed.")
            print("To enable plotting, install matplotlib: pip install matplotlib")
        else:
            steps = [item[0] for item in accuracy_history]
            correctness = [item[1] for item in accuracy_history]
            total_samples = accuracy_history[0][2]
            correctness_percent = [(c / total_samples * 100) for c in correctness]
            
            plt.figure(figsize=(10, 6))
            plt.plot(steps, correctness_percent, linewidth=2, color='blue')
            plt.xlabel('Step', fontsize=12)
            plt.ylabel('Accuracy (%)', fontsize=12)
            plt.title(f'Training Accuracy Progress (Total Samples: {total_samples})', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.axhline(y=100, color='green', linestyle='--', linewidth=1, label='Perfect Accuracy (100%)')
            plt.ylim(0, 105)
            plt.legend()
            plt.tight_layout()
            
            # Save plot
            plot_filename = 'accuracy_plot.png'
            plt.savefig(plot_filename, dpi=150)
            print(f"\nAccuracy plot saved to {plot_filename}")
            plt.close()
    
    print("Test completed.")
