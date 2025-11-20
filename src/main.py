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
    if len(sys.argv) < 6 or len(sys.argv) > 7:
        print("Usage: python main.py <n_inputs> <n_hidden_layers> <hidden_layer_size> <train_data_file> <test_data_file> [learning_rate]")
        print("Example: python main.py 6 10 20 dataset/3bit_Multiplier.txt dataset/3bit_Multiplier.txt")
        print("         python main.py 6 10 20 dataset/3bit_Multiplier.txt dataset/3bit_Multiplier.txt 0.05")
        print("  This creates: n_inputs=6, layers_config=[20, 20, ..., 20, 6]")
        print("  Default learning rate: 0.08")
        sys.exit(1)
    
    try:
        n_inputs = int(sys.argv[1])
        n_hidden_layers = int(sys.argv[2])
        hidden_layer_size = int(sys.argv[3])
        train_data_file = sys.argv[4]
        test_data_file = sys.argv[5]
        learning_rate = float(sys.argv[6]) if len(sys.argv) == 7 else 0.08  # Default LR
        
        # Build layers_config: n_hidden_layers of hidden_layer_size, then n_inputs as output layer
        layers_config = [hidden_layer_size] * n_hidden_layers + [n_inputs]
        
        print(f"Network Configuration:")
        print(f"  n_inputs = {n_inputs}")
        print(f"  layers_config = {layers_config}")
        print(f"  learning_rate = {learning_rate}")
        print(f"  train_data_file = {train_data_file}")
        print(f"  test_data_file = {test_data_file}")
        
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
    patience = 100  # Increased patience for better convergence
    patience_counter = 0
    best_loss = float('inf')
    best_model_state = None
    
    # Record start time
    start_time = time.time()
    
    # Timing statistics
    timing_stats = {
        'forward_times': [],
        'backprop_times': [],
        'backward_times': [],
        'optimizer_times': []
    }
    
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
        
        for step in range(10000):
            # === TIME PROFILING: Forward Pass ===
            forward_start = time.time()
            node_ab_values = model(train_input_data)
            forward_time = time.time() - forward_start

            # === TIME PROFILING: Backprop Loss Calculation ===
            backprop_start = time.time()
            profile_this_step = (step % 100 == 0)  # Profile every 100 steps
            if profile_this_step:
                loss, backprop_profile = backprop_dc(model, node_ab_values, train_output_data, profile_time=True)
            else:
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
            
            # Update best model if current loss is better
            if current_loss < best_loss:
                best_loss = current_loss
                # Save the best model state (deep copy)
                best_model_state = {
                    'param_w0': [p.detach().clone() for p in model.param_w0],
                    'param_w1': [p.detach().clone() for p in model.param_w1],
                    'step': step
                }
                patience_counter = 0  # Reset patience counter
                current_lr = optimizer.param_groups[0]['lr']
                f.write(f"Step {step:04d} | Loss = {current_loss:.6f} | LR = {current_lr:.6f} | New best! (patience reset)\n")
                
                # Write timing information for this step
                if profile_this_step:
                    f.write(f"  Timing: Forward={forward_time*1000:.2f}ms, Backprop={backprop_time*1000:.2f}ms, "
                           f"Backward={backward_time*1000:.2f}ms, Optimizer={optimizer_time*1000:.2f}ms\n")
                    f.write(f"  Backprop breakdown: Init={backprop_profile['initialization']*1000:.2f}ms, "
                           f"OutputLoss={backprop_profile['output_layer_loss']*1000:.2f}ms, "
                           f"Loop1(m_pq)={backprop_profile['loop1_m_pq_calculation']*1000:.2f}ms, "
                           f"Loop2(g_p)={backprop_profile['loop2_g_p_calculation']*1000:.2f}ms, "
                           f"Loop3(f_p)={backprop_profile['loop3_f_p_calculation']*1000:.2f}ms, "
                           f"Loop4(m_rp)={backprop_profile['loop4_m_rp_calculation']*1000:.2f}ms, "
                           f"LossAccum={backprop_profile['loss_accumulation']*1000:.2f}ms\n")
                
                f.flush()
            else:
                # Loss did not improve
                patience_counter += 1
                current_lr = optimizer.param_groups[0]['lr']
                if step % 10 == 0:  # Write every 10 steps to reduce file size
                    f.write(f"Step {step:04d} | Loss = {current_loss:.6f} | LR = {current_lr:.6f} | Patience: {patience_counter}/{patience}\n")
                    
                    # Write timing information every 100 steps
                    if profile_this_step:
                        f.write(f"  Timing: Forward={forward_time*1000:.2f}ms, Backprop={backprop_time*1000:.2f}ms, "
                               f"Backward={backward_time*1000:.2f}ms, Optimizer={optimizer_time*1000:.2f}ms\n")
                        f.write(f"  Backprop breakdown: Init={backprop_profile['initialization']*1000:.2f}ms, "
                               f"OutputLoss={backprop_profile['output_layer_loss']*1000:.2f}ms, "
                               f"Loop1(m_pq)={backprop_profile['loop1_m_pq_calculation']*1000:.2f}ms, "
                               f"Loop2(g_p)={backprop_profile['loop2_g_p_calculation']*1000:.2f}ms, "
                               f"Loop3(f_p)={backprop_profile['loop3_f_p_calculation']*1000:.2f}ms, "
                               f"Loop4(m_rp)={backprop_profile['loop4_m_rp_calculation']*1000:.2f}ms, "
                               f"LossAccum={backprop_profile['loss_accumulation']*1000:.2f}ms\n")
                    
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
            f.write(f"Training Time: {training_time:.2f} seconds\n")
            f.write(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
            
            # Write timing statistics
            write_timing_summary(f, timing_stats, final_step + 1)
            
            f.flush()
    
    # Restore best model parameters
    if best_model_state is not None:
        print(f"\nRestoring best model from step {best_model_state['step']} with loss {best_loss:.6f}")
        for i, p in enumerate(model.param_w0):
            p.data.copy_(best_model_state['param_w0'][i])
        for i, p in enumerate(model.param_w1):
            p.data.copy_(best_model_state['param_w1'][i])
        
        # Save best parameters to file
        with open("best_parameters.txt", "w") as f:
            f.write(f"=== Best Model Parameters ===\n")
            f.write(f"Network Architecture: {layers_config}\n")
            f.write(f"Number of Inputs: {n_inputs}\n")
            f.write(f"Best Loss: {best_loss:.6f}\n")
            f.write(f"Best Step: {best_model_state['step']}\n")
            f.write(f"Stop Reason: {stop_reason}\n\n")
            
            f.write("=== param_w0 (weights for input 0) ===\n")
            for i, w0 in enumerate(best_model_state['param_w0']):
                f.write(f"Layer {i} w0:\n")
                f.write(f"{w0.cpu().numpy()}\n\n")
            
            f.write("=== param_w1 (weights for input 1) ===\n")
            for i, w1 in enumerate(best_model_state['param_w1']):
                f.write(f"Layer {i} w1:\n")
                f.write(f"{w1.cpu().numpy()}\n\n")
        
        print("Best parameters saved to best_parameters.txt")
    
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
    
    print("Test completed.")
