import torch
import torch.nn.functional as F
import time
import sys
from model import LevelizedModel

def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

def count_nodes_in_layers_before(model, node_p):
    node_layer = None
    if node_p < model.n_inputs:
        node_layer = 0  
    else:
        for layer_idx, layer_nodes in enumerate(model.layers):
            if node_p in layer_nodes:
                node_layer = layer_idx + 1 
                break
    if node_layer is None:
        raise ValueError(f" Node {node_p} does not exist in the model.")
    total_nodes = 0
    if node_layer > 0:
        total_nodes += model.n_inputs  
    if node_layer > 1:
        for layer_idx in range(node_layer - 1): 
            total_nodes += len(model.layers[layer_idx])
    return total_nodes

def backprop_dc(model, node_ab_values, output_bits, eps=1e-8):
    """
    Compute loss using don't-care backprop logic (batch-aware, autograd-friendly).
    - model: LevelizedModel
    - node_ab_values: dict mapping node_id -> {'ap0': Tensor[B], 'ap1': Tensor[B],
                                              'bp0': Tensor[B], 'bp1': Tensor[B], 'out': Tensor[B]}
    - output_bits: Tensor [B, num_outputs] with {0,1}
    Returns scalar loss (tensor) suitable for loss.backward()
    """
    device = next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device("cpu")
    B = output_bits.size(0)
    n_inputs = model.n_inputs

    # find max node id
    max_node_id = n_inputs - 1
    for layer in model.layers:
        max_node_id = max(max_node_id, max(layer))

    # f: dict node -> Tensor shape [B, 2]
    # g: dict node -> Tensor shape [B]
    # t: dict node -> scalar Tensor (no batch)
    # m: dict (node_r, node_p, i, j) -> Tensor (for storing m[r,p,i,j])
    f = {}
    g = {}
    t = {}
    m = {}

    # initialize containers
    for nid in range(max_node_id + 1):
        # f default zeros [B,2]
        f[nid] = torch.zeros(B, 2, device=device)
        g[nid] = torch.zeros(B, device=device)
        t[nid] = torch.tensor(0.0, device=device)

    # --- initialize f for output layer using provided output_bits ---
    # output_bits: [B, num_outputs]
    if output_bits.dim() != 2:
        raise ValueError("output_bits should be [B, num_outputs]")
    output_layer = model.layers[-1]
    num_outputs = len(output_layer)
    assert output_bits.size(1) == num_outputs or output_bits.size(1) == len(model.output_nodes), \
        "output_bits width mismatch with model's output layer"

    # Map {0,1} -> {-1,1}
    transformed = (output_bits[:, :num_outputs].float() * 2.0 - 1.0)  # [B, num_outputs]
    # assign into f for each q in output_layer
    for idx, q in enumerate(output_layer):
        val = transformed[:, idx]               # [B]
        f[q] = torch.stack([val, torch.zeros_like(val)], dim=1)   # [B,2]

    loss = torch.tensor(0.0, device=device, requires_grad=True)

    # traverse layers backwards (excluding the output layer which is set)
    for layer_idx in range(len(model.layers) - 2, -1, -1):
        layer_nodes = model.layers[layer_idx]
        
        for p in layer_nodes:
            t_p = torch.tensor(0.0, device=device)
            
            # Step 1: Calculate m[p,q] and accumulate t[p] for all next layers
            # Iterate through all layers after current layer
            for next_layer_idx in range(layer_idx + 1, len(model.layers)):
                for q in model.layers[next_layer_idx]:
                    # Check if q is in the output layer
                    is_output_layer = (next_layer_idx == len(model.layers) - 1)

                    if is_output_layer:
                        # q is in output layer: m[p,q] comes directly from q's w0
                        # Use softmax(w0) as s[q,0,:]
                        row_idx = q - n_inputs
                        w0_row = model.param_w0[row_idx]
                        s0_row = F.softmax(w0_row, dim=0)  # Apply softmax
                        
                        idx0 = p
                        idx1 = p + count_nodes_in_layers_before(model, q)
                        
                        m[(p, q, 0, 0)] = s0_row[idx0]
                        m[(p, q, 0, 1)] = s0_row[idx1]

                        # Accumulate t[p] (only w0 terms for output layer)
                        t_p = t_p + m[(p, q, 0, 0)] + m[(p, q, 0, 1)]
                        
                    else:
                        # Accumulate t[p] (both w0 and w1 terms for non-output layers)
                        t_p = t_p + m[(p, q, 0, 0)] + m[(p, q, 0, 1)] + m[(p, q, 1, 0)] + m[(p, q, 1, 1)]
            
            # Step 2: Now that t[p] is fully calculated, compute g[p]
            t[p] = t_p
            t_p_safe = t_p if (t_p.abs() > eps).item() else (t_p + eps)
            
            g_p = torch.zeros(B, device=device)
            for next_layer_idx in range(layer_idx + 1, len(model.layers)):
                for q in model.layers[next_layer_idx]:
                    is_output_layer = (next_layer_idx == len(model.layers) - 1)
                    
                    fq0 = f[q][:, 0]
                    fq1 = f[q][:, 1]

                    if is_output_layer:
                        # Output layer: only use w0 terms
                        g_p = g_p + (m[(p, q, 0, 0)] / t_p_safe) * fq0 + (m[(p, q, 0, 1)] / t_p_safe) * (-fq0)
                    else:
                        # Non-output layer: use both w0 and w1 terms
                        g_p = g_p + (m[(p, q, 0, 0)] / t_p_safe) * fq0 + (m[(p, q, 1, 0)] / t_p_safe) * fq1 \
                                   + (m[(p, q, 0, 1)] / t_p_safe) * (-fq0) + (m[(p, q, 1, 1)] / t_p_safe) * (-fq1)
            
            g[p] = g_p
            
            # Set f[p] according to rules
            ap0 = node_ab_values[p]['ap0'].to(device)
            ap1 = node_ab_values[p]['ap1'].to(device)
            
            mask0 = (g_p < 0) | (ap0 == 1)
            mask1 = (ap1 == -1)
            maskA = mask0 & mask1
            
            mask2 = (g_p > 0) | (ap1 == 1)
            mask3 = (ap0 == -1)
            maskB = mask2 & mask3
            
            f_p0 = g_p.clone()
            f_p1 = g_p.clone()
            f_p0 = torch.where(maskA, torch.zeros_like(f_p0), f_p0)
            f_p1 = torch.where(maskB, torch.zeros_like(f_p1), f_p1)
            
            f[p] = torch.stack([f_p0, f_p1], dim=1)
            
            # Calculate m[r,p] for all previous layers
            # m[r,p,i,j] = t[p] * s[p,i,j] where s[p,i,j] is softmax of p's weight
            for prev_layer_idx in range(layer_idx):
                for r in model.layers[prev_layer_idx]:
                    row_idx_p = p - n_inputs
                    w0_row_p = model.param_w0[row_idx_p]
                    w1_row_p = model.param_w1[row_idx_p]
                    
                    # Apply softmax to get s[p,0,:] and s[p,1,:]
                    s0_row_p = F.softmax(w0_row_p, dim=0)
                    s1_row_p = F.softmax(w1_row_p, dim=0)
                    
                    r_idx0 = r
                    r_idx1 = r + count_nodes_in_layers_before(model, p)
                    
                    m[(r, p, 0, 0)] = t_p * s0_row_p[r_idx0]
                    m[(r, p, 0, 1)] = t_p * s0_row_p[r_idx1]
                    m[(r, p, 1, 0)] = t_p * s1_row_p[r_idx0]
                    m[(r, p, 1, 1)] = t_p * s1_row_p[r_idx1]

            
            # Add to loss
            bp0 = node_ab_values[p]['bp0'].to(device)
            bp1 = node_ab_values[p]['bp1'].to(device)
            sign_f0 = torch.sign(f_p0)
            sign_f1 = torch.sign(f_p1)
            term0 = torch.abs(f_p0) * ((bp0 - sign_f0) ** 2)
            term1 = torch.abs(f_p1) * ((bp1 - sign_f1) ** 2)
            loss = loss + term0.sum() + term1.sum()

    # final: average over batch size
    loss = loss / float(B)
    return loss

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

if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) != 4:
        print("Usage: python backprop_dc.py <n_inputs> <n_hidden_layers> <hidden_layer_size>")
        print("Example: python backprop_dc.py 8 4 10")
        print("  This creates: n_inputs=8, layers_config=[10, 10, 10, 10, 8]")
        sys.exit(1)
    
    try:
        n_inputs = int(sys.argv[1])
        n_hidden_layers = int(sys.argv[2])
        hidden_layer_size = int(sys.argv[3])
        
        # Build layers_config: n_hidden_layers of hidden_layer_size, then n_inputs as output layer
        layers_config = [hidden_layer_size] * n_hidden_layers + [n_inputs]
        
        print(f"Network Configuration:")
        print(f"  n_inputs = {n_inputs}")
        print(f"  layers_config = {layers_config}")
        
    except ValueError:
        print("Error: All arguments must be integers")
        sys.exit(1)
    
    # GPU
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Force use CPU
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    model = LevelizedModel(n_inputs=n_inputs, layers_config=layers_config)
    model = model.to(device)  # Move model to GPU
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, betas=(0, 0.9), eps=1e-8)
    
    # Load training data
    train_data_file = "dataset/3bit_Multiplier.txt"
    train_input_data, train_output_data = load_data(train_data_file)
    train_input_data = train_input_data.to(device)  # Move data to GPU
    train_output_data = train_output_data.to(device)
    
    # Load test data
    test_data_file = "dataset/3bit_Multiplier.txt"
    test_input_data, test_output_data = load_data(test_data_file)
    test_input_data = test_input_data.to(device)  # Move data to GPU
    test_output_data = test_output_data.to(device)
    
    print(f"Loaded {len(train_input_data)} training samples")
    print(f"Loaded {len(test_input_data)} test samples")
    
    # Patience mechanism for early stopping
    patience = 100
    patience_counter = 0
    best_loss = float('inf')
    best_model_state = None
    
    # Record start time
    start_time = time.time()
    # Open results file in append mode
    with open("results.txt", "a") as f:
        f.write(f"\n=== Training Session Started ===\n")
        f.write(f"Device: {device}\n")
        f.write(f"Network Architecture: {layers_config}\n")
        f.write(f"Number of Inputs: {n_inputs}\n")
        f.write(f"Training samples: {len(train_input_data)}\n")
        f.write(f"Patience: {patience}\n")
        f.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
        
        early_stop = False
        final_step = 0
        stop_reason = ""
        
        for step in range(10000):
            # Use all training samples (batch training)
            node_ab_values = model(train_input_data)

            # --- Calculate loss ---
            loss = backprop_dc(model, node_ab_values, train_output_data)

            # --- Adam update ---
            optimizer.zero_grad()
            loss.backward()   # Automatically compute gradients for param_w0, param_w1
            optimizer.step()

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
                f.write(f"Step {step:04d} | Loss = {current_loss:.6f} | New best! (patience reset)\n")
                f.flush()
            else:
                # Loss did not improve
                patience_counter += 1
                if step % 10 == 0:  # Write every 10 steps to reduce file size
                    f.write(f"Step {step:04d} | Loss = {current_loss:.6f} | Patience: {patience_counter}/{patience}\n")
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
                f.flush()
                break

            # Write to file and check early stopping condition
            if step % 10 == 0:  # Write every 10 steps to reduce file size
                f.write(f"Step {step:03d} | Loss = {loss.item():.4f}\n")
                f.flush()  # Ensure immediate writing to file
            
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
        
        # Write final test results to file
        with open("results.txt", "a") as f:
            f.write("=== Final Test Results ===\n")
            for i in range(len(test_input_data)):
                f.write(f"Input: {test_input_data[i].int().tolist()}, "
                       f"Expected: {test_output_data[i].int().tolist()}, "
                       f"Predicted: {test_pred_bits[i].tolist()}\n")
            f.write("=== Training Session Completed ===\n\n")
    print("Test completed.")