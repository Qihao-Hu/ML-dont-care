import torch
import time
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
    f = {}
    g = {}
    t = {}

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

    # We assume output_bits correspond to output_layer order; user used output_nodes earlier but
    # here follow your prior code that set f for nodes in last layer.
    # Map {0,1} -> {-1,1}
    transformed = (output_bits[:, :num_outputs].float() * 2.0 - 1.0)  # [B, num_outputs]
    # assign into f for each q in output_layer
    for idx, q in enumerate(output_layer):
        val = transformed[:, idx]               # [B]
        f[q] = torch.stack([val, val], dim=1)   # [B,2]

    loss = torch.tensor(0.0, device=device)

    # traverse layers backwards (excluding the output layer which is set)
    for layer_idx in range(len(model.layers) - 2, -1, -1):
        layer_nodes = model.layers[layer_idx]
        layer_num = layer_idx + 1

        for p in layer_nodes:
            # t_p is scalar (accumulate weights for node p)
            t_p = torch.tensor(0.0, device=device)
            # g_p is per-batch
            g_p = torch.zeros(B, device=device)

            if layer_num == len(model.layers) - 1:
                # Special handling matching the non-comment backprop_dc for last-1 layer
                for next_layer_idx in range(layer_idx + 1, len(model.layers)):
                    for q in model.layers[next_layer_idx]:
                        row_idx = q - n_inputs
                        w0_row = model.param_w0[row_idx]
                        w1_row = model.param_w1[row_idx]

                        idx0 = p
                        idx1 = p + count_nodes_in_layers_before(model, q)

                        m00 = w0_row[idx0]
                        m01 = w0_row[idx1]

                        # accumulate only m00+m01 for t_p
                        t_p = t_p + (m00 + m01)
                        t_p_safe = t_p if (t_p.abs() > eps).item() else (t_p + eps)

                        fq0 = f[q][:, 0]
                        # g[p] += (m00/t[p]) * f[q,0] + (m01/t[p]) * (- f[q,0])
                        g_p = g_p + (m00 / t_p_safe) * fq0 + (m01 / t_p_safe) * (-fq0)
            else:
                # General case matching the non-comment backprop_dc
                for next_layer_idx in range(layer_idx + 1, len(model.layers)):
                    for q in model.layers[next_layer_idx]:
                        row_idx = q - n_inputs
                        w0_row = model.param_w0[row_idx]
                        w1_row = model.param_w1[row_idx]

                        idx0 = p
                        idx1 = p + count_nodes_in_layers_before(model, q)

                        m00 = w0_row[idx0]
                        m01 = w0_row[idx1]
                        m10 = w1_row[idx0]
                        m11 = w1_row[idx1]

                        # accumulate all four into t_p
                        t_p = t_p + (m00 + m01 + m10 + m11)
                        t_p_safe = t_p if (t_p.abs() > eps).item() else (t_p + eps)

                        fq0 = f[q][:, 0]
                        fq1 = f[q][:, 1]
                        # g[p] += (m00/t)*f[q,0] + (m10/t)*f[q,1] + (m01/t)*(- f[q,0]) + (m11/t)*(- f[q,1])
                        g_p = g_p + (m00 / t_p_safe) * fq0 + (m10 / t_p_safe) * fq1 \
                                   + (m01 / t_p_safe) * (-fq0) + (m11 / t_p_safe) * (-fq1)

            # now set f[p] according to your rules, but do it elementwise for batch
            # node_ab_values[p]['ap0'] and ['ap1'] are tensors [B]
            ap0 = node_ab_values[p]['ap0'].to(device)  # [B]
            ap1 = node_ab_values[p]['ap1'].to(device)  # [B]

            # elementwise masks (python bools per element avoided; keep tensors)
            mask0 = (g_p < 0) | (ap0 == 1)
            mask1 = (ap1 == -1)

            maskA = mask0 & mask1  # condition to set f[p,0]=0

            mask2 = (g_p > 0) | (ap1 == 1)
            mask3 = (ap0 == -1)
            maskB = mask2 & mask3  # condition to set f[p,1]=0

            # default both equal to g_p
            f_p0 = g_p.clone()
            f_p1 = g_p.clone()

            # set f_p0 to 0 where maskA true (elementwise), but use tensor ops (no in-place)
            f_p0 = torch.where(maskA, torch.zeros_like(f_p0), f_p0)

            # set f_p1 to 0 where maskB true
            f_p1 = torch.where(maskB, torch.zeros_like(f_p1), f_p1)

            # stack back into [B,2] and store
            f[p] = torch.stack([f_p0, f_p1], dim=1)
            t[p] = t_p
            g[p] = g_p

            # compute loss term (use tensor ops)
            bp0 = node_ab_values[p]['bp0'].to(device)  # [B]
            bp1 = node_ab_values[p]['bp1'].to(device)  # [B]

            # use torch.sign rather than python sign to remain in graph
            sign_f0 = torch.sign(f_p0)  # returns -1,0,1 per element
            sign_f1 = torch.sign(f_p1)

            term0 = torch.abs(f_p0) * ((bp0 - sign_f0) ** 2)  # [B]
            term1 = torch.abs(f_p1) * ((bp1 - sign_f1) ** 2)  # [B]

            # sum over batch and add to scalar loss
            loss = loss + term0.sum() + term1.sum()

    # final: average over batch size (optional; keep consistent with optimizer scale)
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
    model = LevelizedModel(n_inputs=6, layers_config=[20, 20, 20, 6])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, betas=(0, 0.9), eps=1e-8)
    
    # Load training data
    train_data_file = "/export/qhu56/ML-dont-care/train/3bit.txt"
    train_input_data, train_output_data = load_data(train_data_file)
    
    # Load test data
    test_data_file = "/export/qhu56/ML-dont-care/test/3bit.txt"
    test_input_data, test_output_data = load_data(test_data_file)
    
    print(f"Loaded {len(train_input_data)} training samples")
    print(f"Loaded {len(test_input_data)} test samples")
    
    # Record start time
    start_time = time.time()
    
    # Open results file in append mode
    with open("results.txt", "a") as f:
        f.write(f"\n=== Training Session Started ===\n")
        f.write(f"Training samples: {len(train_input_data)}\n")
        f.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
        
        early_stop = False
        final_step = 0
        
        for step in range(10000):
            # Use all training samples (batch training)
            node_ab_values = model(train_input_data)

            # --- Calculate loss ---
            loss = backprop_dc(model, node_ab_values, train_output_data)

            # --- Adam update ---
            optimizer.zero_grad()
            loss.backward()   # Automatically compute gradients for param_w0, param_w1
            optimizer.step()

            # Write to file and check early stopping condition
            if step % 10 == 0:  # Write every 10 steps to reduce file size
                f.write(f"Step {step:03d} | Loss = {loss.item():.4f}\n")
                f.flush()  # Ensure immediate writing to file
            
            # Check early stopping condition
            if loss.item() < 0.1:
                final_step = step
                early_stop = True
                end_time = time.time()
                training_time = end_time - start_time
                
                f.write(f"\n=== Early Stopping Triggered ===\n")
                f.write(f"Final Step: {step:03d}\n")
                f.write(f"Final Loss: {loss.item():.4f}\n")
                f.write(f"Training Time: {training_time:.2f} seconds\n")
                f.write(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
                f.flush()
                break
        
        if not early_stop:
            end_time = time.time()
            training_time = end_time - start_time
            final_step = 9999
            f.write(f"\n=== Training Completed (Max Steps Reached) ===\n")
            f.write(f"Final Step: {final_step:03d}\n")
            f.write(f"Training Time: {training_time:.2f} seconds\n")
            f.write(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
    
    if early_stop:
        print(f"Training stopped early at step {final_step} (loss < 0.1)")
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
