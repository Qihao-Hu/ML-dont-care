import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import numpy as np

from model import LevelizedModel

def binary_to_decimal(binary_list):
    """Convert binary list to decimal"""
    return sum(bit * (2 ** i) for i, bit in enumerate(reversed(binary_list)))

def decimal_to_binary(decimal, bits=4):
    """Convert decimal to binary list"""
    return [(decimal >> i) & 1 for i in range(bits)]

def test_4x4_multiplier():
    print("=== 4×4 Multiplier Model Test ===")
    print("Configuration:")
    print("- Input: 8 bits (two 4-bit numbers)")
    print("- Layers: [4, 6, 8] (4 nodes, 6 nodes, 8 nodes)")
    print("- Output: 8 bits (8-bit product)")
    print()
    
    # Create the 4x4 multiplier model
    model = LevelizedModel(n_inputs=8, layers_config=[4, 6, 8])
    
    # Test case: 3 × 5 = 15
    a = 3  # 0011 in binary
    b = 5  # 0101 in binary
    
    a_binary = decimal_to_binary(a, 4)
    b_binary = decimal_to_binary(b, 4)
    
    # Construct input: [a0, a1, a2, a3, b0, b1, b2, b3]
    test_input = torch.tensor([a_binary + b_binary], dtype=torch.float32)
    
    print(f"Test multiplication: {a} × {b} = {a * b}")
    print(f"Binary representation:")
    print(f"  a = {a} = {a_binary} (LSB first)")
    print(f"  b = {b} = {b_binary} (LSB first)")
    print(f"  Expected product = {a * b} = {decimal_to_binary(a * b, 8)}")
    print(f"Input tensor: {test_input}")
    print()
    
    # Run forward pass and collect detailed information
    print("=== Forward Pass Analysis ===")
    
    # Manual forward pass with detailed logging
    B = test_input.shape[0]
    device = test_input.device
    node_vals = {}
    all_details = []
    
    # Process inputs
    print("Input Processing:")
    for i in range(model.n_inputs):
        node_vals[i] = test_input[:, i] * 2.0 - 1.0  # 0→-1, 1→+1
        print(f"  Input node {i}: {test_input[:, i].item()} → {node_vals[i].item()}")
    print()
    
    param_idx = 0
    
    for layer_idx, layer_nodes in enumerate(model.layers):
        print(f"=== Layer {layer_idx + 1} ({len(layer_nodes)} nodes) ===")
        
        for node_idx, node in enumerate(layer_nodes):
            print(f"\n--- Node {node} (param_idx={param_idx}) ---")
            
            # Get parameters and candidates
            cand = model.candidates[param_idx]
            w0 = model.param_w0[param_idx]
            w1 = model.param_w1[param_idx]
            M_i = len(cand)
            
            print(f"Candidates: {cand}")
            print(f"M_i = {M_i}, Parameter length = {len(w0)} (should be 2*{M_i})")
            
            # Collect candidate values and construct yi
            candidate_values = []
            for cand_idx in cand:
                candidate_values.append(node_vals[cand_idx])
            x = torch.stack(candidate_values, dim=0)
            yi = torch.cat([x, -x], dim=0)
            
            print(f"x (candidate values): {x.squeeze().tolist()}")
            print(f"yi ([x, -x]): {yi.squeeze().tolist()}")
            
            # Apply formulas
            softmax_w0 = torch.softmax(w0, dim=0)
            softmax_w1 = torch.softmax(w1, dim=0)
            
            argmax_w0 = int(torch.argmax(w0).item())
            argmax_w1 = int(torch.argmax(w1).item())
            
            ap0 = yi[argmax_w0]
            bp0 = torch.sum(yi * softmax_w0.unsqueeze(1), dim=0)
            ap1 = yi[argmax_w1]
            bp1 = torch.sum(yi * softmax_w1.unsqueeze(1), dim=0)
            
            # AND gate
            in0 = ap0
            in1 = ap1
            out = torch.where((in0 > 0) & (in1 > 0), torch.tensor(1.0, device=device), torch.tensor(-1.0, device=device))
            
            node_vals[node] = out
            
            # Store details
            node_details = {
                'node_id': node,
                'layer': layer_idx + 1,
                'candidates': cand,
                'omega_0': w0.detach().cpu().numpy().tolist(),
                'omega_1': w1.detach().cpu().numpy().tolist(),
                'x_values': x.squeeze().cpu().numpy().tolist(),
                'yi_values': yi.squeeze().cpu().numpy().tolist(),
                'argmax_w0': argmax_w0,
                'argmax_w1': argmax_w1,
                'ap0': ap0.item(),
                'ap1': ap1.item(),
                'bp0': bp0.item(),
                'bp1': bp1.item(),
                'output': out.item(),
                'softmax_w0': softmax_w0.detach().cpu().numpy().tolist(),
                'softmax_w1': softmax_w1.detach().cpu().numpy().tolist()
            }
            all_details.append(node_details)
            
            print(f"ω0: {[f'{x:.4f}' for x in w0.detach().cpu().numpy()]}")
            print(f"ω1: {[f'{x:.4f}' for x in w1.detach().cpu().numpy()]}")
            print(f"Softmax ω0: {[f'{x:.4f}' for x in softmax_w0.detach().cpu().numpy()]}")
            print(f"Softmax ω1: {[f'{x:.4f}' for x in softmax_w1.detach().cpu().numpy()]}")
            print(f"argmax(ω0) = {argmax_w0} → ap0 = yi[{argmax_w0}] = {ap0.item():.4f}")
            print(f"argmax(ω1) = {argmax_w1} → ap1 = yi[{argmax_w1}] = {ap1.item():.4f}")
            print(f"bp0 = {bp0.item():.4f}")
            print(f"bp1 = {bp1.item():.4f}")
            print(f"AND({ap0.item():.4f}, {ap1.item():.4f}) = {out.item():.4f}")
            
            param_idx += 1
        
        print()
    
    # Get final outputs
    print("=== Final Output Analysis ===")
    outputs, softmax_b, argmax_idx = model(test_input)
    
    print(f"Output nodes: {model.output_nodes}")
    print(f"Output values (0/1): {outputs.squeeze().tolist()}")
    
    # Convert output to decimal
    output_binary = outputs.squeeze().cpu().numpy().tolist()
    output_decimal = binary_to_decimal(output_binary)
    expected_decimal = a * b
    
    print(f"Output as decimal: {output_decimal}")
    print(f"Expected decimal: {expected_decimal}")
    print(f"Correct: {output_decimal == expected_decimal}")
    print()
    
    # Save all details to file
    print("=== Saving Details to File ===")
    with open('multiplier_analysis.txt', 'w') as f:
        f.write("4×4 Multiplier Model Analysis\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test Case: {a} × {b} = {a * b}\n")
        f.write(f"Input: {test_input.squeeze().tolist()}\n")
        f.write(f"Expected Output: {decimal_to_binary(a * b, 8)}\n")
        f.write(f"Actual Output: {output_binary}\n")
        f.write(f"Correct: {output_decimal == expected_decimal}\n\n")
        
        for details in all_details:
            f.write(f"Node {details['node_id']} (Layer {details['layer']})\n")
            f.write("-" * 30 + "\n")
            f.write(f"Candidates: {details['candidates']}\n")
            f.write(f"x_values: {details['x_values']}\n")
            f.write(f"yi_values: {details['yi_values']}\n")
            f.write(f"omega_0: {details['omega_0']}\n")
            f.write(f"omega_1: {details['omega_1']}\n")
            f.write(f"softmax_omega_0: {details['softmax_w0']}\n")
            f.write(f"softmax_omega_1: {details['softmax_w1']}\n")
            f.write(f"argmax_omega_0: {details['argmax_w0']}\n")
            f.write(f"argmax_omega_1: {details['argmax_w1']}\n")
            f.write(f"ap0: {details['ap0']}\n")
            f.write(f"ap1: {details['ap1']}\n")
            f.write(f"bp0: {details['bp0']}\n")
            f.write(f"bp1: {details['bp1']}\n")
            f.write(f"output: {details['output']}\n\n")
    
    print("Analysis saved to 'multiplier_analysis.txt'")
    print("✅ 4×4 Multiplier model test completed!")

if __name__ == "__main__":
    test_4x4_multiplier()
