"""
Test script for LevelizedModel - outputs detailed node information
Outputs all node parameters (ap0, ap1, bp0, bp1, w0, w1, out) to a text file
"""

import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from model import LevelizedModel

def test_detailed_node_output():
    """Test: Output all node parameters to file"""
    print("=" * 80)
    print("Model Node Parameters Test")
    print("=" * 80)
    
    # Create model
    n_inputs = 6
    layers_config = [10, 10, 6]
    model = LevelizedModel(n_inputs=n_inputs, layers_config=layers_config)
    
    print(f"Model Configuration:")
    print(f"  n_inputs: {n_inputs}")
    print(f"  layers_config: {layers_config}")
    print(f"  Total layers: {len(model.layers)}")
    print()
    
    # Create test input (single sample for clarity)
    test_input = torch.tensor([[1, 0, 1, 0, 1, 0]], dtype=torch.float32)
    print(f"Test Input: {test_input.tolist()[0]}")
    print()
    
    # Forward pass
    node_ab_values = model(test_input)
    
    # Open output file
    output_file = "node_parameters_output.txt"
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL NODE PARAMETERS DETAILED OUTPUT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Model Configuration:\n")
        f.write(f"  n_inputs: {n_inputs}\n")
        f.write(f"  layers_config: {layers_config}\n")
        f.write(f"  Test input: {test_input.tolist()[0]}\n\n")
        
        # Output input nodes
        f.write("=" * 80 + "\n")
        f.write("INPUT NODES (0 - {})\n".format(n_inputs - 1))
        f.write("=" * 80 + "\n\n")
        
        for i in range(n_inputs):
            input_val = test_input[0, i].item()
            mapped_val = input_val * 2.0 - 1.0  # Map 0->-1, 1->1
            f.write(f"Input Node {i}:\n")
            f.write(f"  Input bit:     {input_val}\n")
            f.write(f"  Mapped value:  {mapped_val:.4f}\n")
            f.write("\n")
        
        # Output hidden and output layer nodes
        param_idx = 0
        for layer_idx, layer_nodes in enumerate(model.layers):
            f.write("=" * 80 + "\n")
            f.write(f"LAYER {layer_idx + 1} - Nodes {layer_nodes[0]} to {layer_nodes[-1]}\n")
            if layer_idx == len(model.layers) - 1:
                f.write("(OUTPUT LAYER - w1 should be all zeros)\n")
            f.write("=" * 80 + "\n\n")
            
            for node_id in layer_nodes:
                # Get node values
                node_data = node_ab_values[node_id]
                
                # Get parameters
                w0 = model.param_w0[param_idx]
                w1 = model.param_w1[param_idx]
                
                f.write(f"Node {node_id}:\n")
                f.write(f"-" * 80 + "\n")
                
                # Output a, b values
                f.write(f"  ap0: {node_data['ap0'][0].item():.6f}\n")
                f.write(f"  ap1: {node_data['ap1'][0].item():.6f}\n")
                f.write(f"  bp0: {node_data['bp0'][0].item():.6f}\n")
                f.write(f"  bp1: {node_data['bp1'][0].item():.6f}\n")
                f.write(f"  out: {node_data['out'][0].item():.6f}\n")
                f.write(f"\n")
                
                # Output w0 parameters - FULL ARRAY
                f.write(f"  w0 (length={len(w0)}, requires_grad={w0.requires_grad}):\n")
                w0_list = w0.detach().tolist()
                f.write(f"    Values: {w0_list}\n")
                f.write(f"    Stats: Mean={w0.mean().item():.6f}, Std={w0.std().item():.6f}, ")
                f.write(f"Min={w0.min().item():.6f}, Max={w0.max().item():.6f}\n")
                f.write(f"\n")
                
                # Output w1 parameters - FULL ARRAY
                f.write(f"  w1 (length={len(w1)}, requires_grad={w1.requires_grad}):\n")
                w1_list = w1.detach().tolist()
                f.write(f"    Values: {w1_list}\n")
                f.write(f"    Stats: Mean={w1.mean().item():.6f}, Std={w1.std().item():.6f}, ")
                f.write(f"Min={w1.min().item():.6f}, Max={w1.max().item():.6f}\n")
                f.write(f"    All zeros: {torch.all(w1 == 0).item()}\n")
                f.write(f"\n")
                
                param_idx += 1
        
        f.write("=" * 80 + "\n")
        f.write("END OF OUTPUT\n")
        f.write("=" * 80 + "\n")
    
    print(f"✓ Node parameters written to: {output_file}")
    print(f"  Total nodes processed: {len(node_ab_values)}")
    print(f"  Output file size: {os.path.getsize(output_file)} bytes")
    
    return model

def main():
    """Run the test"""
    print("\n" + "=" * 80)
    print("MODEL NODE PARAMETERS TEST")
    print("=" * 80)
    print()
    
    # Run test
    model = test_detailed_node_output()
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)
    print("\nCheck 'node_parameters_output.txt' for detailed node information.")

if __name__ == "__main__":
    main()
