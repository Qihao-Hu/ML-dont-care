#!/usr/bin/env python3
"""
Generate multiplier dataset files for training neural networks.
Creates a dataset with all possible multiplication combinations for n-bit numbers.
"""

import argparse
import random


def int_to_binary_list(num, bit_width):
    """Convert an integer to a binary list of specified bit width."""
    binary_str = format(num, f'0{bit_width}b')
    return [int(b) for b in binary_str]


def generate_multiplier_dataset(n_bits, output_file, shuffle=False):
    """
    Generate a complete multiplier dataset for n-bit numbers.
    
    Args:
        n_bits: Number of bits for each input number
        output_file: Path to the output file
        shuffle: Whether to shuffle the dataset (default: False for ordered)
    """
    max_value = 2 ** n_bits - 1
    output_bits = 2 * n_bits
    
    # Generate all multiplication combinations
    dataset = []
    for a in range(max_value + 1):
        for b in range(max_value + 1):
            result = a * b
            
            # Convert to binary lists
            a_binary = int_to_binary_list(a, n_bits)
            b_binary = int_to_binary_list(b, n_bits)
            result_binary = int_to_binary_list(result, output_bits)
            
            # Create input (concatenate a and b)
            inputs = a_binary + b_binary
            outputs = result_binary
            
            # Format as string
            input_str = ','.join(map(str, inputs))
            output_str = ','.join(map(str, outputs))
            line = f"Inputs=[{input_str}] Outputs=[{output_str}] # {a}*{b}={result}"
            
            dataset.append(line)
    
    # Shuffle if requested
    if shuffle:
        random.shuffle(dataset)
    
    # Write to file
    with open(output_file, 'w') as f:
        for line in dataset:
            f.write(line + '\n')
    
    print(f"Generated {len(dataset)} samples for {n_bits}-bit multiplier")
    print(f"Input size: {2 * n_bits} bits")
    print(f"Output size: {output_bits} bits")
    print(f"Saved to: {output_file}")


def generate_sample_dataset(n_bits, num_samples, output_file):
    """
    Generate a random sample of multiplier dataset for n-bit numbers.
    
    Args:
        n_bits: Number of bits for each input number
        num_samples: Number of samples to generate
        output_file: Path to the output file
    """
    max_value = 2 ** n_bits - 1
    output_bits = 2 * n_bits
    
    # Generate random samples
    dataset = set()
    while len(dataset) < num_samples:
        a = random.randint(0, max_value)
        b = random.randint(0, max_value)
        result = a * b
        
        # Convert to binary lists
        a_binary = int_to_binary_list(a, n_bits)
        b_binary = int_to_binary_list(b, n_bits)
        result_binary = int_to_binary_list(result, output_bits)
        
        # Create input (concatenate a and b)
        inputs = a_binary + b_binary
        outputs = result_binary
        
        # Format as string
        input_str = ','.join(map(str, inputs))
        output_str = ','.join(map(str, outputs))
        line = f"Inputs=[{input_str}] Outputs=[{output_str}] # {a}*{b}={result}"
        
        dataset.add(line)
    
    # Convert to list and shuffle
    dataset = list(dataset)
    random.shuffle(dataset)
    
    # Write to file
    with open(output_file, 'w') as f:
        for line in dataset:
            f.write(line + '\n')
    
    print(f"Generated {len(dataset)} random samples for {n_bits}-bit multiplier")
    print(f"Input size: {2 * n_bits} bits")
    print(f"Output size: {output_bits} bits")
    print(f"Saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate multiplier dataset for neural network training'
    )
    parser.add_argument('n_bits', type=int, help='Number of bits for each input number')
    parser.add_argument('-o', '--output', type=str, 
                       help='Output file path (default: dataset/Nbit_Multiplier.txt)')
    parser.add_argument('-s', '--shuffle', action='store_true',
                       help='Shuffle the dataset (default: ordered)')
    parser.add_argument('-n', '--num-samples', type=int,
                       help='Generate only N random samples instead of all combinations')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for shuffling/sampling (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Determine output file
    if args.output:
        output_file = args.output
    else:
        output_file = f'dataset/{args.n_bits}bit_Multiplier.txt'
    
    # Generate dataset
    if args.num_samples:
        generate_sample_dataset(args.n_bits, args.num_samples, output_file)
    else:
        generate_multiplier_dataset(args.n_bits, output_file, args.shuffle)


if __name__ == '__main__':
    main()
