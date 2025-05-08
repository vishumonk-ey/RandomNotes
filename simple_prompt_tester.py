from lightweight_code_generator import LightweightCodeGenerator
import sys
from transformers import BitsAndBytesConfig
import torch

def main():
    """Simple script to test code generation with a single prompt"""
    
    # Check if a prompt was provided as a command line argument
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        # Get prompt from user input
        print("Enter your code prompt (e.g., 'def fibonacci(n):'):")
        prompt = input().strip()
    
    # Initialize the code generator
    print("\nInitializing code generator (this may take a moment)...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    generator = LightweightCodeGenerator(model_name="bigcode/starcoderbase-1b")
    
    # Generate code
    print(f"\nGenerating code for prompt: {prompt}\n")
    generated_code = generator.generate_code(prompt)[0]
    
    # Print the result
    print("-" * 60)
    print("GENERATED CODE:")
    print("-" * 60)
    print(generated_code)
    print("-" * 60)
    
    # Offer to save to file
    # save = input("\nSave this code to a file? (y/n): ").lower().strip()
    # # if save == 'y':
    #     filename = input("Enter filename (default: generated_code.py): ").strip() or "generated_code.py"
    #     if not filename.endswith('.py'):
    #         filename += '.py'
            
    #     with open(filename, 'w') as f:
    #         f.write(generated_code)
    #     print(f"Code saved to {filename}")

if __name__ == "__main__":
    main() 
