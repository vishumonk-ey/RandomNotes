import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

class LightweightCodeGenerator:
    def __init__(self, model_name="bigcode/starcoderbase-1b", device=None):
        """
        Initialize the code generator with a lightweight pretrained model.
        
        Args:
            model_name: The name of the pretrained model to use
            device: The device to use (cpu or cuda if available)
        """
        # Determine device (use CPU if CUDA not available)
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Fix for models where pad_token is not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model in 8-bit precision for memory efficiency if supported
        print("Loading model (this might take a moment)...")
        if self.device == "cuda" and hasattr(torch, "int8"):
            # 8-bit quantization for better memory efficiency
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                device_map="auto",
                load_in_8bit=True
            )
        else:
            # Load on CPU normally
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.to(self.device)
        
        print("Model loaded successfully!")

    def generate_code(self, prompt, max_length=150, temperature=0.7, top_p=0.95, num_return_sequences=1):
        """
        Generate code based on the given prompt.
        
        Args:
            prompt: The text prompt to generate code from
            max_length: Maximum length of the generated code
            temperature: Controls randomness (lower means more deterministic)
            top_p: Nucleus sampling parameter
            num_return_sequences: Number of code samples to generate
            
        Returns:
            List of generated code strings
        """
        try:
            # Tokenize the prompt with explicit padding
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True, 
                return_attention_mask=True
            ).to(self.device)
            
            # Generate text with explicit attention mask
            with torch.no_grad():
                generated_ids = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=num_return_sequences,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode the generated text
            generated_code = []
            for i in range(num_return_sequences):
                code = self.tokenizer.decode(generated_ids[i], skip_special_tokens=True)
                generated_code.append(code)
            
            return generated_code
        
        except Exception as e:
            print(f"Error during code generation: {e}")
            return [f"Error: {str(e)}"]


def main():
    # Example usage
    print("Initializing code generator with a lightweight pretrained model...")
    
    # Create the code generator
    code_gen = LightweightCodeGenerator()
    
    # Example prompt for Python function
    prompt = "# Function to calculate factorial of a number\ndef factorial(n):"
    
    print("\nGenerating code for prompt:")
    print(prompt)
    print("\nGenerating...")
    
    start_time = time.time()
    generated_code = code_gen.generate_code(prompt)
    end_time = time.time()
    
    print(f"\nGeneration completed in {end_time - start_time:.2f} seconds")
    print("\nGenerated Code:")
    print("-" * 50)
    print(generated_code[0])
    print("-" * 50)
    
    # Example prompt for a simple sorting algorithm
    prompt = "# Implement merge sort algorithm in Python\ndef merge_sort(arr):"
    
    print("\nGenerating code for another prompt:")
    print(prompt)
    print("\nGenerating...")
    
    start_time = time.time()
    generated_code = code_gen.generate_code(prompt)
    end_time = time.time()
    
    print(f"\nGeneration completed in {end_time - start_time:.2f} seconds")
    print("\nGenerated Code:")
    print("-" * 50)
    print(generated_code[0])
    print("-" * 50)


if __name__ == "__main__":
    main() 
