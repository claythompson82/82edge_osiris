from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Define the FastAPI app
app = FastAPI()

# Define the request body model
class PromptRequest(BaseModel):
    prompt: str
    max_length: int = 128

# Load the model and tokenizer when the server starts
# Ensure the model_name_or_path points to the location where the model was cloned in the Dockerfile
model_name_or_path = "/app/hermes-model" 
device = "cuda" if torch.cuda.is_available() else "cpu"

# It's crucial that the model and tokenizer are compatible with the AutoGPTQ version used.
# For auto-gptq 0.7.1, use_safetensors might need to be True if .safetensors files are present.
# device_map="auto" should correctly utilize the GPU.
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto", # Automatically map model parts to available devices (CPU/GPU)
        trust_remote_code=True, # Required for some models
        # use_safetensors=True, # Uncomment if using .safetensors and it's required
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    print(f"Model and tokenizer loaded successfully on device: {device}")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    # Fallback or error handling if model loading fails
    model = None
    tokenizer = None

@app.post("/generate/")
async def generate_text(request: PromptRequest):
    if not model or not tokenizer:
        return {"error": "Model not loaded. Please check server logs."}

    try:
        inputs = tokenizer(request.prompt, return_tensors="pt").to(device)
        
        # Generate text
        # Ensure generation parameters are appropriate for the model
        output_sequences = model.generate(
            input_ids=inputs.input_ids,
            max_length=request.max_length,
            # num_beams=5, # Example: Using beam search for potentially better quality
            # no_repeat_ngram_size=2, # Example: Prevent repeating n-grams
            # early_stopping=True # Example: Stop generation early if EOS token is found
        )
        
        generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        return {"generated_text": generated_text}
    except Exception as e:
        print(f"Error during text generation: {e}")
        return {"error": f"Failed to generate text: {str(e)}"}

@app.get("/health")
async def health_check():
    if model and tokenizer:
        return {"status": "ok", "model_loaded": True, "device": str(model.device)}
    else:
        return {"status": "error", "model_loaded": False, "message": "Model or tokenizer not loaded."}

if __name__ == "__main__":
    import uvicorn
    # This part is for local development/testing and won't be used when running with Docker
    # Docker uses the CMD instruction in the Dockerfile
    uvicorn.run(app, host="0.0.0.0", port=8000)
