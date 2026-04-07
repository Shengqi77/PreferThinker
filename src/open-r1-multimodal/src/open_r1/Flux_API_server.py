import torch
from diffusers import FluxPipeline
from flask import Flask, request, send_file, jsonify
import threading
import time
import os
import shutil
import uuid
from datetime import datetime
import traceback

# Set CUDA device
torch.cuda.set_device(0)

app = Flask(__name__)
pipe = None
init_lock = threading.Lock()
init_attempted = False
last_error = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def initialize_flux_pipeline(model_path="black-forest-labs/FLUX.1-schnell"):
    global pipe, init_attempted, last_error, generator
    generator = torch.Generator(device=device).manual_seed(1024)

    if init_attempted or pipe is not None:
        return pipe
    
    print("🔒 Acquiring model initialization lock...")
    with init_lock:
        if not init_attempted:
            init_attempted = True
            print(f"🔄 Starting to load Flux model (Path: {model_path})...")
            try:
                pipe = FluxPipeline.from_pretrained(
                    model_path, 
                    torch_dtype=torch.bfloat16
                ).to(device)     
                
                # Performance optimization
                pipe.enable_attention_slicing()
                
                pipe.to(device)
                print("✅ Flux model loaded successfully")
                return pipe
            except Exception as e:
                last_error = str(e)
                print(f"❌ Model loading failed: {last_error}")
                traceback.print_exc() 
                return None
    return pipe

@app.route('/status', methods=['GET'])
def service_status():
    # Attempt to initialize if not already done
    initialize_flux_pipeline()
    
    status = {
        "status": "ready" if pipe else "initializing",
        "model_loaded": bool(pipe),
        "init_attempted": init_attempted,
        "error": last_error or ""
    }
    return jsonify(status)

@app.route('/generate', methods=['POST'])
def generate_image():
    if not pipe:
        return jsonify({"error": "Model not initialized"}), 503

    prompt = request.json.get('prompt', '')
    print(f"Prompt received: {prompt}")
    print("Starting image generation...")

    unique_id = uuid.uuid4().hex
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Ensure output directory exists
    folder_path = "./tmp/flux_server_data/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    filename = f"image_{timestamp}_{unique_id}.png"
    filepath = os.path.join(folder_path, filename)  

    try:
        with torch.cuda.device(0):
            image = pipe(
                prompt,
                generator=generator,
                height=512,
                width=512,
                guidance_scale=0,
                num_inference_steps=4,
            ).images[0]
            image.save(filepath)
        
        return send_file(filepath, mimetype='image/png')
    except Exception as e:
        print(f"Generation error: {str(e)}")
        return jsonify({"error": "Failed to generate image"}), 500


if __name__ == '__main__':
    # Pre-load the model before starting the server
    initialize_flux_pipeline()

    app.run(
        host='127.0.0.1', 
        port=5000, 
        threaded=True, 
        debug=False
    )