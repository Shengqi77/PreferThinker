import requests
import time
import sys
import uuid
from PIL import Image
from io import BytesIO
import pdb
import fcntl


LOCK_FILE = "/tmp/accelerate_api_lock"

def acquire_lock():
    lock_file = open(LOCK_FILE, "w")
    fcntl.flock(lock_file, fcntl.LOCK_EX)
    return lock_file

def release_lock(lock_file):
    fcntl.flock(lock_file, fcntl.LOCK_UN)
    lock_file.close()

def generate_image(prompt, **kwargs):
    """Calls the image generation API and handles the response correctly"""
    lock_file = acquire_lock()
    try:
        try:
            # Check server health/status
            health_response = requests.get("http://127.0.0.1:5000/status", timeout=10)
            health = health_response.json()
            
            if not health.get("model_loaded", False):
                error_msg = health.get("error") or "Model not loaded. Please check server logs."
                print(f"⚠️ {error_msg}")
                return None
        except Exception as e:
            print(f"❌ Service status check failed: {str(e)}")
            return None
        
        request_id = str(uuid.uuid4())[:8]
        
        data = {
            "prompt": prompt,
            "request_id": request_id
        }
        data.update(kwargs)
        
        start_time = time.time()
        print(f"📦 [{request_id}] Sending generation request...")
        
        try:
            response = requests.post(
                "http://127.0.0.1:5000/generate",
                json=data,
                timeout=300
            )

            content_type = response.headers.get('Content-Type', '')
            
            if 'text/html' in content_type:
                elapsed = time.time() - start_time
                print(f"🖼️ [{request_id}] Received image path response (Time: {elapsed:.2f}s)")
                print(f"✅ Reward returned as: {response.content}")
                return response.content

            elif 'image' in content_type:
                elapsed = time.time() - start_time
                print(f"🖼️ [{request_id}] Received image response (Time: {elapsed:.2f}s)")
                
                image = Image.open(BytesIO(response.content))
                return image
            
            elif 'application/json' in content_type:
                try:
                    error_data = response.json()
                    print(f"❌ [{request_id}] Error response (Status {response.status_code}): {error_data}")
                    return None
                except:
                    print(f"❌ [{request_id}] Failed to parse error response (Status {response.status_code}): {response.text[:200]}")
                    return None
            
            else:
                print(f"❌ [{request_id}] Unknown response type: {content_type}")
                print(f"Response preview: {response.text[:200]}")
                return None
        
        except requests.exceptions.Timeout:
            print(f"⏰ [{request_id}] Request timed out! Please check the server status.")
            return None
        except requests.exceptions.ConnectionError:
            print(f"🔌 [{request_id}] Connection failed! Please ensure the service is running.")
            return None
        except Exception as e:
            print(f"❌ [{request_id}] Request exception: {str(e)}")
            return None
        
    finally:
        release_lock(lock_file)
