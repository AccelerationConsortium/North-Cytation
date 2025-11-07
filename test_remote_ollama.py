#!/usr/bin/env python3
"""
Demo program to test connecting to remote Ollama server.
Tests basic connectivity and model interaction.
"""

import requests
import json
import time

# Your colleague's Ollama server
OLLAMA_URL = "https://mac-llm.tail2a00e9.ts.net/ollama"

def test_server_connection():
    """Test if the Ollama server is reachable."""
    try:
        print("Testing server connection...")
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        if response.status_code == 200:
            print("✅ Server is reachable!")
            data = response.json()
            models = data.get('models', [])
            print(f"Available models: {[m['name'] for m in models]}")
            return models
        else:
            print(f"❌ Server returned status {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection failed: {e}")
        return None

def ask_ollama(prompt, model_name):
    """Send a prompt to Ollama and get response."""
    try:
        print(f"\nSending prompt to model '{model_name}'...")
        print(f"Prompt: {prompt}")
        
        url = f"{OLLAMA_URL}/api/generate"
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False  # Get complete response at once
        }
        
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        elapsed = time.time() - start_time
        
        print(f"✅ Response received in {elapsed:.1f}s")
        print(f"Response: {result['response']}")
        return result['response']
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None
    except KeyError as e:
        print(f"❌ Unexpected response format: {e}")
        print(f"Full response: {response.text}")
        return None

def main():
    """Main demo function."""
    print("=== Remote Ollama Demo ===")
    print(f"Target server: {OLLAMA_URL}")
    
    # Test connection and get available models
    models = test_server_connection()
    if not models:
        print("Cannot connect to server. Exiting.")
        return
    
    # Use the first available model
    if models:
        model_name = models[0]['name']
        print(f"\nUsing model: {model_name}")
    else:
        print("No models available. Trying 'llama2' as default...")
        model_name = "llama2"
    
    # Test with simple questions
    test_prompts = [
        "What is 2 + 2?",
        "Explain what a polymer is in one sentence.",
        "What is the capital of France?"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i}/3 ---")
        response = ask_ollama(prompt, model_name)
        if response is None:
            print("Skipping remaining tests due to error.")
            break
        
        # Small delay between requests
        if i < len(test_prompts):
            time.sleep(1)
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    main()