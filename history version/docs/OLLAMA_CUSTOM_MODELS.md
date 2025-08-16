# 🔧 Adding Custom Models to Ollama

## Method 1: Import GGUF Files

### Step 1: Create a Modelfile
Create a text file called `Modelfile` (no extension):

```
FROM ./path/to/your/model.gguf

TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

"""

PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
```

### Step 2: Import the Model
```cmd
ollama create my-custom-model -f Modelfile
```

### Step 3: Use in S2T2SS
Update your config:
```python
LLM_SERVER_URL = "http://10.0.0.43:11434"
# The model will auto-detect as "my-custom-model"
```

## Method 2: Download More Official Models

### Chinese Language Models
```cmd
# Qwen models (excellent for Chinese)
ollama pull qwen2.5:7b         # Best Chinese support
ollama pull qwen2.5:14b        # Better quality, more resources
ollama pull qwen2.5:3b         # Faster, smaller

# Other multilingual options
ollama pull llama3.1:8b        # Good multilingual
ollama pull mistral:7b         # Fast general purpose
ollama pull gemma2:9b          # Google's model
```

### Specialized Models
```cmd
# Fast lightweight models
ollama pull gemma2:2b          # Very fast
ollama pull llama3.2:1b        # Ultra fast

# Larger, higher quality
ollama pull llama3.1:70b       # Best quality (needs lots of RAM)
ollama pull qwen2.5:32b        # High quality Chinese
```

## Method 3: Custom GGUF from Hugging Face

### Step 1: Download GGUF
From Hugging Face, download any `.gguf` file, for example:
- `Qwen2.5-7B-Instruct-Q4_K_M.gguf`
- `Yi-34B-Chat-Q4_K_M.gguf`
- Any custom fine-tuned model

### Step 2: Create Advanced Modelfile
```
FROM ./Qwen2.5-7B-Instruct-Q4_K_M.gguf

TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""

PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.7
PARAMETER top_p 0.8
PARAMETER repeat_penalty 1.05
```

### Step 3: Import and Test
```cmd
ollama create qwen2.5-custom -f Modelfile
ollama run qwen2.5-custom "Hello, test in Chinese: 你好"
```

## Method 4: Model Management Commands

### List Available Models
```cmd
ollama list
```

### Remove Models
```cmd
ollama rm model-name
```

### Show Model Info
```cmd
ollama show llama3.2:latest
```

## Examples for Different Use Cases

### For Chinese Speech Processing
```cmd
# Best Chinese understanding
ollama pull qwen2.5:7b

# If you want custom Chinese model
# 1. Download from HuggingFace: Qwen2.5-7B-Instruct-GGUF
# 2. Create Modelfile with Chinese template
# 3. ollama create qwen-chinese -f Modelfile
```

### For Speed (Real-time Processing)
```cmd
# Ultra fast models
ollama pull gemma2:2b
ollama pull llama3.2:1b
```

### For Quality (When speed isn't critical)
```cmd
# High quality models
ollama pull qwen2.5:14b
ollama pull llama3.1:70b  # Needs 40GB+ RAM
```

## S2T2SS Integration

After adding any model, S2T2SS will automatically detect it:

1. **Auto-detection**: S2T2SS queries `/v1/models` and uses the first available
2. **Manual selection**: You could modify the code to prefer specific models
3. **Fallback**: If your custom model fails, it falls back to available models

## Performance Tips

### Model Size vs Speed
- **2B models**: ~0.1s response time
- **7B models**: ~0.5s response time  
- **14B+ models**: 1s+ response time

### For S2T2SS Use Case
Since you want real-time processing, I recommend:
- **qwen2.5:3b** - Good balance of Chinese support and speed
- **gemma2:2b** - Very fast, decent multilingual
- **llama3.2:1b** - Ultra fast for simple text cleanup

Would you like me to help you set up a specific custom model, or would you prefer to try some of the official Chinese-optimized models first?
