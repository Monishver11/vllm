import torch
import torch._dynamo
from vllm.config import (
    ModelConfig,
    DeviceConfig,
    LoadConfig,
    ParallelConfig,
)
from vllm.model_executor.model_loader import get_model_loader

# Suppress errors to get a clean trace output
torch._dynamo.config.suppress_errors = True

def main():
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    device = "cuda"
    
    print(f"Loading model: {model_name}")
    
    # 1. Setup Configs
    model_config = ModelConfig(
        model=model_name,
        task="generate",
        tokenizer=model_name,
        tokenizer_mode="auto",
        trust_remote_code=True,
        dtype=torch.float16,
        seed=42,
    )
    
    # These are usually required by the loader internally
    parallel_config = ParallelConfig(1, 1, False)
    load_config = LoadConfig()
    device_config = DeviceConfig(device=device)

    # 2. Initialize the loader
    loader = get_model_loader(load_config)
    
    # 3. Corrected load_model call
    # Most vLLM 0.6+ versions use this signature:
    model = loader.load_model(
        model_config=model_config,
        device_config=device_config, # Some versions want this, some don't
        parallel_config=parallel_config,
        vision_language_config=None,
        lora_config=None,
    )
    
    # IF THE ABOVE STILL FAILS, use the most stripped-down version:
    # model = loader.load_model(model_config=model_config, parallel_config=parallel_config)

    print(f"\nModel loaded. Type: {type(model)}")
    
    # 4. Access the MLP
    # DeepSeek-R1-Distill-Qwen is based on Qwen2 architecture
    layer = model.model.layers[0]
    mlp = layer.mlp
    
    # 5. Trace Setup
    hidden_size = model.config.hidden_size
    x = torch.randn(1, hidden_size, dtype=torch.float16, device=device)
    
    graphs = []
    def capture_graph(gm, inputs):
        graphs.append(gm)
        return gm
    
    print("\nTracing MLP...")
    compiled_mlp = torch.compile(mlp, backend=capture_graph)
    
    with torch.no_grad():
        compiled_mlp(x)
    
    if graphs:
        print(f"\nCaptured {len(graphs)} graph(s).")
        graphs[0].graph.print_tabular()
    else:
        print("\nNo graphs captured. Check if torch.compile is falling back to eager.")

if __name__ == "__main__":
    main()