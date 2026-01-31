import torch
import torch._dynamo

from vllm.config import (
    ModelConfig,
    DeviceConfig,
    LoadConfig,
    ParallelConfig,
    VllmConfig,
)
from vllm.model_executor.model_loader import get_model_loader

# Suppress Dynamo errors for a clean trace
torch._dynamo.config.suppress_errors = True

def main():
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    device = "cuda"

    print(f"Loading model: {model_name}")

    # ---------------------------------------
    # 1. Build configs
    # ---------------------------------------
    model_config = ModelConfig(
        model=model_name,
        task="generate",
        tokenizer=model_name,
        tokenizer_mode="auto",
        trust_remote_code=True,
        dtype=torch.float16,
        seed=42,
    )

    load_config = LoadConfig()
    device_config = DeviceConfig(device=device)
    parallel_config = ParallelConfig(1, 1, False)

    # Build full vLLM config
    vllm_config = VllmConfig(
        model_config=model_config,
        load_config=load_config,
        device_config=device_config,
        parallel_config=parallel_config,
    )

    # ---------------------------------------
    # 2. Load model via loader
    # ---------------------------------------
    loader = get_model_loader(load_config)

    model = loader.load_model(
        vllm_config=vllm_config,
        model_config=model_config,
    )

    print(f"Model loaded: {type(model)}")

    # ---------------------------------------
    # 3. Unwrap HF model
    # ---------------------------------------
    if hasattr(model, "model"):
        hf_model = model.model
    else:
        # Fallback deeper unwrap if needed
        hf_model = (
            model.llm_engine
                 .model_executor
                 .driver_worker
                 .model
                 .model
        )

    print(f"HF model type: {type(hf_model)}")

    # ---------------------------------------
    # 4. Access transformer MLP
    # ---------------------------------------
    first_layer = hf_model.layers[0]
    mlp = first_layer.mlp

    print("\nMLP module:\n", mlp)

    # ---------------------------------------
    # 5. Trace the MLP with torch.compile
    # ---------------------------------------
    hidden_size = hf_model.config.hidden_size
    x = torch.randn(1, hidden_size, dtype=torch.float16, device=device)

    graphs = []
    def capture_graph(gm, inputs):
        graphs.append(gm)
        return gm

    print("\nTracing MLP...")

    compiled_mlp = torch.compile(mlp, backend=capture_graph, fullgraph=False)

    with torch.no_grad():
        compiled_mlp(x)

    if graphs:
        print(f"\nCaptured {len(graphs)} graph(s):")
        graphs[0].graph.print_tabular()
    else:
        print("\n⚠️ No graphs captured (Dynamo fallback).")


if __name__ == "__main__":
    main()
