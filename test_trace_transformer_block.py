import torch
import torch._dynamo

from vllm.config import (
    ModelConfig,
    DeviceConfig,
    LoadConfig,
    ParallelConfig,
)
from vllm.model_executor.model_loader import get_model_loader


# Cleaner tracing (avoid crashing on graph breaks)
torch._dynamo.config.suppress_errors = True


def main():
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    device = "cuda"

    print(f"Loading model: {model_name}")

    # -------------------------
    # 1. Setup configs
    # -------------------------
    model_config = ModelConfig(
        model=model_name,
        task="generate",
        tokenizer=model_name,
        tokenizer_mode="auto",
        trust_remote_code=True,
        dtype=torch.float16,
        seed=42,
    )

    # Needed internally by vLLM
    parallel_config = ParallelConfig(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        enable_async_output=False,
    )

    load_config = LoadConfig()

    # NOTE: we DO NOT pass DeviceConfig to load_model (older versions break)
    # device is handled internally by vLLM


    # -------------------------
    # 2. Load model
    # -------------------------
    loader = get_model_loader(load_config)

    try:
        # Most versions
        model = loader.load_model(
            model_config=model_config,
            parallel_config=parallel_config,
        )
    except TypeError:
        # Some versions want these explicitly
        model = loader.load_model(
            model_config=model_config,
            parallel_config=parallel_config,
            vision_language_config=None,
            lora_config=None,
        )

    print(f"Model loaded: {type(model)}")


    # -------------------------
    # 3. Unwrap model safely
    # -------------------------
    # vLLM wraps the HF model differently by version

    if hasattr(model, "model"):
        hf_model = model.model
    elif hasattr(model, "layers"):
        hf_model = model
    else:
        # Deep wrapper path (older vLLM)
        hf_model = (
            model.llm_engine
                 .model_executor
                 .driver_worker
                 .model
                 .model
        )

    print(f"HF model type: {type(hf_model)}")


    # -------------------------
    # 4. Grab transformer MLP
    # -------------------------
    layer0 = hf_model.layers[0]
    mlp = layer0.mlp

    print("\nMLP module:")
    print(mlp)


    # -------------------------
    # 5. Prepare input
    # -------------------------
    hidden_size = hf_model.config.hidden_size

    x = torch.randn(
        1,
        hidden_size,
        device=device,
        dtype=torch.float16,
    )


    # -------------------------
    # 6. Capture TorchDynamo graph
    # -------------------------
    graphs = []

    def capture_graph(gm, inputs):
        graphs.append(gm)
        return gm

    print("\nTracing MLP...")

    compiled_mlp = torch.compile(
        mlp,
        backend=capture_graph,
        fullgraph=False,
    )

    with torch.no_grad():
        compiled_mlp(x)

    if graphs:
        print(f"\nCaptured {len(graphs)} graph(s):\n")
        graphs[0].graph.print_tabular()
    else:
        print("\n⚠️ No graph captured (Dynamo fallback).")


if __name__ == "__main__":
    main()
