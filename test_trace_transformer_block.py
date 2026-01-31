import torch
import torch._dynamo

from vllm.config import (
    ModelConfig,
    LoadConfig,
)
from vllm.model_executor.model_loader import get_model_loader


torch._dynamo.config.suppress_errors = True


def main():
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    device = "cuda"

    print(f"Loading model: {model_name}")

    # -------------------------
    # 1. Model config
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

    load_config = LoadConfig()

    # -------------------------
    # 2. Load model (OLD API)
    # -------------------------
    loader = get_model_loader(load_config)

    try:
        model = loader.load_model(model_config)
    except TypeError:
        model = loader.load_model(load_config, model_config)



    # -------------------------
    # 3. Unwrap HF model
    # -------------------------
    if hasattr(model, "model"):
        hf_model = model.model
    else:
        hf_model = (
            model.llm_engine
                 .model_executor
                 .driver_worker
                 .model
                 .model
        )

    print(f"HF model type: {type(hf_model)}")


    # -------------------------
    # 4. Access first block MLP
    # -------------------------
    layer0 = hf_model.layers[0]
    mlp = layer0.mlp

    print("\nMLP module:\n")
    print(mlp)


    # -------------------------
    # 5. Trace with Dynamo
    # -------------------------
    hidden_size = hf_model.config.hidden_size

    x = torch.randn(
        1,
        hidden_size,
        device=device,
        dtype=torch.float16,
    )

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
        print(f"\nCaptured {len(graphs)} graph(s)\n")
        graphs[0].graph.print_tabular()
    else:
        print("\n⚠️ No graph captured (fallback to eager)")


if __name__ == "__main__":
    main()
