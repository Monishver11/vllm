import torch
import torch._dynamo

from vllm.engine.llm_engine import LLMEngine
from vllm.engine.arg_utils import EngineArgs

torch._dynamo.config.suppress_errors = True


def main():
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    device = "cuda"

    print(f"Loading model: {model_name}")

    # --------------------------------
    # 1. Build engine args (IMPORTANT)
    # --------------------------------
    engine_args = EngineArgs(
        model=model_name,
        tokenizer=model_name,
        trust_remote_code=True,
        dtype="float16",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        device=device,
        seed=42,
    )

    # --------------------------------
    # 2. Create engine (initializes parallel state)
    # --------------------------------
    engine = LLMEngine.from_engine_args(engine_args)

    # --------------------------------
    # 3. Get underlying HF model
    # --------------------------------
    # This is where vLLM stores the actual nn.Module
    worker = engine.model_executor.driver_worker
    model = worker.model.model   # <- HF Qwen2Model

    print("HF model type:", type(model))


    # --------------------------------
    # 4. Access first transformer block MLP
    # --------------------------------
    layer0 = model.layers[0]
    mlp = layer0.mlp

    print("\nMLP:\n", mlp)


    # --------------------------------
    # 5. TorchDynamo trace
    # --------------------------------
    hidden_size = model.config.hidden_size

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
        print(f"\nCaptured {len(graphs)} graph(s):\n")
        graphs[0].graph.print_tabular()
    else:
        print("\n⚠️ No graph captured (fallback to eager).")


if __name__ == "__main__":
    main()
