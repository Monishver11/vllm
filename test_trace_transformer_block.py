import torch
import torch._dynamo

from vllm.engine.llm_engine import LLMEngine
from vllm.engine.arg_utils import EngineArgs

torch._dynamo.config.suppress_errors = True


def main():
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    print(f"Loading model: {model_name}")

    # --------------------------------
    # 1. Engine args (NO device param)
    # --------------------------------
    engine_args = EngineArgs(
        model=model_name,
        tokenizer=model_name,
        trust_remote_code=True,
        dtype="float16",

        # turn off parallelism
        tensor_parallel_size=1,
        pipeline_parallel_size=1,

        # turn off distributed infra
        worker_use_ray=False,
        disable_custom_all_reduce=True,

        seed=42,
    )

    # --------------------------------
    # 2. Build engine (sets everything up)
    # --------------------------------
    engine = LLMEngine.from_engine_args(engine_args)

    # --------------------------------
    # 3. Get raw HF model
    # --------------------------------
    worker = engine.model_executor.driver_worker
    model = worker.model.model   # Qwen2Model

    print("HF model type:", type(model))


    # --------------------------------
    # 4. Grab MLP
    # --------------------------------
    mlp = model.layers[0].mlp

    print("\nMLP:\n", mlp)


    # --------------------------------
    # 5. Trace
    # --------------------------------
    hidden_size = model.config.hidden_size

    x = torch.randn(
        1,
        hidden_size,
        device="cuda",
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
        print("\nCaptured graph:\n")
        graphs[0].graph.print_tabular()
    else:
        print("\n⚠️ No graph captured (fallback eager)")


if __name__ == "__main__":
    main()
