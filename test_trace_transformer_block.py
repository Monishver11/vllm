import torch
import torch._dynamo

from vllm.engine.llm_engine import LLMEngine
from vllm.engine.arg_utils import EngineArgs

torch._dynamo.config.suppress_errors = True


def main():
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    print(f"Loading model: {model_name}")

    # --------------------------------
    # 1. Engine args (minimal + correct)
    # --------------------------------
    engine_args = EngineArgs(
        model=model_name,
        tokenizer=model_name,
        trust_remote_code=True,
        dtype="float16",

        # no parallelism
        tensor_parallel_size=1,
        pipeline_parallel_size=1,

        seed=42,
    )

    # --------------------------------
    # 2. Create engine
    # --------------------------------
    engine = LLMEngine.from_engine_args(engine_args)

    # --------------------------------
    # 3. Get HF model
    # --------------------------------
    worker = engine.model_executor.driver_worker
    model = worker.model.model   # Qwen2Model

    print("HF model:", type(model))


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
        print("\nCaptured FX graph:\n")
        graphs[0].graph.print_tabular()
    else:
        print("\n⚠️ No graph captured")


if __name__ == "__main__":
    main()
