# test_trace_transformer_block.py

import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx import symbolic_trace

# Import vLLM components
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8


class SimpleMLP(nn.Module):
    """Simplified MLP that mimics DeepSeek's structure"""
    def __init__(self, hidden_size: int, intermediate_size: int, group_size: int = 128):
        super().__init__()
        # gate_up_proj: projects to 2x intermediate for gate and up
        self.gate_up_proj = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.silu_and_mul = SiluAndMul()
        self.group_size = group_size
        
        # FP8 constants
        self.fp8_dtype = torch.float8_e4m3fn
        finfo = torch.finfo(self.fp8_dtype)
        self.fp8_min = finfo.min
        self.fp8_max = finfo.max
        
    def forward(self, x):
        # Gate and up projection (fused)
        gate_up = self.gate_up_proj(x)
        
        # SiLU activation and multiply
        y = self.silu_and_mul(gate_up)
        
        # Block quantization
        output_q = torch.empty(y.shape, device=y.device, dtype=self.fp8_dtype)
        scale_shape = (y.shape[0], y.shape[1] // self.group_size)
        output_s = torch.empty(scale_shape, device=y.device, dtype=torch.float32)
        
        torch.ops._C.per_token_group_fp8_quant(
            y, output_q, output_s,
            self.group_size,
            1e-10,
            self.fp8_min,
            self.fp8_max,
            False  # scale_ue8m0
        )
        
        # For now, just return quantized output (skip down_proj for simplicity)
        return output_q, output_s


class SimpleMLPWithWrapper(nn.Module):
    """MLP using the per_token_group_quant_fp8 wrapper function"""
    def __init__(self, hidden_size: int, intermediate_size: int, group_size: int = 128):
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
        self.silu_and_mul = SiluAndMul()
        self.group_size = group_size
        
    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        y = self.silu_and_mul(gate_up)
        
        # Use the wrapper function (might cause graph breaks)
        result, scale = per_token_group_quant_fp8(
            y,
            group_size=self.group_size,
            column_major_scales=False,
        )
        return result, scale


def trace_with_dynamo(model, example_input, name="model"):
    """Trace model using torch.compile and print the graph"""
    print(f"\n{'='*60}")
    print(f"Tracing: {name}")
    print(f"{'='*60}")
    
    graphs = []
    
    def capture_graph(gm, inputs):
        graphs.append(gm)
        return gm
    
    try:
        compiled = torch.compile(model, backend=capture_graph)
        
        # Mark dynamic to avoid recompilation
        torch._dynamo.mark_dynamic(example_input, 0)
        
        # Run once to trigger compilation
        with torch.no_grad():
            output = compiled(example_input)
        
        print(f"\nCaptured {len(graphs)} graph(s)")
        
        for i, gm in enumerate(graphs):
            print(f"\n--- Graph {i+1} ---")
            gm.graph.print_tabular()
            
            print(f"\n--- Ops in Graph {i+1} ---")
            for node in gm.graph.nodes:
                if node.op == "call_function":
                    print(f"  Op: {node.target}")
                    if node.kwargs:
                        print(f"      kwargs: {node.kwargs}")
                        
    except Exception as e:
        import traceback
        print(f"Tracing failed: {e}")
        traceback.print_exc()
    
    return graphs


def main():
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)
    
    # Model dimensions (small for testing)
    hidden_size = 256
    intermediate_size = 512  # Must be divisible by group_size
    group_size = 128
    batch_size = 4
    
    # Example input
    x = torch.randn(batch_size, hidden_size)
    
    # Test 1: Direct custom op calls
    print("\n" + "="*60)
    print("TEST 1: SimpleMLP with direct custom op calls")
    print("="*60)
    
    model1 = SimpleMLP(hidden_size, intermediate_size, group_size)
    model1 = model1.cuda().bfloat16()
    
    # First run without compile to verify it works
    with torch.no_grad():
        out1 = model1(x)
        print(f"Output shapes: {out1[0].shape}, {out1[1].shape}")
    
    # Trace
    trace_with_dynamo(model1, x, "SimpleMLP (direct ops)")
    
    # Test 2: Using wrapper function (might have graph breaks)
    print("\n" + "="*60)
    print("TEST 2: SimpleMLPWithWrapper")
    print("="*60)
    
    model2 = SimpleMLPWithWrapper(hidden_size, intermediate_size, group_size)
    model2 = model2.cuda().bfloat16()
    
    with torch.no_grad():
        out2 = model2(x)
        print(f"Output shapes: {out2[0].shape}, {out2[1].shape}")
    
    trace_with_dynamo(model2, x, "SimpleMLPWithWrapper")
    
    # Test 3: Just silu_and_mul + quant (no linear)
    print("\n" + "="*60)
    print("TEST 3: Just SiluAndMul + Quant (no Linear)")
    print("="*60)
    
    class JustSiluMulQuant(nn.Module):
        def __init__(self, group_size=128):
            super().__init__()
            self.silu_and_mul = SiluAndMul()
            self.group_size = group_size
            self.fp8_dtype = torch.float8_e4m3fn
            finfo = torch.finfo(self.fp8_dtype)
            self.fp8_min = finfo.min
            self.fp8_max = finfo.max
            
        def forward(self, x):
            y = self.silu_and_mul(x)
            output_q = torch.empty(y.shape, device=y.device, dtype=self.fp8_dtype)
            scale_shape = (y.shape[0], y.shape[1] // self.group_size)
            output_s = torch.empty(scale_shape, device=y.device, dtype=torch.float32)
            
            torch.ops._C.per_token_group_fp8_quant(
                y, output_q, output_s,
                self.group_size, 1e-10, self.fp8_min, self.fp8_max, False
            )
            return output_q, output_s
    
    model3 = JustSiluMulQuant(group_size)
    model3 = model3.cuda().bfloat16()
    
    # Input for silu_and_mul should be 2x the output size
    x3 = torch.randn(batch_size, intermediate_size * 2)
    
    with torch.no_grad():
        out3 = model3(x3)
        print(f"Output shapes: {out3[0].shape}, {out3[1].shape}")
    
    trace_with_dynamo(model3, x3, "JustSiluMulQuant")


if __name__ == "__main__":
    main()