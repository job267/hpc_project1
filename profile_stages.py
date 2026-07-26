"""
    定位prefill和decode的热点算子
"""

import math
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
DEVICE = "cuda"
ATTN_IMPLEMENTATION = "eager"

PROMPT_TOKENS = 128

WARMUP_RUNS = 3
WARMUP_DECODE_STEPS = 8

PREFILL_PROFILE_RUNS = 3
DECODE_PROFILE_STEPS = 8

PROFILE_DIR = Path("profiles")

def build_fixed_input(tokenizer, target_length):
    seed_text = (
        "高性能计算通过并行计算，硬件加速和算法优化"
        "提高大规模计算任务的执行效率"
    )

    input_ids = tokenizer(
        seed_text,
        return_tensors = "pt",
        add_special_tokens = False,
    )["input_ids"]

    repeat_count = math.ceil(target_length / input_ids.shape[1])

    input_ids = input_ids.repeat(1, repeat_count)
    input_ids = input_ids[:, :target_length]

    return input_ids.to(DEVICE)

@torch.inference_mode()
def run_prefill(
    model,
    input_ids,
    attention_mask,
    cache_position,
):
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        cache_position=cache_position,
        logits_to_keep=1,
        return_dict=True,
    )

    next_token = torch.argmax(
        outputs.logits[:, -1, :],
        dim=-1,
        keepdim=True,
    )

    return next_token, outputs.past_key_values

@torch.inference_mode()
def run_decode_step(
    model,
    next_token,
    attention_mask,past_key_values,
    cache_position,
):
    outputs = model(
        input_ids= next_token,
        attention_mask= attention_mask,
        past_key_values = past_key_values,
        use_cache = True,
        cache_position = cache_position,
        logits_to_keep = 1,
        return_dict = True,
    )

    next_token = torch.argmax(
        outputs.logits[:, -1, :],
        dim = -1,
        keepdim = True,
    )

    return next_token, outputs.past_key_values

def warmup(
    model,
    input_ids,
    full_attention_mask,
    prompt_positions,
    decode_positions,
):
    print("开始预热...")

    for _ in range(WARMUP_RUNS):
        next_token, past_key_values = run_prefill(
            model = model,
            input_ids = input_ids,
            attention_mask = full_attention_mask[:,:PROMPT_TOKENS],
            cache_position = prompt_positions,
        )

        for step in range(WARMUP_DECODE_STEPS):
            current_length = PROMPT_TOKENS + step + 1

            next_token, past_key_values = run_decode_step(
                model = model,
                next_token = next_token,
                attention_mask = full_attention_mask[:,:current_length],
                past_key_values = past_key_values,
                cache_position = decode_positions[step : step + 1],
            )

        del next_token
        del past_key_values

    torch.cuda.synchronize()
    print("预热完成")

def create_profiler():
    return profile(
        activities = [
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        record_shapes = True,
        profile_memory = True,
        with_stack = True,
        with_flops = True,
    )

def profile_prefill(
    model,
    input_ids,
    attention_mask,
    prompt_positions,
):
    print("\n开始分析 prefill...")

    torch.cuda.synchronize()

    with create_profiler() as profiler:
        for _ in range(PREFILL_PROFILE_RUNS):
            with record_function("prefill"):
                _, past_key_values = run_prefill(
                    model = model,
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    cache_position = prompt_positions,
                )
            
            del past_key_values
        
        # 确保 cuda kernel 全部执行完毕后再结束记录
        torch.cuda.synchronize()

    return profiler

def profile_decode(
    model,
    input_ids,
    full_attention_mask,
    prompt_positions,
    decode_positions,
):
    print("\n开始分析 decode...")

    # prefill 放在 profiler 外面，防止污染decode数据
    next_token, past_key_values = run_prefill(
        model = model,
        input_ids = input_ids,
        attention_mask = full_attention_mask[:,:PROMPT_TOKENS],
        cache_position = prompt_positions,
    )

    torch.cuda.synchronize()

    with create_profiler() as profiler:
        for step in range(DECODE_PROFILE_STEPS):
            current_length = PROMPT_TOKENS + step + 1
            current_attention_mask = full_attention_mask[:, :current_length]
            current_cache_position = decode_positions[step : step + 1]

            with record_function("decode_step"):
                next_token, past_key_values = run_decode_step(
                    model = model,
                    next_token = next_token,
                    attention_mask = current_attention_mask,
                    past_key_values = past_key_values,
                    cache_position = current_cache_position,
                )
        torch.cuda.synchronize()
    return profiler
def export_profile(profiler, stage_name):
    cuda_table = profiler.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=20,
    )

    cpu_table = profiler.key_averages().table(
        sort_by="self_cpu_time_total",
        row_limit=20,
    )

    report = (
        f"========== {stage_name.upper()} CUDA 热点 ==========\n"
        f"{cuda_table}\n\n"
        f"========== {stage_name.upper()} CPU 热点 ==========\n"
        f"{cpu_table}\n"
    )

    print("\n" + report)

    report_path = PROFILE_DIR / f"{stage_name}_ops.txt"
    trace_path = PROFILE_DIR / f"{stage_name}_trace.json"

    report_path.write_text(report, encoding="utf-8")
    profiler.export_chrome_trace(str(trace_path))

    print(f"算子报告：{report_path}")
    print(f"时间线文件：{trace_path}")

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("当前pytorch无法使用cuda")
    
    PROFILE_DIR.mkdir(parents = True, exist_ok = True)

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA:", torch.version.cuda)
    print("Attention:", ATTN_IMPLEMENTATION)

    print("\n正在加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("正在加载 FP16 eager 模型...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype = torch.float16,
        low_cpu_mem_usage = True,
        attn_implementation = ATTN_IMPLEMENTATION,
    )

    model = model.to(DEVICE)
    model.eval()

    input_ids = build_fixed_input(
        tokenizer,
        PROMPT_TOKENS,
    )

    maximum_decode_steps = max(
        WARMUP_DECODE_STEPS,
        DECODE_PROFILE_STEPS,
    )

    full_attention_mask = torch.ones(
        (
        input_ids.shape[0],
        PROMPT_TOKENS + maximum_decode_steps,
        ),
        dtype = torch.long,
        device = DEVICE,
    )

    prompt_positions = torch.arange(
        PROMPT_TOKENS,
        dtype = torch.long,
        device = DEVICE,
    )

    decode_positions = torch.arange(
        PROMPT_TOKENS,
        PROMPT_TOKENS + maximum_decode_steps,
        dtype=torch.long,
        device=DEVICE,
    )

    print("实际输入长度:", input_ids.shape[1])

    warmup(
        model=model,
        input_ids=input_ids,
        full_attention_mask=full_attention_mask,
        prompt_positions=prompt_positions,
        decode_positions=decode_positions,
    )

    prefill_profiler = profile_prefill(
        model=model,
        input_ids=input_ids,
        attention_mask=full_attention_mask[:, :PROMPT_TOKENS],
        prompt_positions=prompt_positions,
    )

    export_profile(
        profiler=prefill_profiler,
        stage_name="prefill",
    )

    del prefill_profiler

    decode_profiler = profile_decode(
        model=model,
        input_ids=input_ids,
        full_attention_mask=full_attention_mask,
        prompt_positions=prompt_positions,
        decode_positions=decode_positions,
    )

    export_profile(
        profiler=decode_profiler,
        stage_name="decode",
    )

    print("\n性能剖析完成")
    print("请重点检查 profiles/prefill_ops.txt")
    print("以及 profiles/decode_ops.txt")


if __name__ == "__main__":
    main()


