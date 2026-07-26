"""

本文件将推理拆分，将推理过程拆分为如下三项：
- Prefill：处理完整 Prompt，产生第一个 token 和 KV Cache。
- Decode：每次只输入一个新 token，并复用 KV Cache。
- 分别统计 TTFT、Prefill 吞吐、Decode 吞吐和 TPOT

"""

import csv
import math
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
DEVICE = "cuda"

PROMPT_TOKENS = 128
NEW_TOKENS = 64

WARMUP_RUNS = 5
TEST_RUNS = 20

RESULT_FILE = Path("stage_benchmark_result.csv")

def build_fixed_input(tokenizer, target_length):
    # 构造长度固定的输入，避免文本长度变化影响基准

    seed_text = (
        "高性能计算通过并行处理，硬件加速和算法优化"
        "提高大规模计算任务的执行效率"
    )

    encoded = tokenizer(
        seed_text,
        return_tensors = "pt",
        add_special_tokens = False,
    )["input_ids"]

    repeat_count = math.ceil(target_length / encoded.shape[1])

    input_ids = encoded.repeat(1, repeat_count)
    input_ids = input_ids[:, :target_length]

    return input_ids.to(DEVICE)

@torch.inference_mode()
def benchmark_once(model, input_ids, full_attention_mask, new_tokens):
    """
    执行一次阶段化推理。

    NEW_TOKENS 包括：
    - Prefill 产生的第一个 token
    - Decode 产生的剩余 token

    """
    if new_tokens < 2:
        raise ValueError("NEW_TOKENS 至少应为2")
    
    batch_size, prompt_length = input_ids.shape

    # prefill

    prefill_attention_mask = full_attention_mask[:,:prompt_length]

    # prompt tokens 的绝对位置：0，1，2，...，prompt_length-1
    cache_position = torch.arange(
        prompt_length,
        dtype = torch.long,
        device = input_ids.device
    )

    torch.cuda.synchronize()
    prefill_start = time.perf_counter()

    outputs = model(
        input_ids = input_ids,
        attention_mask = prefill_attention_mask,
        use_cache = True,
        cache_position = cache_position,

        # 生成阶段只需要最后一个位置的 logits
        logits_to_keep = 1,
        return_dict = True,
    )

    # prefill最后一个位置预测第一个输出token
    next_token = torch.argmax(
        outputs.logits[:, -1, :],
        dim = -1,
        keepdim = True,
    )

    past_key_values = outputs.past_key_values

    torch.cuda.synchronize()
    ttft_seconds = time.perf_counter() - prefill_start

    generated_tokens = [next_token]

    # decode

    # 第一个生成 token的绝对位置就是prompt_length
    cache_position = torch.tensor(
        [prompt_length],
        dtype = torch.long,
        device = input_ids.device,
    )

    # 第一个token已经由 prefill产生，因此只需要在生成 N-1 个
    decode_steps = new_tokens - 1

    decode_start = time.perf_counter()

    for step in range(decode_steps):
        # 当前传入的token会被加入KV cache
        current_sequence_length = prompt_length + step + 1

        outputs = model(
            # decode 每一步只输入一个新token
            input_ids = next_token,

            #attention_mask 覆盖 prompt + 已输入的生成 token
            attention_mask = full_attention_mask[
                :,:current_sequence_length
            ],
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

        past_key_values = outputs.past_key_values
        generated_tokens.append(next_token)

        # 下一轮token的绝对位置+1
        cache_position = cache_position + 1
    
    # 不要再循环体内部 synchronize, 否则会认为放大 token 开销
    torch.cuda.synchronize()
    decode_seconds = time.perf_counter() - decode_start

    generated_ids = torch.cat(generated_tokens, dim=1)

    total_seconds = ttft_seconds + decode_seconds

    metrics = {
        "ttft_ms": ttft_seconds * 1000,
        "prefill_tokens_per_second": prompt_length / ttft_seconds,
        "decode_ms": decode_seconds * 1000,
        "decode_tokens_per_second": decode_steps / decode_seconds,
        "tpot_ms": decode_seconds / decode_steps * 1000,
        "end_to_end_ms": total_seconds * 1000,
        "output_tokens_per_second": new_tokens / total_seconds,
    }

    return metrics, generated_ids

def percentile_95(values):
    calues = sorted(values)
    index = math.ceil(len(values) * 0.95) - 1
    return values[index]

def print_summary(results):
    print("\n========== 阶段基准汇总 ==========")

    latency_metrics = [
        ("ttft_ms", "TTFT"),
        ("decode_ms", "Decode 总时间"),
        ("tpot_ms", "TPOT"),
        ("end_to_end_ms", "端到端时间"),
    ]

    for key, name in latency_metrics:
        values = [result[key] for result in results]

        print(
            f"{name:<14}"
            f"平均 {statistics.mean(values):8.3f} ms | "
            f"中位数 {statistics.median(values):8.3f} ms | "
            f"P95 {percentile_95(values):8.3f} ms"
        )

    throughput_metrics = [
        ("prefill_tokens_per_second", "Prefill 吞吐"),
        ("decode_tokens_per_second", "Decode 吞吐"),
        ("output_tokens_per_second", "整体输出吞吐"),
    ]

    print()

    for key, name in throughput_metrics:
        values = [result[key] for result in results]

        print(
            f"{name:<14}"
            f"{statistics.mean(values):8.2f} token/s"
        )

    peak_memory = max(result["peak_vram_gib"] for result in results)
    print(f"\n峰值显存：{peak_memory:.3f} GiB")


def save_results(results):
    fieldnames = [
        "run",
        "model",
        "attention",
        "dtype",
        "prompt_tokens",
        "new_tokens",
        "ttft_ms",
        "prefill_tokens_per_second",
        "decode_ms",
        "decode_tokens_per_second",
        "tpot_ms",
        "end_to_end_ms",
        "output_tokens_per_second",
        "peak_vram_gib",
    ]

    with RESULT_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for run_index, result in enumerate(results, start=1):
            writer.writerow(
                {
                    "run": run_index,
                    "model": MODEL_ID,
                    "attention": "eager",
                    "dtype": "float16",
                    "prompt_tokens": PROMPT_TOKENS,
                    "new_tokens": NEW_TOKENS,
                    **result,
                }
            )

    print(f"结果已保存到：{RESULT_FILE}")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("当前 PyTorch 无法使用 CUDA")

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch CUDA:", torch.version.cuda)
    print("模型:", MODEL_ID)
    print("输入 token:", PROMPT_TOKENS)
    print("输出 token:", NEW_TOKENS)

    print("\n正在加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("正在加载 FP16 eager 模型...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )

    model = model.to(DEVICE)
    model.eval()

    input_ids = build_fixed_input(
        tokenizer,
        PROMPT_TOKENS,
    )

    # 最长需要覆盖 Prompt 和整个生成阶段
    full_attention_mask = torch.ones(
        (
            input_ids.shape[0],
            PROMPT_TOKENS + NEW_TOKENS,
        ),
        dtype=torch.long,
        device=DEVICE,
    )

    print("实际输入长度:", input_ids.shape[1])

    # Warmup 

    print(f"\n开始预热，共 {WARMUP_RUNS} 次...")

    for _ in range(WARMUP_RUNS):
        _, warmup_ids = benchmark_once(
            model,
            input_ids,
            full_attention_mask,
            NEW_TOKENS,
        )
        del warmup_ids

    torch.cuda.synchronize()

    # Benchmark 

    print(f"开始正式测试，共 {TEST_RUNS} 次...\n")

    results = []
    sample_ids = None

    for run_index in range(TEST_RUNS):
        if sample_ids is not None:
            del sample_ids

        torch.cuda.reset_peak_memory_stats()

        metrics, sample_ids = benchmark_once(
            model,
            input_ids,
            full_attention_mask,
            NEW_TOKENS,
        )

        metrics["peak_vram_gib"] = (
            torch.cuda.max_memory_allocated() / 1024**3
        )

        results.append(metrics)

        print(
            f"[{run_index + 1:02d}/{TEST_RUNS}] "
            f"TTFT={metrics['ttft_ms']:.2f} ms | "
            f"Decode={metrics['decode_tokens_per_second']:.2f} token/s | "
            f"TPOT={metrics['tpot_ms']:.2f} ms | "
            f"E2E={metrics['end_to_end_ms']:.2f} ms"
        )

    assert sample_ids.shape[1] == NEW_TOKENS

    print_summary(results)
    save_results(results)

    print("\n最后一次生成结果：")
    print(
        tokenizer.decode(
            sample_ids[0],
            skip_special_tokens=True,
        )
    )


if __name__ == "__main__":
    main()
