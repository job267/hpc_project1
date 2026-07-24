import csv
import math
import statistics
import time
from datetime import datetime
from pathlib import Path 

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
DEVICE = "cuda"

# 固定实验参数
PROMPT_TOKENS = 128
NEW_TOKENS = 64
WARMUP_RUNS = 5
TEST_RUNS = 20

def build_fixed_input(tokenizer, prompt_tokens):
    """
    构造长度严格等于prompt_tokens 的输入，性能基准主要关心张量
    形状和序列长度，不关心模型回答，因此使用重复文本构造固定长度输入
    """
    seed_text = (
        "高性能计算研究如何利用并行处理器，gpu和高效算法，"
        "提高计算任务的运行速度和资源利用率"
    )

    token_ids = tokenizer(
        seed_text,
        add_special_tokens = False,
        return_tensors = "pt",
    )["input_ids"]

    original_length = token_ids.shape[1]

    # 计算需要重复多少次才能达到目标长度
    repeat_times = math.ceil(prompt_tokens / original_length)

    # 在序列维度上重读，然后截取目标长度
    input_ids = token_ids.repeat(1, repeat_times)
    input_ids = input_ids[:,:prompt_tokens]
    input_ids = input_ids.to(DEVICE)

    attention_mask = torch.ones_like(input_ids)

    return input_ids, attention_mask

def percentile_95(values):
    # 计算p95，避免安装numpy
    sorted_values = sorted(values)
    index = math.ceil(0.95 * len(sorted_values)) - 1
    return sorted_values[index]

def main():
    # 检查cuda环境
    if not torch.cuda.is_available():
        raise RuntimeError(
            "pytorch无法使用cuda,请检查驱动"
        )
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    gpu_name = torch.cuda.get_device_name(0)

    print("=== 实验环境 ===")
    print("GPU:", gpu_name)
    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA:", torch.version.cuda)
    print("模型:", MODEL_ID)

    # 加载分词器
    print("\n正在加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # 加载FP16 eager模型
    print("正在加载FP16模型...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype = torch.float16,
        attn_implementation = "eager",
        low_cpu_mem_usage = True,
    )

    model = model.to(DEVICE)
    model.eval()

    model_dtype = next(model.parameters()).dtype
    attention_backend = getattr(
        model.config,
        "_attn_implementation",
        "unknown",
    )

    print("模型精度:",model_dtype)
    print("attention实现：", attention_backend)

    if model_dtype != torch.float16:
        raise RuntimeError(
            f"模型精度不是fp16,而是{model_dtype}"
        )

    # 构造固定长度输入
    input_ids, attention_mask = build_fixed_input(
        tokenizer,
        PROMPT_TOKENS,
    )

    actual_prompt_tokens = input_ids.shape[1]

    print("\n=== 基准参数 ===")
    print("Batch size:", input_ids.shape[0])
    print("输入Token数:", actual_prompt_tokens)
    print("输出Token数:", NEW_TOKENS)
    print("预热次数:", WARMUP_RUNS)
    print("正式测试次数:", TEST_RUNS)

    # 所有实验保持完全相同的生成参数
    generation_config = {
        "max_new_tokens" : NEW_TOKENS,
        "min_new_tokens" : NEW_TOKENS,
        "do_sample" : False,
        "use_cache" : True,
        "pad_token_id" : tokenizer.eos_token_id,
    }

    # 预热
    print("\n正在预热GPU")

    with torch.inference_mode():
        for index in range(WARMUP_RUNS):
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_config,
            )

            print(
                f"预热{index + 1} / {WARMUP_RUNS}",
                end = "\r",
            )
    
    # 等待全部cuda任务执行完成
    torch.cuda.synchronize()

    # 删除最后一次预热输出的引用
    del outputs

    print("\n预热完成")

    # 记录模型加载后的基础显存
    model_memory_gib = (
        torch.cuda.memory_allocated() / 1024**3
    )

    # 从这里重新统计峰值显存
    torch.cuda.reset_peak_memory_stats()

    # 正式测试
    latencies = []
    throughputs = []

    print("\n=== 开始正式测试 ===")
    
    with torch.inference_mode():
        for index in range(TEST_RUNS):
            # 计时前等待此前CUDA任务完成
            torch.cuda.synchronize()
            start_time = time.perf_counter()

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_config,
            )

            # generate会异步启动cuda任务，必须同步后才能停止计时
            torch.cuda.synchronize()
            end_time = time.perf_counter()

            elapsed_seconds = end_time - start_time

            output_length = outputs.shape[1]
            generated_tokens = (
                output_length - actual_prompt_tokens
            )

            if generated_tokens != NEW_TOKENS:
                raise RuntimeError(
                    f"预期生成{NEW_TOKENS}个TOKEN,"
                    f"实际生成{generated_tokens}个TOKEN"
                )

            tokens_per_second = (
                generated_tokens / elapsed_seconds
            )

            latencies.append(elapsed_seconds)
            throughputs.append(tokens_per_second)

            print(
                f"第{index + 1:02d}/{TEST_RUNS}次："
                f"延迟={elapsed_seconds * 1000:.2f} ms，"
                f"速度={tokens_per_second:.2f} tokens/s"
            )

            del outputs
    # 7. 统计结果
    average_latency = statistics.fmean(latencies)
    median_latency = statistics.median(latencies)
    p95_latency = percentile_95(latencies)

    average_throughput = statistics.fmean(throughputs)
    median_throughput = statistics.median(throughputs)

    peak_memory_gib = (
        torch.cuda.max_memory_allocated() / 1024**3
    )

    print("\n=== FP16 eager基线结果 ===")
    print(
        f"平均生成延迟：{average_latency * 1000:.2f} ms"
    )
    print(
        f"中位数生成延迟：{median_latency * 1000:.2f} ms"
    )
    print(
        f"P95生成延迟：{p95_latency * 1000:.2f} ms"
    )
    print(
        f"平均生成速度：{average_throughput:.2f} tokens/s"
    )
    print(
        f"中位数生成速度：{median_throughput:.2f} tokens/s"
    )
    print(
        f"模型加载后显存：{model_memory_gib:.3f} GiB"
    )
    print(
        f"推理峰值显存：{peak_memory_gib:.3f} GiB"
    )

    # 8. 保存CSV结果
    result_file = Path("benchmark_results.csv")

    result = {
        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),
        "model": MODEL_ID,
        "gpu": gpu_name,
        "pytorch": torch.__version__,
        "pytorch_cuda": torch.version.cuda,
        "dtype": str(model_dtype),
        "attention": attention_backend,
        "batch_size": input_ids.shape[0],
        "prompt_tokens": actual_prompt_tokens,
        "new_tokens": NEW_TOKENS,
        "warmup_runs": WARMUP_RUNS,
        "test_runs": TEST_RUNS,
        "average_latency_ms": average_latency * 1000,
        "median_latency_ms": median_latency * 1000,
        "p95_latency_ms": p95_latency * 1000,
        "average_tokens_per_second": average_throughput,
        "median_tokens_per_second": median_throughput,
        "model_memory_gib": model_memory_gib,
        "peak_memory_gib": peak_memory_gib,
    }

    file_exists = result_file.exists()

    with result_file.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=result.keys(),
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(result)

    print(f"\n结果已保存到：{result_file.resolve()}")


if __name__ == "__main__":
    main()

    