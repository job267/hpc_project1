import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import CompileConfig
except ImportError:
    CompileConfig = None

MODEL_ID = "Qwen/Qwen3-0.6B"
DEVICE = "cuda"

def parse_args():
    parser = argparse.ArgumentParser(
        description = "Qwen3 fp16 推理优化变量对照实验"
    )

    parser.add_argument(
        "--attention",
        choices = ["eager", "sdpa"],
        default = "eager",
    )

    parser.add_argument(
        "--cache",
        choices = ["dynamic", "static"],
        default = "dynamic",
    )

    parser.add_argument(
        "--compile-mode",
        choices = ["none", "reduce-overhead"],
        default = "none",
    )

    parser.add_argument(
        "--prompt-tokens",
        type = int,
        default = 64,
    )

    parser.add_argument(
        "--new-tokens",
        type = int,
        default = 64,
    )

    parser.add_argument(
        "--warmup-runs",
        type = int,
        default = 5,
    )

    parser.add_argument(
        "--test-runs",
        type = int,
        default = 20,
    )

    parser.add_argument(
        "--result-dir",
        type = Path,
        default = Path("results"),
    )

    args = parser.parse_args()
    if args.compile_mode != "none" and args.cache != "static":
        parser.error("当前实验只允许stasuc cache and torch.compile")
    if args.prompt_tokens <= 0:
        parser.error("--prompt-tokens 必须大于0")
    if args.new_tokens <= 0:
        parser.error("--new-tokens 必须大于0")

    return args

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

    repeat_count = math.ceil(
        target_length / input_ids.shape[1]
    )

    input_ids = input_ids.repeat(1, repeat_count)
    input_ids = input_ids[:, :target_length]

    attention_mask = torch.ones_like(input_ids)

    return (
        input_ids.to(DEVICE),
        attention_mask.to(DEVICE)
    )

def build_generation_kwargs(
    args,
    tokenizer,
    input_ids,
    attention_mask,
):
    kwargs = {
        "input_ids" : input_ids,
        "attention_mask" : attention_mask,
        "use_cache" : True,
        "do_sample" : False,

        # 强制每次生成完全相同数量的token
        "min_new_tokens": args.new_tokens,
        "max_new_tokens": args.new_tokens,

        "pad_token_id" : tokenizer.eos_token_id,
        "return_dict_in_generate": False,
    }

    if args.cache == "static":
        kwargs["cache_implementation"] = "static"
    
        if args.compile_mode == "none" :
            # static cache 单独实验，不允许自动编译
            kwargs["disable_compile"] = True
        
        else:
            if CompileConfig is None:
                raise RuntimeError(
                    "当前transformers 不支持 compileConfig , 请先检查transformers 版本"
                )
            
            kwargs["disable_compile"] = False

            kwargs["compile_config"] = CompileConfig(
                backend = "inductor",
                mode = "reduce-overhead",
                fullgraph = True,
                dynamic = False,
            )

    return kwargs

@torch.inference_mode()
def timed_generate(model, generation_kwargs):
    torch.cuda.synchronize()
    start = time.perf_counter()

    outputs = model.generate(**generation_kwargs)

    torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - start

    return outputs, elapsed_seconds

def token_hash(token_ids):
    token_string = ",".join(
        str(token_id) for token_id in token_ids
    )

    return hashlib.sha256(
        token_string.encode("utf-8")
    ).hexdigest()[:16]

def percentile_95(values):
    ordered = sorted(values)
    index = math.ceil(len(ordered) * 0.05) - 1
    return ordered[index]

def create_variant_name(args):
    compile_name = args.compile_mode.replace("-","_")

    return (
        f"{args.attention}_"
        f"{args.cache}_"
        f"{compile_name}_"
        f"p{args.prompt_tokens}_"
        f"o{args.new_tokens}_"
    )

def print_summary(summary):
    print("\n========实验结果========")
    print("实验配置:", summary["variant"])
    print(
        f"平均延迟: {summary['mean_latency_ms']:.3f} ms"
    )
    print(
        f"中位延迟: {summary['median_latency_ms']:.3f} ms"
    )
    print(
        f"P95 延迟: {summary['p95_latency_ms']:.3f} ms"
    )
    print(
        f"延迟波动 CV: {summary['latency_cv_percent']:.2f}%"
    )
    print(
        "平均输出吞吐: "
        f"{summary['mean_output_tokens_per_second']:.2f} token/s"
    )
    print(
        "峰值已分配显存: "
        f"{summary['peak_allocated_gib']:.3f} GiB"
    )
    print(
        "峰值保留显存: "
        f"{summary['peak_reserved_gib']:.3f} GiB"
    )
    print("输出 Token Hash:", summary["output_token_hash"])

def save_results(args, variant, results, summary):
    args.result_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.result_dir / f"{variant}.csv"
    json_path = args.result_dir / f"{variant}_summary.json"

    fieldnames = [
        "run",
        "variant",
        "model",
        "attention",
        "cache",
        "compile_mode",
        "prompt_tokens",
        "new_tokens",
        "latency_ms",
        "output_tokens_per_second",
        "peak_allocated_gib",
        "peak_reserved_gib",
        "output_token_hash",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:
            writer.writerow(result)

    json_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n原始数据:", csv_path)
    print("汇总数据:", json_path)

def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("当前无法使用cuda")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    variant = create_variant_name(args)

    print("========== 实验环境 ==========")
    print("Variant:", variant)
    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("Transformers:", transformers.__version__)
    print("CUDA:", torch.version.cuda)
    print("Attention:", args.attention)
    print("Cache:", args.cache)
    print("Compile:", args.compile_mode)
    print("Prompt tokens:", args.prompt_tokens)
    print("New tokens:", args.new_tokens)

    print("\n正在加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("正在加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation=args.attention,
    )
    model = model.to(DEVICE)
    model.eval()

    input_ids, attention_mask = build_fixed_input(
        tokenizer,
        args.prompt_tokens,
    )

    generation_kwargs = build_generation_kwargs(
        args = args,
        tokenizer = tokenizer,
        input_ids = input_ids,
        attention_mask = attention_mask
    )

    # warmup
    print(f"\n开始预热，共{args.warmup_runs} 次")
    for warmup_index in range(args.warmup_runs):
        outputs, elapsed_seconds = timed_generate(
            model,
            generation_kwargs,
        )

        generated_count = (
            outputs.shape[1] - input_ids.shape[1]
        )

        print(
            f"Warmup {warmup_index + 1}: "
            f"{elapsed_seconds * 1000:.2f} ms，"
            f"生成 {generated_count} token"
        )

        del outputs
    torch.cuda.synchronize()

# benchmark
    print(f"\n开始正式测试，共{args.test_runs} 次")

    results = []
    output_hashes = set()

    last_generated_ids = None

    for run_index in range(args.test_runs):
        if last_generated_ids is not None:
            del last_generated_ids

        torch.cuda.reset_peak_memory_stats()

        outputs, elapsed_seconds = timed_generate(
            model,
            generation_kwargs,
        )

        generated_ids = outputs[
            :,
            input_ids.shape[1] :,
        ]

        generated_count = generated_ids.shape[1]

        if generated_count != args.new_tokens:
            raise RuntimeError(
                f"预期生成{args.new_tokens} token"
                f"实际生成{generated_count} token"
            )
        
        generated_token_list = (
            generated_ids[0]
            .detach().cpu().tolist()
        )

        current_hash = token_hash(generated_token_list)
        output_hashes.add(current_hash)
        peak_allocated_gib = (torch.cuda.max_memory_allocated() / 1024**3)
        peak_reserved_gib = (torch.cuda.max_memory_reserved() / 1024**3)
        latency_ms = elapsed_seconds * 1000
        output_tps = (args.new_tokens / elapsed_seconds)

        result = {
            "run": run_index + 1,
            "variant": variant,
            "model": MODEL_ID,
            "attention": args.attention,
            "cache": args.cache,
            "compile_mode": args.compile_mode,
            "prompt_tokens": args.prompt_tokens,
            "new_tokens": args.new_tokens,
            "latency_ms": latency_ms,
            "output_tokens_per_second": output_tps,
            "peak_allocated_gib": peak_allocated_gib,
            "peak_reserved_gib": peak_reserved_gib,
            "output_token_hash": current_hash,
        }

        results.append(result)
        last_generated_ids = generated_ids
        print(
            f"[{run_index + 1:02d}/{args.test_runs}] "
            f"Latency={latency_ms:.2f} ms | "
            f"Throughput={output_tps:.2f} token/s | "
            f"VRAM={peak_allocated_gib:.3f} GiB"
        )
    if len(output_hashes) != 1:
        raise RuntimeError(
            "同一配置的多次greedy输出不一致，请先检查确定性和输入是否发生变化"
        )
    latency_values = [result["latency_ms"] for result in results]
    throughput_values = [result["output_tokens_per_second"] for result in results]
    mean_latency = statistics.mean(latency_values)
    latency_std = statistics.pstdev(latency_values)

    summary = {
        "variant": variant,
        "model": MODEL_ID,
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda,
        "attention": args.attention,
        "cache": args.cache,
        "compile_mode": args.compile_mode,
        "prompt_tokens": args.prompt_tokens,
        "new_tokens": args.new_tokens,
        "warmup_runs": args.warmup_runs,
        "test_runs": args.test_runs,
        "mean_latency_ms": mean_latency,
        "median_latency_ms": statistics.median(
            latency_values
        ),
        "p95_latency_ms": percentile_95(
            latency_values
        ),
        "latency_std_ms": latency_std,
        "latency_cv_percent": (
            latency_std / mean_latency * 100
        ),
        "mean_output_tokens_per_second": (
            statistics.mean(throughput_values)
        ),
        "peak_allocated_gib": max(
            result["peak_allocated_gib"]
            for result in results
        ),
        "peak_reserved_gib": max(
            result["peak_reserved_gib"]
            for result in results
        ),
        "output_token_hash": next(
            iter(output_hashes)
        ),
        "sample_token_ids": (
            last_generated_ids[0]
            .detach()
            .cpu()
            .tolist()
        ),
        "sample_text": tokenizer.decode(
            last_generated_ids[0],
            skip_special_tokens=True,
        ),
    }
    print_summary(summary)

    print("\n最后一次输出：")
    print(summary["sample_text"])

    save_results(
        args=args,
        variant=variant,
        results=results,
        summary=summary,
    )

if __name__ == "__main__":
    main()