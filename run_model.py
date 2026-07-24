import torch 
from transformers import AutoModelForCausalLM , AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
DEVICE = "cuda"

def main():

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # 检查cuda
    if not torch.cuda.is_available():
        raise RuntimeError(
            "python 无法使用cuda , 请检查显卡驱动和pytorch cuda 版本"
        )

    print("gpu:", torch.cuda.get_device_name(0))
    print("pytorch cuda", torch.version.cuda)

    # 加载分词器
    print("正在加载 tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # 以FP16加载模型
    print("正在加载fp16模型")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype = torch.float16,
        low_cpu_mem_usage = True,
    )

    model = model.to(DEVICE)
    model.eval()

    print("模型参数类型:",next(model.parameters()).dtype)
    print("模型所在设备:",next(model.parameters()).device)

    # 构造对话
    messages = [
        {
            "role": "system",
            "content": "你是一名中文助手。无论用户使用什么语言，你都必须使用简体中文回答。",
        },
        {
            "role": "user",
            "content": "请用简单语言解释什么是高性能计算"
        },
    ]
    
    # 关闭qwen3 的思考模式，减少不必要的输出
    text = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True,
        enable_thinking = False,
    )

    # 检查 user 消息是否真的进入模板
    print("聊天模板内容：")
    print(repr(text))

    # 分词并移动到gpu
    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to(DEVICE)

    print("输入token数:", inputs["input_ids"].shape[1])
    print("正在生成回答")

    # 执行推理
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens = 128,
            do_sample = True,
            temperature = 0.7,
            top_p = 0.8,
            top_k = 20,
            use_cache = True,
            pad_token_id = tokenizer.eos_token_id,
        )
    
    
    # 只提取新生成的token
    input_length = inputs["input_ids"].shape[1]
    generated_ids = outputs[:, inputs["input_ids"].shape[1]:]

    answer = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens = True,
    )[0]

    print("\n模型回答")
    print(answer)

    # 显示当前显存占用
    allocated_memory = torch.cuda.memory_allocated() / 1024 ** 3
    reserved_memory = torch.cuda.memory_reserved() / 1024 ** 3

    print("\n显存信息")
    print(f"已分配内存：{allocated_memory:.2f} GiB")
    print(f"pytorch 保留显存：{reserved_memory:.2f} GiB")

if __name__ == "__main__" :
    main()