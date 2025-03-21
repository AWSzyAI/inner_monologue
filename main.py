import pandas as pd
import re
import json
import logging
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from kimi_api import send_messages, MODEL_NAME  # 引入 MODEL_NAME

# 配置日志
logging.basicConfig(
    filename="generation_log.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8"
)

# 设定并发线程数 (根据 API 速率调整)
MAX_WORKERS = 5

CHECKPOINT_FILE = "checkpoint.txt"
CACHE_FILE = "cache.csv"
FAIL_FILE = "fail_data.csv"
OUTPUT_FILE = "自我肯定语_生成旁白.csv"

def clean_value(value):
    """清理字符串，确保换行符 \n 被转换为 '前文\\n后文' 形式"""
    if isinstance(value, str):
        return value.replace("\n", "\\n")  # 转换换行符
    return value

def extract_json(response):
    """解析 AI 响应，确保 JSON 格式正确"""
    match = re.search(r'\{.*\}', response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))  # 解析 JSON 数据
        except json.JSONDecodeError as e:
            logging.warning(f"JSON 解析失败: {e}, 跳过该条数据: {response}")
            return None  # 解析失败
    else:
        logging.warning(f"未找到 JSON 数据: {response}")
        return None  # 未找到 JSON

def process_sentence(index, sentence):
    """处理单个自我肯定语，生成内心旁白"""
    prompt = f"""
    自我肯定语：{sentence}

    请仿照萨提亚的《当我真的愿意看见自己时》的风格，为输入的自我肯定语生成一段内心旁白。
    注意适当换行以减少读者的阅读难度。分三到四段生成内心旁白。不要写诗。
    约500字。
    必须以第一人称叙述。
    
    请严格按照以下 JSON 格式返回数据：
    {{
      "inner_monologue": "这里是生成的内心旁白内容"
    }}
    """

    try:
        # 第一次生成
        messages = [{"role": "user", "content": prompt}]
        response = send_messages(messages)
        parsed_response = extract_json(response)
        
        if parsed_response and "inner_monologue" in parsed_response:
            # 二次校验
            check_prompt = f"""
            针对上一次生成的内心旁白：
            {parsed_response["inner_monologue"]}

            请检查并优化以下内容：
            - 修正标点/空格问题
            - 改善语句通顺度
            - 统一人称（第一人称），必须以第一人称叙述。
            - 删除外语内容
            - 防止场景过于具体
            - 确保500字长度
            - 删除奇怪比喻
            - 修正语病/错别字
            
            直接返回优化后的JSON：
            {{
                "inner_monologue": "这里是修改后生成的内心旁白内容"
            }}
            """
            messages.append({"role": "user", "content": check_prompt})
            # 第二次生成
            check_response = send_messages(messages)
            check_parsed_response = extract_json(check_response)
            
            if check_parsed_response and "inner_monologue" in check_parsed_response:
                return index, {
                    "自我肯定语": sentence,
                    "自我旁白": clean_value(check_parsed_response["inner_monologue"]),
                    "MODEL_NAME": MODEL_NAME
                }
            else:
                logging.warning(f"生成失败，跳过: {sentence}")
                return index, None
        else:
            logging.warning(f"生成失败，跳过: {sentence}")
            return index, None

    except Exception as e:
        logging.error(f"处理 {sentence} 时发生错误: {e}")
        return index, None

def save_checkpoint(completed_indexes):
    """保存已完成的任务索引"""
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        f.write(",".join(map(str, completed_indexes)))

def load_checkpoint():
    """加载已完成的任务索引"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            return set(map(int, content.split(","))) if content else set()
    return set()

def process_sentences_concurrently(sentences, start_index=0, ignore_checkpoint=False):
    """
    使用多线程并发处理句子。
    当 ignore_checkpoint=True 时，忽略所有已经完成的checkpoint记录。
    """
    results = []
    fail_data = []
    
    if ignore_checkpoint:
        completed_indexes = set()
    else:
        completed_indexes = load_checkpoint()

    logging.info(f"开始并发处理 {len(sentences)} 条自我肯定语...")

    # 筛除掉已经完成的索引（当 ignore_checkpoint=False 时才会起作用）
    to_process = [
        (i + start_index, s)
        for i, s in enumerate(sentences)
        if (i + start_index) not in completed_indexes
    ]
    
    if not to_process:
        logging.info("所有数据已完成，跳过处理。")
        return results, fail_data

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_index = {
            executor.submit(process_sentence, idx, sent): idx
            for idx, sent in to_process
        }

        for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="生成内心旁白", unit="条"):
            index, result = future.result()
            if result:
                results.append(result)
                completed_indexes.add(index)
            else:
                # 记录失败数据
                original_sentence = sentences[index - start_index]
                fail_data.append({"自我肯定语": original_sentence})
    
    # 保存最新的 checkpoint（仅当不忽略checkpoint时才保存）
    if not ignore_checkpoint:
        save_checkpoint(completed_indexes)

    return results, fail_data

def save_results(results, fail_data):
    """保存成功和失败数据"""
    # 1. 成功数据追加保存
    if results:
        output_df = pd.DataFrame(results)
        if os.path.exists(OUTPUT_FILE):
            existing_df = pd.read_csv(OUTPUT_FILE, encoding="utf-8-sig")
            output_df = pd.concat([existing_df, output_df], ignore_index=True)

        output_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        logging.info(f"✅ 结果已追加保存至 {OUTPUT_FILE}")
    else:
        logging.info("没有新的成功数据需要保存。")

    # 2. 失败数据覆盖保存
    if fail_data:
        fail_df = pd.DataFrame(fail_data)
        fail_df.to_csv(FAIL_FILE, index=False, encoding="utf-8-sig")
        logging.warning(f"⚠️ 失败数据已保存至 {FAIL_FILE}")
    else:
        # 若无失败数据，则清理之前的 fail_data.csv
        if os.path.exists(FAIL_FILE):
            os.remove(FAIL_FILE)
        logging.info("🎉 所有数据处理成功，无失败数据")

def main():
    """主入口：用户选择运行模式"""
    print("请选择运行模式：")
    print("0. EXIT")
    print("1. 从总数据中开始生产")
    print("2. 重试失败的案例（读取 fail_data.csv）")
    print("3. 从 checkpoint 中恢复生产")
    choice = input("请输入 0, 1, 2 或 3: ").strip()

    if choice == "1":
        # 读取整份数据
        # df = pd.read_csv("0315句子更新 - 汇总表.csv", encoding="utf-8-sig").sample(100)
        df = pd.read_csv("0315句子更新 - 汇总表.csv", encoding="utf-8-sig")
        df = df[df["权重"] == 3]
        df.to_csv(CACHE_FILE, index=False, encoding="utf-8-sig")  # 保存到 cache 以便断点续传
        start_index = 0
        sentences = df['自我肯定语'].dropna().tolist()
        print(f"len(sentences): {len(sentences)}")
        # 使用并发处理数据（此时不忽略checkpoint，以便可以从中断处继续）
        results, fail_data = process_sentences_concurrently(sentences, start_index, ignore_checkpoint=False)
        save_results(results, fail_data)

    elif choice == "2":
        # 重试失败的案例
        if os.path.exists(FAIL_FILE):
            df = pd.read_csv(FAIL_FILE, encoding="utf-8-sig")
            sentences = df['自我肯定语'].dropna().tolist()
            start_index = 0
            # 关键：忽略 checkpoint，确保可以重新处理
            results, fail_data = process_sentences_concurrently(sentences, start_index, ignore_checkpoint=True)
            save_results(results, fail_data)
        else:
            print("❌ 没有 fail_data.csv，所有数据已成功处理！")

    elif choice == "3":
        # 从 checkpoint 中恢复
        if os.path.exists(CACHE_FILE):
            df = pd.read_csv(CACHE_FILE, encoding="utf-8-sig")
            completed_indexes = load_checkpoint()
            if len(completed_indexes) >= len(df):
                print("🎉 所有数据已成功处理，无需恢复")
                return
            start_index = 0
            sentences = df['自我肯定语'].dropna().tolist()
            results, fail_data = process_sentences_concurrently(sentences, start_index, ignore_checkpoint=False)
            save_results(results, fail_data)
        else:
            print("❌ 未找到 cache.csv，无法从 checkpoint 恢复！")

    elif choice == "0":
        exit(1)
    else:
        print("❌ 无效输入，请输入 0, 1, 2 或 3")

if __name__ == "__main__":
    main()