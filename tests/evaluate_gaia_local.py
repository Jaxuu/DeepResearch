import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("langchain_openai").setLevel(logging.WARNING)
import os
import json
import asyncio
import time
from langchain_community.callbacks import get_openai_callback
from langchain_core.tracers.langchain import wait_for_all_tracers
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model
from open_deep_research.deep_researcher import deep_researcher
from open_deep_research.utils import cleanup_mcp_sessions
from dotenv import load_dotenv

load_dotenv(override=True)

# 定义 GAIA 验证集文件的绝对路径
GAIA_VALIDATION_DIR = r"D:\Project\llm-project\open_deep_research\datasets\GAIA\2023\validation"

extractor_model = init_chat_model(
    model="qwen-turbo",
    model_provider="openai",
    max_tokens=1024,
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0
)


async def extract_final_answer(question: str, report: str) -> str:
    """从长篇研报中精准提取最终答案"""
    prompt = f"""
            You are a precision answer extractor for the GAIA benchmark. 
            Question: {question}
            Report/Thoughts: {report}

            Task: Extract the exact, specific answer to the question based ONLY on the provided text. 

            CRITICAL RULES:
            1. If the question asks for a full statement, logical formula, or paper title, you MUST output the EXACT FULL STRING (e.g., "(¬A → B) ↔ (A ∨ ¬B)" or "A New Software Agent Learning Algorithm").
            2. NEVER shrink a formula, title, or text option into a single section number. (e.g., Do NOT output "5" if the answer is the text of option 5).
            3. If the question asks for a numeric calculation, output just the number.
            4. Do NOT output conversational fillers like "The answer is" or "Based on the report".
            5. If the report does not contain the answer, output 'NOT_FOUND'.
            6. LANGUAGE ALIGNMENT: You MUST output the final answer in the exact same language as the original Question.
            7. AVOID DISTRACTIONS: The report may contain multiple entities. Look closely at the exact phrasing of the Question to select the one that fits perfectly.
            8. ENTITY & MODIFIER PRESERVATION: When extracting lists, ingredients, or specific nouns, NEVER strip descriptive adjectives (e.g., keep "ripe strawberries" instead of "strawberries", keep "pure vanilla extract") if they are present in the report.
            9. EXACT FORMAT MATCHING: If the question requests a specific format (e.g., "comma separated list"), your output MUST strictly maintain that exact format without adding extra spaces, bullets, or periods at the end.
            """
    response = await extractor_model.ainvoke([HumanMessage(content=prompt)])

    raw_content = response.content
    if isinstance(raw_content, list):
        text_content = "".join([
            block.get("text", "")
            for block in raw_content
            if isinstance(block, dict) and block.get("type") == "text"
        ])
    else:
        text_content = str(raw_content)

    return text_content.strip()


def extract_token_usage(result: dict) -> int:
    """从 deep_researcher 的返回结果中提取 token 总消耗"""
    total = 0

    # 方式一：LangGraph / LangChain 常见的 usage_metadata 挂在 messages 上
    for msg in result.get("messages", []):
        usage = getattr(msg, "usage_metadata", None)
        if usage:
            total += usage.get("total_tokens", 0)

    # 方式二：部分版本会直接在 state 里放 token 统计字段
    if total == 0:
        for key in ("total_tokens", "token_usage", "usage"):
            val = result.get(key)
            if isinstance(val, int):
                total += val
            elif isinstance(val, dict):
                total += val.get("total_tokens", 0)

    return total


async def run_single_dataset(dataset_path: str, dataset_name: str, run_tag: str) -> dict:
    """运行单个数据集，返回统计结果 + 正确/错误原始题目

    Args:
        dataset_path: 数据集 JSON 路径
        dataset_name: 数据集名称（如 research_sample）
        run_tag: 本次运行标签（时间戳），用于隔离 thread_id
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    correct_count = 0
    total_tests = len(dataset)
    total_tokens = 0
    durations = []
    correct_items = []  # 答对的原始题目
    error_items = []    # 答错的原始题目

    for i, item in enumerate(dataset):
        original_question = item["Question"]
        ground_truth = str(item["Final answer"]).strip()
        file_name = item.get("file_name", "").strip()

        question_with_file = original_question
        if file_name:
            file_path = os.path.join(GAIA_VALIDATION_DIR, file_name)
            if os.path.exists(file_path):
                question_with_file += (
                    f"\n\n[CRITICAL SYSTEM NOTE: The user has attached a file for this task. "
                    f"The file is located at absolute path: {file_path}. "
                    f"You MUST use your tools to analyze this specific file.]"
                )
            else:
                print(f"⚠️ 警告: 找不到本地附件文件 {file_path}，这可能导致大模型无法解答！")
                question_with_file += f"\n\n[Attached File Not Found: {file_name}]"

        print(f"\n--- [{dataset_name}] 测试题目 [{i + 1}/{total_tests}] ---\n{original_question}")
        if file_name:
            print(f"📁 挂载文件: {file_name}")

        config = {
            "configurable": {
                "thread_id": f"{dataset_name}_{run_tag}_test_{i}",
                "simulate_human_approval": True,
                "allow_clarification": False,
            }
        }

        hit = False
        extracted_answer = None

        try:
            start_time = time.perf_counter()

            # 使用回调管理器拦截所有 LLM 调用的 Token 消耗
            with get_openai_callback() as cb:
                result = await deep_researcher.ainvoke(
                    {"messages": [{"role": "user", "content": question_with_file}]},
                    config=config,
                )
                # 直接从回调对象中获取累加的总 Token
                run_tokens = cb.total_tokens

            elapsed = time.perf_counter() - start_time
            durations.append(elapsed)

            total_tokens += run_tokens

            agent_output = result.get("final_report", "")
            extracted_answer = await extract_final_answer(original_question, agent_output)

            print(f"提取出的核心答案: {extracted_answer}")
            print(f"标准答案: {ground_truth}")

            norm_gt = ground_truth.replace(" ", "").lower()
            norm_ext = extracted_answer.replace(" ", "").lower()

            # 归一化：去空格 + 转小写 + 统一各种分隔符为英文逗号
            def _normalize(s: str) -> str:
                s = s.replace(" ", "").lower()
                # 把常见中文/英文分隔符统一成英文逗号，方便切分
                for sep in ["、", "，", ";", "；", "/", "|", "\n"]:
                    s = s.replace(sep, ",")
                return s

            norm_gt = _normalize(ground_truth)
            norm_ext = _normalize(extracted_answer)

            # 标准答案可能由多个子答案组成（用逗号分隔），逐个检查是否都被生成答案包含
            gt_parts = [p for p in norm_gt.split(",") if p]
            hit = (
                    extracted_answer != "NOT_FOUND"
                    and len(gt_parts) > 0
                    and all(part in norm_ext for part in gt_parts)
            )

            if hit:
                print("✅ [判分] 命中正确答案")
                correct_count += 1
            else:
                print("❌ [判分] 未命中")
        except Exception as e:
            print(f"❌ [报错] 节点执行异常: {e}")

        # 按正确/错误归类到原始 item（保持和测试集一样的格式）
        if hit:
            correct_items.append(item)
        else:
            error_items.append(item)

    # 耗时统计
    if durations:
        sorted_d = sorted(durations)

        def percentile(data, p):
            k = (len(data) - 1) * p
            f = int(k)
            c = min(f + 1, len(data) - 1)
            return data[f] + (data[c] - data[f]) * (k - f)

        avg_time = sum(durations) / len(durations)
        p50 = percentile(sorted_d, 0.50)
        p95 = percentile(sorted_d, 0.95)
        p99 = percentile(sorted_d, 0.99)
    else:
        avg_time = p50 = p95 = p99 = 0.0

    # 一致性断言
    assert correct_count == len(correct_items), (
        f"[{dataset_name}] correct_count({correct_count}) != "
        f"len(correct_items)({len(correct_items)})"
    )
    assert (total_tests - correct_count) == len(error_items), (
        f"[{dataset_name}] 错题数不匹配: total-correct={total_tests - correct_count}, "
        f"len(error_items)={len(error_items)}"
    )

    return {
        "dataset_name": dataset_name,
        "total_tests": total_tests,
        "correct_count": correct_count,
        "accuracy": correct_count / total_tests if total_tests else 0.0,
        "total_tokens": total_tokens,
        "avg_tokens_per_test": total_tokens / total_tests if total_tests else 0.0,
        "timing_seconds": {
            "avg": avg_time,
            "p50": p50,
            "p95": p95,
            "p99": p99,
            "all_durations": durations,
        },
        "correct_items": correct_items,
        "error_items": error_items,
    }


async def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 创建带日期的结果文件夹
    timestamp = time.strftime("%Y%m%d%H%M%S")  # 例如 20261005233500
    result_dir_name = f"result{timestamp}"
    result_dir = os.path.join(current_dir, result_dir_name)
    os.makedirs(result_dir, exist_ok=True)
    print(f"📂 结果文件夹已创建: {result_dir}")

    # 本次运行标签（用于 thread_id 隔离）
    run_tag = timestamp

    # 数据集列表
    dataset_files = [
        "multimodal_sample.json"
    ]

    per_dataset_stats = []
    grand_total_tests = 0
    grand_total_correct = 0
    grand_total_tokens = 0
    grand_all_durations = []

    for fname in dataset_files:
        dataset_path = os.path.join(current_dir, fname)
        if not os.path.exists(dataset_path):
            print(f"⚠️ 找不到数据集文件: {dataset_path}，跳过。")
            continue

        base_filename = os.path.basename(fname)
        dataset_name = os.path.splitext(base_filename)[0]
        print(f"\n{'=' * 60}\n🚀 开始测试数据集: {dataset_name}\n{'=' * 60}")

        stats = await run_single_dataset(dataset_path, dataset_name, run_tag)

        # 去掉 _sample 后缀，得到 short_name（例如 research）
        short_name = (
            dataset_name[:-len("_sample")]
            if dataset_name.endswith("_sample")
            else dataset_name
        )

        # 保存答对 / 答错的原始题目（格式和测试集一样）
        correct_path = os.path.join(result_dir, f"{short_name}_correct.json")
        error_path = os.path.join(result_dir, f"{short_name}_error.json")

        with open(correct_path, "w", encoding="utf-8") as f:
            json.dump(stats["correct_items"], f, ensure_ascii=False, indent=2)

        with open(error_path, "w", encoding="utf-8") as f:
            json.dump(stats["error_items"], f, ensure_ascii=False, indent=2)

        print(f"   ✅ 已保存 {len(stats['correct_items'])} 道正确题 -> {os.path.basename(correct_path)}")
        print(f"   ❌ 已保存 {len(stats['error_items'])} 道错题   -> {os.path.basename(error_path)}")

        # 只保留统计字段，剔除 correct_items / error_items（不进汇总文件）
        stats_slim = {
            k: v for k, v in stats.items()
            if k not in ("correct_items", "error_items")
        }
        # 从 timing_seconds 里剔除 all_durations
        stats_slim["timing_seconds"] = {
            tk: tv for tk, tv in stats_slim["timing_seconds"].items()
            if tk != "all_durations"
        }
        per_dataset_stats.append(stats_slim)

        grand_total_tests += stats["total_tests"]
        grand_total_correct += stats["correct_count"]
        grand_total_tokens += stats["total_tokens"]
        grand_all_durations.extend(stats["timing_seconds"]["all_durations"])

        print(
            f"\n📌 [{dataset_name}] 正确率: {stats['correct_count']}/{stats['total_tests']} "
            f"= {stats['accuracy']:.2%} | tokens: {stats['total_tokens']}"
        )

    # 汇总统计
    if grand_all_durations:
        sorted_d = sorted(grand_all_durations)

        def percentile(data, p):
            k = (len(data) - 1) * p
            f = int(k)
            c = min(f + 1, len(data) - 1)
            return data[f] + (data[c] - data[f]) * (k - f)

        avg_time = sum(grand_all_durations) / len(grand_all_durations)
        p50 = percentile(sorted_d, 0.50)
        p95 = percentile(sorted_d, 0.95)
        p99 = percentile(sorted_d, 0.99)
    else:
        avg_time = p50 = p95 = p99 = 0.0

    overall_stats = {
        "total_tests": grand_total_tests,
        "correct_count": grand_total_correct,
        "accuracy": grand_total_correct / grand_total_tests if grand_total_tests else 0.0,
        "total_tokens": grand_total_tokens,
        "avg_tokens_per_test": grand_total_tokens / grand_total_tests if grand_total_tests else 0.0,
        "timing_seconds": {
            "avg": avg_time,
            "p50": p50,
            "p95": p95,
            "p99": p99
        },
    }

    # 组装最终汇总：三个数据集各自的统计 + 总体统计
    final_stats = {
        "per_dataset": per_dataset_stats,
        "overall": overall_stats,
    }

    stats_path = os.path.join(result_dir, "gaia_eval_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(final_stats, f, ensure_ascii=False, indent=2)

    print(f"\n📊 统计结果已保存至: {stats_path}")
    print(f"   总正确率: {grand_total_correct}/{grand_total_tests} = "
          f"{(grand_total_correct / grand_total_tests if grand_total_tests else 0):.2%}")
    print(f"   总 token 消耗: {grand_total_tokens}")
    print(f"   平均耗时: {avg_time:.2f}s | p50: {p50:.2f}s | p95: {p95:.2f}s | p99: {p99:.2f}s")

    print("正在等待 LangSmith 追踪数据提交...")
    wait_for_all_tracers()
    print("追踪数据提交完成。")

    await cleanup_mcp_sessions()


if __name__ == "__main__":
    asyncio.run(main())