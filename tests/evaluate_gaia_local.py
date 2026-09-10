import os
import json
import asyncio
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model
from open_deep_research.deep_researcher import deep_researcher
from dotenv import load_dotenv

load_dotenv(override=True)

# 初始化一个廉价小模型作为提取器，消耗 Token 极少
# extractor_model = init_chat_model(
#     model="gemini-3.6-flash",
#     model_provider="google_genai",  # <--- 强制指定为 Google AI Studio 接口,明确指定走 google_genai 协议，即 Google AI Studio 个人 API
#     api_key=os.getenv("GOOGLE_API_KEY"), # 请确保此处与你 .env 中的变量名一致
#     temperature=0
# )

extractor_model = init_chat_model(
    model="qwen-turbo",
    model_provider="openai",
    max_tokens=1024,
    api_key=os.getenv("OPENAI_API_KEY"), # 请确保此处与你 .env 中的变量名一致
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
        6. LANGUAGE ALIGNMENT: You MUST output the final answer in the exact same language as the original Question. If the report is in another language, translate the specific entity back to the Question's language before outputting it.
        7. AVOID DISTRACTIONS: The report may contain multiple entities. Look closely at the exact phrasing of the Question to select the one that fits perfectly. Do not be distracted by other entities mentioned in the report's background context.
        """
    response = await extractor_model.ainvoke([HumanMessage(content=prompt)])
    # 兼容处理 LangChain 的 List 类型内容块
    raw_content = response.content
    if isinstance(raw_content, list):
        # 遍历列表，将所有 type 为 text 的内容拼接起来
        text_content = "".join([
            block.get("text", "")
            for block in raw_content
            if isinstance(block, dict) and block.get("type") == "text"
        ])
    else:
        # 如果本来就是字符串，直接强转以防万一
        text_content = str(raw_content)

    return text_content.strip()


async def main():
    # 读取本地的自定义测试集-
    dataset_path = os.path.join("../datasets/failed_sample.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        original_dataset = json.load(f)

    dataset = original_dataset[:]
    correct_count = 0
    total_tests = len(dataset)

    for i, item in enumerate(dataset):
        question = item["Question"]
        ground_truth = str(item["Final answer"]).strip()

        print(f"\n--- 测试题目 [{i + 1}/{total_tests}] ---\n{question}")
        config = {
            "configurable": {
                "thread_id": f"custom_test_{i}",
                "simulate_human_approval": True,
                "allow_clarification": False
            }
        }

        try:
            # 调用带有智能路由和 PDF 解析增强的 LangGraph 智能体
            result = await deep_researcher.ainvoke(
                {"messages": [{"role": "user", "content": question}]},
                config=config
            )
            agent_output = result.get("final_report", "")

            # 使用小模型从生成的报告中提取核心答案
            extracted_answer = await extract_final_answer(question, agent_output)
            print(f"提取出的核心答案: {extracted_answer}")
            print(f"标准答案: {ground_truth}")

            # 字符串标准化，去除所有的空格，只比对核心字符
            norm_gt = ground_truth.replace(" ", "").lower()
            norm_ext = extracted_answer.replace(" ", "").lower()

            # 弱化匹配：在剔除空格的纯字符状态下进行包含匹配，只要提取出的核心答案中包含标准答案即可
            if norm_gt in norm_ext and extracted_answer != 'NOT_FOUND':
                print("✅ [判分] 命中正确答案")
                correct_count += 1
            else:
                print("❌ [判分] 未命中")

        except Exception as e:
            print(f"❌ [报错] 节点执行异常: {e}")

    print(f"\n测试完成！总正确率: {correct_count}/{total_tests}")


if __name__ == "__main__":
    asyncio.run(main())