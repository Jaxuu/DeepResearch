"""Open Deep Research 系统的配置管理模块。"""

import os
from enum import Enum
from typing import Any, List, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv(override=True)

class SearchAPI(Enum):
    """可用搜索 API 提供商的枚举类。"""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    TAVILY = "tavily"
    NONE = "none"

class MCPConfig(BaseModel):
    """模型上下文协议 (MCP) 服务器的配置类。"""

    url: Optional[str] = Field(
        default=None,
        optional=True,
    )
    """MCP 服务器的 URL 地址"""
    tools: Optional[List[str]] = Field(
        default=None,
        optional=True,
    )
    """向大语言模型开放使用的工具列表"""
    auth_required: Optional[bool] = Field(
        default=False,
        optional=True,
    )
    """MCP 服务器是否需要身份验证"""

class Configuration(BaseModel):
    """Deep Research 智能体的主配置类。"""

    # 通用配置
    allow_clarification: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Whether to allow the researcher to ask the user clarifying questions before starting research"
            }
        }
    )
    max_concurrent_research_units: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 5,
                "min": 1,
                "max": 20,
                "step": 1,
                "description": "Maximum number of research units to run concurrently. This will allow the researcher to use multiple sub-agents to conduct research. Note: with more concurrency, you may run into rate limits."
            }
        }
    )
    max_structured_output_retries: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 3,
                "min": 1,
                "max": 10,
                "description": "Maximum number of retries for structured output calls from models"
            }
        }
    )
    max_verification_retries: int = Field(
        default=2,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 2,
                "min": 1,
                "max": 4,
                "description": "Maximum number of retries for verification calls from models"
            }
        }
    )
    # 调研配置
    search_api: SearchAPI = Field(
        default=SearchAPI.TAVILY,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "tavily",
                "description": "Search API to use for research. NOTE: Make sure your Researcher Model supports the selected search API.",
                "options": [
                    {"label": "Tavily", "value": SearchAPI.TAVILY.value},
                    {"label": "OpenAI Native Web Search", "value": SearchAPI.OPENAI.value},
                    {"label": "Anthropic Native Web Search", "value": SearchAPI.ANTHROPIC.value},
                    {"label": "None", "value": SearchAPI.NONE.value}
                ]
            }
        }
    )
    max_researcher_iterations: int = Field(
        default=6,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 6,
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "Maximum number of research iterations for the Research Supervisor. This is the number of times the Research Supervisor will reflect on the research and ask follow-up questions."
            }
        }
    )
    max_react_tool_calls: int = Field(
        default=10,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 10,
                "min": 1,
                "max": 30,
                "step": 1,
                "description": "Maximum number of tool calling iterations to make in a single researcher step."
            }
        }
    )
    # 模型配置
    max_content_length: int = Field(
        default=12000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 50000,
                "min": 1000,
                "max": 200000,
                "description": "Maximum character length for webpage content before summarization"
            }
        }
    )
    research_model: str = Field(
        default=os.getenv("RESEARCH_MODEL"),
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": os.getenv("RESEARCH_MODEL"),   # 前端提示信息，不影响运行
                "description": "Model for conducting research. NOTE: Make sure your Researcher Model supports the selected search API."
            }
        }
    )
    research_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum output tokens for research model"
            }
        }
    )
    summarization_model: str = Field(
        default=os.getenv("SUMMARIZATION_MODEL"),
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": os.getenv("SUMMARIZATION_MODEL"),
                "description": "Model for summarizing research results from Tavily search results"
            }
        }
    )
    summarization_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for summarization model"
            }
        }
    )
    compression_model: str = Field(
        default=os.getenv("COMPRESSION_MODEL"),
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": os.getenv("COMPRESSION_MODEL"),
                "description": "Model for compressing research findings from sub-agents. NOTE: Make sure your Compression Model supports the selected search API."
            }
        }
    )
    compression_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for compression model"
            }
        }
    )
    final_report_model: str = Field(
        default=os.getenv("FINAL_REPORT_MODEL"),
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": os.getenv("FINAL_REPORT_MODEL"),
                "description": "Model for writing the final report from all research findings"
            }
        }
    )
    final_report_model_max_tokens: int = Field(
        default=12000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum output tokens for final report model"
            }
        }
    )
    verifier_model: str = Field(
        default=os.getenv("VERIFIER_MODEL"),
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": os.getenv("VERIFIER_MODEL"),
                "description": "Model for verifying the generated report"
            }
        }
    )
    verifier_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum output tokens for verifier model"
            }
        }
    )
    rewrite_model: str = Field(
        default=os.getenv("REWRITE_MODEL"),
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": os.getenv("REWRITE_MODEL"),
                "description": "Model for rewriting the generated report"
            }
        }
    )
    rewrite_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum output tokens for verifier model"
            }
        }
    )
    # MCP 服务器配置
    mcp_config: Optional[MCPConfig] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "mcp",
                "description": "MCP server configuration"
            }
        }
    )
    mcp_prompt: Optional[str] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "description": "Any additional instructions to pass along to the Agent regarding the MCP tools that are available to it."
            }
        }
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """从 RunnableConfig 或环境变量创建 Configuration 实例。"""
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {}

        for field_name in field_names:
            val = os.environ.get(field_name.upper(), configurable.get(field_name))

            # 自动补全规则：若是模型字段且以 qwen 开头，自动拼接 openai: 前缀
            if isinstance(val, str) and "model" in field_name:
                # if val.startswith("qwen") and not val.startswith("openai:"):
                val = f"openai:{val}"

            values[field_name] = val

        return cls(**{k: v for k, v in values.items() if v is not None})

    class Config:
        """Pydantic 配置。"""
        
        arbitrary_types_allowed = True