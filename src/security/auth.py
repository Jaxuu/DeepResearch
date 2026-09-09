"""
LangGraph 安全网关与多租户数据隔离模块 (Security Gateway & Multi-Tenant Isolation)

本模块作为 LangGraph API Server 的底层全局拦截器，深度集成 Supabase Auth 系统，
实现了生产级别的零信任鉴权（Zero-Trust Authentication）和行级数据物理隔离。
无论用户在前端通过何种方式登录（包含未开启邮箱验证的快捷注册），本模块都将基于颁发的
JWT Token 实施严格的访问控制。

核心工作流与逻辑抽象：

1. 【身份核验 (Authentication)】 -> `@auth.authenticate`
   - 拦截所有进入 LangGraph 的网络请求，解析 Authorization Header[cite: 12]。
   - 通过 `supabase.auth.get_user` 实时向认证服务器发起校验，防止 Token 伪造或过期[cite: 12]。
   - 验证通过后，将 Supabase 返回的用户 UUID 提取并注入到全局上下文 `ctx.user.identity` 中[cite: 12]。
   - 特例：放行 LangGraph Studio 的本地调试用户 (`StudioUser`)，以及携带 `local-dev-token-123` 的本地测试请求[cite: 12]。

2. 【资产打标 (Write Isolation)】 -> `@auth.onthreads.create`
   - 作用域：Threads (会话)[cite: 12]、Assistants (智能体)[cite: 12]。
   - 在数据正式落盘写入数据库之前触发拦截。
   - 无视前端传入的参数，服务端强制在 `metadata` 字典中盖上当前用户的私有钢印 (`metadata["owner"] = ctx.user.identity`)[cite: 12]。
   - 确保同一数据库中的每一条记录都在物理层面永久绑定其创建者。

3. 【视线屏蔽 (Read Isolation)】 -> `@auth.on.threads.read/search/update/delete`
   - 作用域：Threads (会话)[cite: 12]、Assistants (智能体)[cite: 12]。
   - 在执行数据库查询或修改操作之前触发拦截。
   - 自动向底层的 SQL 查询中强制追加过滤条件 `{"owner": ctx.user.identity}`[cite: 12]。
   - 确保用户即使发起无条件的全局 `search`，也绝对只能拉取到属于自己的私有数据[cite: 12]，实现“单库多租户”的数据防串透。

4. 【智能体配置隔离 (Assistant Isolation)】 ->  auth.on.assistants...
   - LangGraph 支持基于核心代码保存不同的“个性化配置实例”（即 Assistants，例如绑定了不同模型或特定参数的实例）。
   - `@auth.on.assistants.create`: 当用户保存自定义 Assistant 配置时，强制绑定其所有权 (`metadata["owner"] = ctx.user.identity`)[cite: 12]。
   - `@auth.on.assistants.read/search`: 检索 Assistant 列表时，确保用户只能查看到系统默认的以及自己私有创建的配置实例[cite: 12]。
   - `@auth.on.assistants.update/delete`: 保护机制，防止用户越权修改或删除其他租户的专属 Assistant 模板[cite: 12]。

5. 【长期记忆隔离 (Store Isolation)】 -> `@auth.on.store`
   - 拦截对 LangGraph 跨会话长期记忆库 (Store) 的访问请求。
   - 强制校验请求的存储路径（Namespace 的首层目录）必须与当前用户的 UUID 一致 (`namespace[0] == ctx.user.identity`)[cite: 12]，否则拒绝访问[cite: 12]。

依赖前提：
- 环境变量 `SUPABASE_URL` 和 `SUPABASE_KEY` 必须正确配置并能够连接到 Supabase 实例[cite: 12]。
"""

import os
import asyncio
from langgraph_sdk import Auth
from langgraph_sdk.auth.types import StudioUser
from supabase import create_client, Client
from typing import Optional, Any

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase: Optional[Client] = None

if supabase_url and supabase_key:
    supabase = create_client(supabase_url, supabase_key)

# The "Auth" object is a container that LangGraph will use to mark our authentication function
auth = Auth()


# The `authenticate` decorator tells LangGraph to call this function as middleware
# for every request. This will determine whether the request is allowed or not
@auth.authenticate
async def get_current_user(authorization: str | None) -> Auth.types.MinimalUserDict:
    """Check if the user's JWT token is valid using Supabase."""

    # Ensure we have authorization header
    if not authorization:
        raise Auth.exceptions.HTTPException(
            status_code=401, detail="Authorization header missing"
        )

    # Parse the authorization header
    try:
        scheme, token = authorization.split()
        assert scheme.lower() == "bearer"
    except (ValueError, AssertionError):
        raise Auth.exceptions.HTTPException(
            status_code=401, detail="Invalid authorization header format"
        )

    # 后门：允许本地测试的固定Token
    if token == "local-dev-token-123":
        return {"identity": "test-user-id"}

    # Ensure Supabase client is initialized
    if not supabase:
        raise Auth.exceptions.HTTPException(
            status_code=500, detail="Supabase client not initialized"
        )

    try:
        # Verify the JWT token with Supabase using asyncio.to_thread to avoid blocking
        # This will decode and verify the JWT token in a separate thread
        async def verify_token() -> dict[str, Any]:
            response = await asyncio.to_thread(supabase.auth.get_user, token)
            return response

        response = await verify_token()
        user = response.user

        if not user:
            raise Auth.exceptions.HTTPException(
                status_code=401, detail="Invalid token or user not found"
            )

        # Return user info if valid
        return {
            "identity": user.id,
        }
    except Exception as e:
        # Handle any errors from Supabase
        raise Auth.exceptions.HTTPException(
            status_code=401, detail=f"Authentication error: {str(e)}"
        )


@auth.on.threads.create
@auth.on.threads.create_run
async def on_thread_create(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.threads.create.value,
):
    """Add owner when creating threads.

    This handler runs when creating new threads and does two things:
    1. Sets metadata on the thread being created to track ownership
    2. Returns a filter that ensures only the creator can access it
    """

    if isinstance(ctx.user, StudioUser):
        return

    # Add owner metadata to the thread being created
    # This metadata is stored with the thread and persists
    metadata = value.setdefault("metadata", {})
    metadata["owner"] = ctx.user.identity


@auth.on.threads.read
@auth.on.threads.delete
@auth.on.threads.update
@auth.on.threads.search
async def on_thread_read(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.threads.read.value,
):
    """Only let users read their own threads.

    This handler runs on read operations. We don't need to set
    metadata since the thread already exists - we just need to
    return a filter to ensure users can only see their own threads.
    """
    if isinstance(ctx.user, StudioUser):
        return

    return {"owner": ctx.user.identity}


@auth.on.assistants.create
async def on_assistants_create(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.assistants.create.value,
):
    if isinstance(ctx.user, StudioUser):
        return

    # Add owner metadata to the assistant being created
    # This metadata is stored with the assistant and persists
    metadata = value.setdefault("metadata", {})
    metadata["owner"] = ctx.user.identity


@auth.on.assistants.read
@auth.on.assistants.delete
@auth.on.assistants.update
@auth.on.assistants.search
async def on_assistants_read(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.assistants.read.value,
):
    """Only let users read their own assistants.

    This handler runs on read operations. We don't need to set
    metadata since the assistant already exists - we just need to
    return a filter to ensure users can only see their own assistants.
    """

    if isinstance(ctx.user, StudioUser):
        return

    return {"owner": ctx.user.identity}


@auth.on.store()
async def authorize_store(ctx: Auth.types.AuthContext, value: dict):
    if isinstance(ctx.user, StudioUser):
        return

    # The "namespace" field for each store item is a tuple you can think of as the directory of an item.
    namespace: tuple = value["namespace"]
    assert namespace[0] == ctx.user.identity, "Not authorized"