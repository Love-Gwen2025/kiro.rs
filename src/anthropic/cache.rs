//! Prompt Caching 模块 - 使用 Redis 实现前缀哈希匹配
//!
//! 模拟 Anthropic 的 cache_control 行为：
//! - Redis 只存标记（是否见过该前缀），不存 token 数
//! - token 数实时计算
//! - API key 哈希后存储，防止泄露

use redis::aio::ConnectionManager;
use redis::AsyncCommands;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::sync::OnceLock;

use crate::anthropic::types::{Message, SystemMessage, Tool};
use crate::token;

/// 全局 Redis 连接管理器
static REDIS_CONN: OnceLock<ConnectionManager> = OnceLock::new();

/// 默认 TTL: 5 分钟
const DEFAULT_TTL_SECS: u64 = 5 * 60;

/// 缓存断点信息
#[derive(Debug, Clone)]
pub struct CacheBreakpoint {
    /// 累积哈希值（从请求开头到此断点）
    pub hash: String,
    /// 到此断点的累积 token 数
    pub tokens: i32,
}

/// 缓存查询结果
#[derive(Debug, Clone, Default)]
pub struct CacheResult {
    pub cache_read_input_tokens: i32,
    pub cache_creation_input_tokens: i32,
}

/// 初始化 Redis 连接
pub async fn init_redis(redis_url: &str) -> anyhow::Result<()> {
    let client = redis::Client::open(redis_url)?;
    let conn = ConnectionManager::new(client).await?;
    REDIS_CONN
        .set(conn)
        .map_err(|_| anyhow::anyhow!("Redis already initialized"))?;
    tracing::info!("Redis cache initialized: {}", redis_url);
    Ok(())
}

/// 检查 Redis 是否可用
pub fn is_redis_available() -> bool {
    REDIS_CONN.get().is_some()
}

/// 对 API key 做 SHA256 哈希，避免明文存储
fn hash_api_key(api_key: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(api_key.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// 递归排序 JSON 对象的 key，确保序列化稳定
fn sort_json_value(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let sorted: BTreeMap<String, serde_json::Value> = map
                .iter()
                .map(|(k, v)| (k.clone(), sort_json_value(v)))
                .collect();
            serde_json::Value::Object(sorted.into_iter().collect())
        }
        serde_json::Value::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(sort_json_value).collect())
        }
        _ => value.clone(),
    }
}

/// 规范化 Tool 为稳定的字符串表示
fn normalize_tool(tool: &Tool) -> String {
    let mut parts = Vec::new();
    parts.push(format!("name={}", tool.name));
    if !tool.description.is_empty() {
        parts.push(format!("desc={}", tool.description));
    }
    if !tool.input_schema.is_empty() {
        let schema_value = serde_json::to_value(&tool.input_schema).unwrap_or_default();
        let sorted = sort_json_value(&schema_value);
        if let Ok(s) = serde_json::to_string(&sorted) {
            parts.push(format!("schema={}", s));
        }
    }
    // 使用 \x00 作为分隔符，不可能出现在正常内容中
    parts.join("\x00")
}

/// 计算请求的缓存断点
///
/// 按 Anthropic 的顺序遍历 tools → system → messages，
/// 遇到 cache_control 标记时记录当前累积哈希和 token 数。
///
/// 自动注入策略：当客户端未标记 cache_control 时，自动在以下位置注入断点：
/// 1. 所有 tools 处理完毕后（如果没有任何 tool 标记了 cache_control）
/// 2. 所有 system 处理完毕后（如果没有任何 system 标记了 cache_control）
/// 3. 倒数第二条 message 的最后一个 content block（对话历史前缀）
pub fn compute_cache_breakpoints(
    tools: &Option<Vec<Tool>>,
    system: &Option<Vec<SystemMessage>>,
    messages: &[Message],
) -> Vec<CacheBreakpoint> {
    let mut hasher = Sha256::new();
    let mut breakpoints = Vec::new();
    let mut cumulative_tokens: i32 = 0;

    // 1. 处理 tools（按 name 排序确保顺序稳定）
    let mut tools_has_cache_control = false;
    if let Some(tools) = tools {
        let mut sorted_tools: Vec<&Tool> = tools.iter().collect();
        sorted_tools.sort_by(|a, b| a.name.cmp(&b.name));

        for tool in &sorted_tools {
            let normalized = normalize_tool(tool);
            hasher.update(normalized.as_bytes());
            cumulative_tokens += token::count_tokens(&normalized) as i32;

            if tool.cache_control.is_some() {
                tools_has_cache_control = true;
                breakpoints.push(CacheBreakpoint {
                    hash: format!("{:x}", hasher.clone().finalize()),
                    tokens: cumulative_tokens,
                });
            }
        }

        // 自动注入：tools 全部处理完但没有任何 cache_control 标记
        if !tools_has_cache_control && !sorted_tools.is_empty() {
            breakpoints.push(CacheBreakpoint {
                hash: format!("{:x}", hasher.clone().finalize()),
                tokens: cumulative_tokens,
            });
            tracing::debug!("Auto-injected breakpoint after tools, tokens={}", cumulative_tokens);
        }
    }

    // 2. 处理 system
    let mut system_has_cache_control = false;
    if let Some(system) = system {
        for msg in system {
            hasher.update(msg.text.as_bytes());
            cumulative_tokens += token::count_tokens(&msg.text) as i32;

            if msg.cache_control.is_some() {
                system_has_cache_control = true;
                breakpoints.push(CacheBreakpoint {
                    hash: format!("{:x}", hasher.clone().finalize()),
                    tokens: cumulative_tokens,
                });
            }
        }

        // 自动注入：system 全部处理完但没有任何 cache_control 标记
        if !system_has_cache_control && !system.is_empty() {
            breakpoints.push(CacheBreakpoint {
                hash: format!("{:x}", hasher.clone().finalize()),
                tokens: cumulative_tokens,
            });
            tracing::debug!("Auto-injected breakpoint after system, tokens={}", cumulative_tokens);
        }
    }

    // 3. 处理 messages
    // 自动注入策略：在倒数第二条 message 的最后一个 block 后注入断点
    // 这样对话历史前缀（除最后一条消息外）都能被缓存覆盖
    let second_to_last_idx = if messages.len() >= 2 {
        Some(messages.len() - 2)
    } else {
        None
    };

    for (msg_idx, msg) in messages.iter().enumerate() {
        if let Some(blocks) = msg.content.as_array() {
            let block_count = blocks.len();
            for (block_idx, block) in blocks.iter().enumerate() {
                let sorted_block = sort_json_value(block);
                let block_json = serde_json::to_string(&sorted_block).unwrap_or_default();
                hasher.update(block_json.as_bytes());

                // 根据 block 类型计算 tokens
                cumulative_tokens += count_block_tokens(block) as i32;

                if block.get("cache_control").is_some() {
                    breakpoints.push(CacheBreakpoint {
                        hash: format!("{:x}", hasher.clone().finalize()),
                        tokens: cumulative_tokens,
                    });
                }

                // 自动注入：倒数第二条 message 的最后一个 block
                let is_last_block = block_idx == block_count - 1;
                if second_to_last_idx == Some(msg_idx)
                    && is_last_block
                    && block.get("cache_control").is_none()
                {
                    breakpoints.push(CacheBreakpoint {
                        hash: format!("{:x}", hasher.clone().finalize()),
                        tokens: cumulative_tokens,
                    });
                    tracing::debug!(
                        "Auto-injected breakpoint after message[{}], tokens={}",
                        msg_idx,
                        cumulative_tokens
                    );
                }
            }
        } else if let Some(text) = msg.content.as_str() {
            hasher.update(text.as_bytes());
            cumulative_tokens += token::count_tokens(text) as i32;

            // 自动注入：倒数第二条 message（纯文本形式）
            if second_to_last_idx == Some(msg_idx) {
                breakpoints.push(CacheBreakpoint {
                    hash: format!("{:x}", hasher.clone().finalize()),
                    tokens: cumulative_tokens,
                });
                tracing::debug!(
                    "Auto-injected breakpoint after message[{}] (text), tokens={}",
                    msg_idx,
                    cumulative_tokens
                );
            }
        }
    }

    tracing::debug!("Cache breakpoints: count={}", breakpoints.len());
    breakpoints
}

/// 查询或创建缓存
///
/// Redis 只存标记（值为 "1"），不存 token 数。
/// 从最后一个断点向前查找，命中则返回 cache_read，未命中则创建标记返回 cache_creation。
pub async fn lookup_or_create(
    api_key: &str,
    breakpoints: &[CacheBreakpoint],
) -> CacheResult {
    let Some(conn) = REDIS_CONN.get() else {
        return CacheResult::default();
    };

    if breakpoints.is_empty() {
        return CacheResult::default();
    }

    let mut conn = conn.clone();
    let key_prefix = hash_api_key(api_key);

    // 从最后一个断点向前查找
    for (i, bp) in breakpoints.iter().enumerate().rev() {
        let key = format!("cache:{}:{}", key_prefix, bp.hash);

        let exists: bool = conn.exists(&key).await.unwrap_or(false);

        if exists {
            // 命中：刷新 TTL
            tracing::debug!("Cache hit: breakpoint {}, tokens={}", i, bp.tokens);
            let _ = conn.expire::<_, ()>(&key, DEFAULT_TTL_SECS as i64).await;

            // 为命中断点之后的断点创建标记
            for later_bp in breakpoints.iter().skip(i + 1) {
                let later_key = format!("cache:{}:{}", key_prefix, later_bp.hash);
                let _ = conn
                    .set_ex::<_, _, ()>(&later_key, "1", DEFAULT_TTL_SECS)
                    .await;
            }

            // 命中的断点覆盖的 token 数 = cache_read
            // 命中之后的断点增量 = cache_creation
            let last_bp = breakpoints.last().unwrap();
            return CacheResult {
                cache_read_input_tokens: bp.tokens,
                cache_creation_input_tokens: last_bp.tokens - bp.tokens,
            };
        }
    }

    // 完全未命中：创建所有断点的标记
    tracing::debug!("Cache miss: creating {} breakpoints", breakpoints.len());
    for bp in breakpoints {
        let key = format!("cache:{}:{}", key_prefix, bp.hash);
        let _ = conn
            .set_ex::<_, _, ()>(&key, "1", DEFAULT_TTL_SECS)
            .await;
    }

    let last_bp = breakpoints.last().unwrap();
    CacheResult {
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: last_bp.tokens,
    }
}

/// 计算单个 content block 的 token 数
///
/// 支持所有 block 类型：text、tool_use、tool_result 等
fn count_block_tokens(block: &serde_json::Value) -> u64 {
    let block_type = block.get("type").and_then(|v| v.as_str()).unwrap_or("");

    match block_type {
        "text" => {
            if let Some(text) = block.get("text").and_then(|v| v.as_str()) {
                token::count_tokens(text)
            } else {
                0
            }
        }
        "tool_use" => {
            let mut tokens = 0u64;
            if let Some(name) = block.get("name").and_then(|v| v.as_str()) {
                tokens += token::count_tokens(name);
            }
            if let Some(input) = block.get("input") {
                let input_str = serde_json::to_string(input).unwrap_or_default();
                tokens += token::count_tokens(&input_str);
            }
            tokens
        }
        "tool_result" => {
            if let Some(content) = block.get("content") {
                if let Some(s) = content.as_str() {
                    token::count_tokens(s)
                } else if let Some(arr) = content.as_array() {
                    arr.iter().map(count_block_tokens).sum()
                } else {
                    0
                }
            } else {
                0
            }
        }
        _ => {
            // 未知类型：用序列化 JSON 估算
            let json_str = serde_json::to_string(block).unwrap_or_default();
            token::count_tokens(&json_str)
        }
    }
}
