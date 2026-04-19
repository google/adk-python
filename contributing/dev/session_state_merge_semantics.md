# Session State Merge Semantics Bug Analysis

## Problem Statement

ADK Python 目前有三种 `BaseSessionService` 实现，但它们对 `state_delta` 的合并语义**不一致**：

| Implementation | Merge Mechanism | Behavior |
|----------------|-----------------|----------|
| `InMemorySessionService` | `dict.update()` | **Shallow merge** (顶层 key 合并) |
| `DatabaseSessionService` | `dict \| dict` operator | **Shallow merge** (顶层 key 合并) |
| `SqliteSessionService` | SQLite `json_patch()` | **Recursive merge** (RFC 7396) |

这是一个明显的 **接口实现不一致 Bug**，违反了 `BaseSessionService` 的行为契约。

---

## Detailed Behavior Comparison

### Test Case

假设初始状态：
```python
session.state = {
    "top_key": "value1",
    "nested": {
        "inner_a": 1,
        "inner_b": 2
    }
}
```

使用以下 `state_delta` 进行更新：
```python
state_delta = {
    "nested": {
        "inner_a": 100,
        "inner_c": 300
    },
    "new_key": "added"
}
```

### Expected Results

#### 1. InMemorySessionService (Shallow Merge)

**Code Location**: `src/google/adk/sessions/in_memory_session_service.py:362`
```python
if session_state_delta:
    storage_session.state.update(session_state_delta)
```

**Result**:
```python
{
    "top_key": "value1",           # 保留
    "nested": {                     # 完全替换!
        "inner_a": 100,
        "inner_c": 300
    },
    "new_key": "added"              # 新增
}
```

**注意**: `inner_b` 丢失了，因为 `nested` dict 被整体替换。

#### 2. DatabaseSessionService (Shallow Merge)

**Code Location**: `src/google/adk/sessions/database_session_service.py:727-729`
```python
storage_session.state = (
    storage_session.state | state_deltas["session"]
)
```

**Result** (与 InMemory 相同):
```python
{
    "top_key": "value1",
    "nested": {
        "inner_a": 100,
        "inner_c": 300
    },
    "new_key": "added"
}
```

#### 3. SqliteSessionService (Recursive Merge)

**Code Location**: `src/google/adk/sessions/sqlite_session_service.py:562-564`
```python
"UPDATE sessions SET state=json_patch(state, ?), update_time=? WHERE"
" app_name=? AND user_id=? AND id=?",
(
    json.dumps(delta),  # delta = {"nested": {"inner_a": 100, "inner_c": 300}, "new_key": "added"}
    now,
    app_name,
    user_id,
    session_id,
),
```

SQLite 的 `json_patch()` 实现的是 **[RFC 7396 JSON Merge Patch](https://datatracker.ietf.org/doc/html/rfc7396)**。

**RFC 7396 规则**:
1. 如果 patch 值为 `null`，从 target 删除该 key
2. 如果 patch 值是 object **且** target 对应值也是 object → **递归合并**
3. 否则 → 直接替换

**Result**:
```python
{
    "top_key": "value1",           # 保留
    "nested": {                     # 递归合并!
        "inner_a": 100,            # 更新
        "inner_b": 2,              # 保留!
        "inner_c": 300             # 新增
    },
    "new_key": "added"              # 新增
}
```

**关键差异**: `inner_b` 被保留了，因为 `nested` dict 是递归合并而非替换。

---

## Impact Analysis

### 1. Functional Impact

| Scenario | InMemory/Database | Sqlite |
|----------|-------------------|--------|
| 简单值更新 | 一致 | 一致 |
| 新增顶层 key | 一致 | 一致 |
| 嵌套 dict 部分更新 | **丢失其他嵌套 key** | **保留其他嵌套 key** |
| 使用 `null` 删除 key | 不支持 (变为值为 None) | RFC 7396 支持 |

### 2. Developer Experience

开发者写的代码在不同存储后端行为不一致：

```python
# 开发者意图：只更新 nested.inner_a，不影响 nested.inner_b
event = Event(
    actions=EventActions(
        state_delta={"nested": {"inner_a": "new_value"}}
    )
)
await session_service.append_event(session, event)

# 实际结果:
# - InMemory/Database: nested = {"inner_a": "new_value"}  (inner_b 丢失!)
# - Sqlite: nested = {"inner_a": "new_value", "inner_b": 2}  (inner_b 保留)
```

---

## Recommended Solution

### Option A: Standardize on RFC 7396 (Recursive Merge)

**推荐方案**。RFC 7396 是业界标准，语义更直观。

**需要修改**:
- `InMemorySessionService`: 将 `dict.update()` 改为递归合并
- `DatabaseSessionService`: 将 `dict | dict` 改为递归合并

**Pros**:
- 符合业界标准 (JSON Merge Patch)
- 语义更符合开发者直觉 ("部分更新" 应该只更新提供的字段)
- Sqlite 已经是此行为，改动最小

**Cons**:
- 需要修改两个实现
- 可能影响依赖当前浅合并语义的现有代码

### Option B: Standardize on Shallow Merge

**不推荐**。浅合并语义不直观，且需要修改 Sqlite (可能更复杂)。

**需要修改**:
- `SqliteSessionService`: 不再使用 `json_patch()`，改为序列化后再浅合并

**Pros**:
- InMemory/Database 已经是此行为

**Cons**:
- Sqlite 改动复杂 (需要放弃原生 json_patch)
- 语义不直观 ("部分更新" 变成 "替换整个嵌套结构")

---

## Implementation Plan (Option A)

### Phase 1: Add Documentation and XFail Tests

- [x] 创建此文档 (`contributing/dev/session_state_merge_semantics.md`)
- [x] 在 `tests/sessions/test_session_integration.py` 添加 xfail 测试
- [ ] 更新 `BaseSessionService` docstring 注明当前不一致状态

### Phase 2: Implement Recursive Merge

修改 `BaseSessionService._update_session_state()` 或各实现：

```python
# 参考实现: RFC 7396 JSON Merge Patch
def rfc7396_merge(target: dict, patch: dict) -> dict:
    """Apply RFC 7396 JSON Merge Patch.
    
    https://datatracker.ietf.org/doc/html/rfc7396
    """
    result = copy.deepcopy(target)
    for key, value in patch.items():
        if value is None:
            result.pop(key, None)
        elif isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = rfc7396_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result
```

### Phase 3: Migration

1. 更新 CHANGELOG 说明此 Breaking Change
2. 提供迁移指南: 如果依赖浅合并语义，需要修改代码
3. 将 xfail 测试改为正常测试

---

## References

- [RFC 7396 - JSON Merge Patch](https://datatracker.ietf.org/doc/html/rfc7396)
- [SQLite json_patch() Documentation](https://www.sqlite.org/json1.html#jpatch)
- [Python dict.update() Documentation](https://docs.python.org/3/library/stdtypes.html#dict.update)
