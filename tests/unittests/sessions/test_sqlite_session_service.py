# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sqlite3
import tempfile
import os

from google.adk.errors.version_mismatch_error import VersionMismatchError
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.sessions import SqliteSessionService
from google.adk.sessions.sqlite_session_service import _get_default_db_path
import pytest


@pytest.mark.asyncio
async def test_create_session():
    """测试创建 session 功能"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test.db")
        service = SqliteSessionService(db_path)
        
        app_name = "test_app"
        user_id = "test_user"
        state = {"key": "value"}
        
        session = await service.create_session(
            app_name=app_name, 
            user_id=user_id, 
            state=state
        )
        
        assert session.app_name == app_name
        assert session.user_id == user_id
        assert session.id is not None
        assert session.state == state
        
        assert os.path.exists(db_path)


@pytest.mark.asyncio
async def test_get_nonexistent_session_returns_none():
    """测试获取不存在的 session 返回 None"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test.db")
        service = SqliteSessionService(db_path)
        
        session = await service.get_session(
            app_name="nonexistent_app",
            user_id="nonexistent_user",
            session_id="nonexistent_id"
        )
        
        assert session is None


@pytest.mark.asyncio
async def test_append_event_and_restore():
    """测试 append event 后能正确恢复"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test.db")
        service = SqliteSessionService(db_path)
        
        session = await service.create_session(
            app_name="test_app",
            user_id="test_user"
        )
        
        event = Event(
            invocation_id="test_invocation",
            author="user",
            content=None
        )
        await service.append_event(session=session, event=event)
        
        restored_session = await service.get_session(
            app_name="test_app",
            user_id="test_user",
            session_id=session.id
        )
        
        assert restored_session is not None
        assert len(restored_session.events) == 1
        assert restored_session.events[0].invocation_id == "test_invocation"
        assert restored_session.events[0].author == "user"


@pytest.mark.asyncio
async def test_list_multiple_sessions():
    """测试 list 多个 session"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test.db")
        service = SqliteSessionService(db_path)
        
        app_name = "test_app"
        user_id = "test_user"
        
        session_ids = []
        for i in range(5):
            session = await service.create_session(
                app_name=app_name,
                user_id=user_id,
                session_id=f"session_{i}"
            )
            session_ids.append(session.id)
        
        response = await service.list_sessions(
            app_name=app_name,
            user_id=user_id
        )
        
        assert len(response.sessions) == 5
        assert {s.id for s in response.sessions} == set(session_ids)


@pytest.mark.asyncio
async def test_delete_session():
    """测试 delete 后无法再 get"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test.db")
        service = SqliteSessionService(db_path)
        
        session = await service.create_session(
            app_name="test_app",
            user_id="test_user"
        )
        
        assert await service.get_session(
            app_name="test_app",
            user_id="test_user",
            session_id=session.id
        ) is not None
        
        await service.delete_session(
            app_name="test_app",
            user_id="test_user",
            session_id=session.id
        )
        
        assert await service.get_session(
            app_name="test_app",
            user_id="test_user",
            session_id=session.id
        ) is None


@pytest.mark.asyncio
async def test_default_db_path():
    """测试默认数据库路径功能"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        service = SqliteSessionService(os.path.join(tmp_dir, "test.db"))
        assert service is not None
        
        nested_dir = os.path.join(tmp_dir, "nested", "dir", "test.db")
        service2 = SqliteSessionService(nested_dir)
        assert os.path.exists(os.path.dirname(nested_dir))


def test_default_path_env_override(monkeypatch):
    """测试 ADK_HOME 环境变量覆盖默认路径"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        adk_home = os.path.join(tmp_dir, "custom_adk")
        monkeypatch.setenv("ADK_HOME", adk_home)
        
        expected_path = os.path.join(adk_home, "sessions.db")
        actual_path = _get_default_db_path()
        
        assert actual_path == expected_path


def test_default_path_xdg_fallback(monkeypatch):
    """测试 XDG_DATA_HOME 环境变量作为备用路径"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        monkeypatch.delenv("ADK_HOME", raising=False)
        
        xdg_data_home = os.path.join(tmp_dir, "xdg_data")
        monkeypatch.setenv("XDG_DATA_HOME", xdg_data_home)
        
        expected_path = os.path.join(xdg_data_home, "adk", "sessions.db")
        actual_path = _get_default_db_path()
        
        assert actual_path == expected_path


def test_default_path_legacy_fallback(monkeypatch, tmp_path):
    """测试当 ~/.adk/sessions.db 存在时使用它"""
    import google.adk.sessions.sqlite_session_service as session_module
    
    original_expanduser = os.path.expanduser
    
    def mock_expanduser(path):
        if path == "~":
            return str(tmp_path)
        return original_expanduser(path)
    
    monkeypatch.setattr(os.path, "expanduser", mock_expanduser)
    
    legacy_dir = tmp_path / ".adk"
    legacy_dir.mkdir()
    legacy_db = legacy_dir / "sessions.db"
    legacy_db.touch()
    
    monkeypatch.delenv("ADK_HOME", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    
    actual_path = _get_default_db_path()
    
    assert actual_path == str(legacy_db)


def test_default_path_xdg_default(monkeypatch, tmp_path):
    """测试默认 XDG 路径 (~/.local/share/adk/sessions.db)"""
    import google.adk.sessions.sqlite_session_service as session_module
    
    original_expanduser = os.path.expanduser
    
    def mock_expanduser(path):
        if path == "~":
            return str(tmp_path)
        return original_expanduser(path)
    
    monkeypatch.setattr(os.path, "expanduser", mock_expanduser)
    
    monkeypatch.delenv("ADK_HOME", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    
    actual_path = _get_default_db_path()
    
    expected_path = str(tmp_path / ".local" / "share" / "adk" / "sessions.db")
    assert actual_path == expected_path


def test_schema_version_mismatch_raises():
    """测试 schema 版本不匹配时抛出 VersionMismatchError"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test.db")
        
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    app_name TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    create_time REAL NOT NULL,
                    update_time REAL NOT NULL,
                    PRIMARY KEY (app_name, user_id, id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT NOT NULL,
                    app_name TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    invocation_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    event_data TEXT NOT NULL,
                    PRIMARY KEY (app_name, user_id, session_id, id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
            
            cursor.execute(
                "INSERT INTO metadata (key, value) VALUES (?, ?)",
                ("schema_version", "999"),
            )
            conn.commit()
        
        with pytest.raises(VersionMismatchError) as exc_info:
            SqliteSessionService(db_path)
        
        assert "999" in str(exc_info.value)
        assert exc_info.value.expected_version == 1
        assert exc_info.value.actual_version == 999


@pytest.mark.asyncio
async def test_schema_version_initialized_on_new_db():
    """测试新数据库初始化时 schema 版本被正确设置"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test.db")
        
        service = SqliteSessionService(db_path)
        await service.create_session(app_name="test_app", user_id="test_user")
        
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT value FROM metadata WHERE key = ?",
                ("schema_version",),
            )
            result = cursor.fetchone()
            
            assert result is not None
            assert int(result[0]) == 1


@pytest.mark.asyncio
async def test_state_merge_shallow_vs_recursive_semantics_documented():
    """测试 SqliteSessionService 使用递归 merge (json_patch) vs DatabaseSessionService 使用浅 merge
    
    这个测试验证 SqliteSessionService 的 state merge 语义：
    - SqliteSessionService: 使用 SQLite json_patch (RFC 7396) - 递归 merge
    - DatabaseSessionService: 使用 dict.update() - 浅 merge
    
    例如：
    - 现有 state: {"nested": {"a": 1, "b": 2}}
    - State delta: {"nested": {"b": 3, "c": 4}}
    - SqliteSessionService 结果: {"nested": {"a": 1, "b": 3, "c": 4}} (递归 merge)
    - DatabaseSessionService 结果: {"nested": {"b": 3, "c": 4}} (浅 merge)
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test.db")
        service = SqliteSessionService(db_path)
        
        session = await service.create_session(
            app_name="test_app",
            user_id="test_user",
            state={"nested": {"a": 1, "b": 2}}
        )
        
        assert session.state == {"nested": {"a": 1, "b": 2}}
        
        event = Event(
            invocation_id="test_invocation",
            author="user",
            actions=EventActions(
                state_delta={"nested": {"b": 3, "c": 4}}
            )
        )
        await service.append_event(session=session, event=event)
        
        restored_session = await service.get_session(
            app_name="test_app",
            user_id="test_user",
            session_id=session.id
        )
        
        assert restored_session is not None
        assert restored_session.state == {"nested": {"a": 1, "b": 3, "c": 4}}
