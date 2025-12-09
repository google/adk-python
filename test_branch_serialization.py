"""Test BranchContext serialization with SQLite session service."""
import asyncio
from google.adk.agents.branch import Branch
from google.adk.events.event import Event
from google.adk.sessions.sqlite_session_service import SqliteSessionService
from google.genai.types import Content, Part
import os
import json

async def test_serialization():
    # Create a test database
    db_path = "test_branch_serialization.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # Create session service
    session_service = SqliteSessionService(db_path=db_path)
    
    # Create a session
    session = await session_service.create_session(
        app_name="test_app",
        user_id="test_user"
    )
    
    # Create events with BranchContext
    branch1 = Branch(tokens=frozenset([1, 2, 3]))
    branch2 = Branch(tokens=frozenset([4, 5]))
    
    event1 = Event(
        author="agent1",
        invocation_id="inv1",
        branch=branch1,
        content=Content(parts=[Part(text="Test message 1")])
    )
    
    event2 = Event(
        author="agent2",
        invocation_id="inv1",
        branch=branch2,
        content=Content(parts=[Part(text="Test message 2")])
    )
    
    # Append events
    await session_service.append_event(session, event1)
    await session_service.append_event(session, event2)
    
    # Retrieve session
    retrieved_session = await session_service.get_session(
        app_name="test_app",
        user_id="test_user",
        session_id=session.id
    )
    
    print("\n" + "="*80)
    print("SERIALIZATION TEST RESULTS")
    print("="*80)
    
    for i, event in enumerate(retrieved_session.events):
        print(f"\nEvent {i+1}:")
        print(f"  Author: {event.author}")
        print(f"  Branch type: {type(event.branch)}")
        print(f"  Branch value: {event.branch}")
        if isinstance(event.branch, Branch):
            print(f"  Tokens: {event.branch.tokens}")
            print(f"  Tokens type: {type(event.branch.tokens)}")
        else:
            print(f"  ERROR: Branch is not a BranchContext!")
    
    # Check raw database
    import sqlite3
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, author, branch FROM events")
    print("\n" + "="*80)
    print("RAW DATABASE CONTENT")
    print("="*80)
    for row in cursor.fetchall():
        event_id, author, branch_json = row
        print(f"\nEvent ID: {event_id}")
        print(f"Author: {author}")
        print(f"Branch JSON: {branch_json}")
        if branch_json:
            parsed = json.loads(branch_json)
            print(f"Parsed: {parsed}")
    conn.close()
    
    # Cleanup
    os.remove(db_path)
    print("\n" + "="*80)

if __name__ == "__main__":
    asyncio.run(test_serialization())
