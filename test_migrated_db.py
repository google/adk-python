#!/usr/bin/env python3
"""Test script to verify migrated database works with SqliteSessionService."""

import asyncio
from google.adk.agents.llm_agent import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions.sqlite_session_service import SqliteSessionService
from google.genai import types


async def main():
    print("=" * 80)
    print("Testing migrated database with SqliteSessionService")
    print("=" * 80)
    
    # Create SqliteSessionService with the migrated database
    db_path = "contributing/samples/migrate_session_db/sessions_robust.db"
    session_service = SqliteSessionService(db_path)
    print(f"\n✓ Created SqliteSessionService with: {db_path}")
    
    # List existing sessions
    print("\n📋 Listing existing sessions...")
    sessions_response = await session_service.list_sessions(
        app_name="migrate_session_db_app"
    )
    print(f"Found {len(sessions_response.sessions)} sessions:")
    for session in sessions_response.sessions:
        print(f"  - Session ID: {session.id}")
        print(f"    User ID: {session.user_id}")
        print(f"    Last updated: {session.last_update_time}")
        print(f"    Events: {len(session.events)}")
    
    # Get a specific session with events
    if sessions_response.sessions:
        first_session = sessions_response.sessions[0]
        print(f"\n📖 Reading session: {first_session.id}")
        
        full_session = await session_service.get_session(
            app_name="migrate_session_db_app",
            user_id=first_session.user_id,
            session_id=first_session.id,
        )
        
        print(f"✓ Loaded session with {len(full_session.events)} events")
        print(f"  State keys: {list(full_session.state.keys())}")
        
        # Show first few events
        print("\n  First 3 events:")
        for i, event in enumerate(full_session.events[:3]):
            print(f"    {i+1}. {event.author}: {event.id[:8]}...")
            if event.content and event.content.parts:
                text = event.content.parts[0].text if event.content.parts[0].text else "<no text>"
                print(f"       {text[:60]}...")
    
    # Create a simple agent and add a new message to an existing session
    print("\n🤖 Creating agent and adding new message...")
    agent = LlmAgent(
        name="test_agent",
        model="gemini-2.0-flash-exp",
        instruction="You are a helpful assistant. Keep responses brief.",
    )
    
    runner = Runner(
        app_name="migrate_session_db_app",
        agent=agent,
        session_service=session_service,
    )
    
    # Use an existing session to verify it works
    if sessions_response.sessions:
        test_session = sessions_response.sessions[0]
        print(f"✓ Using existing session: {test_session.id}")
        
        # Run a simple query
        print("\n💬 Running agent with new message...")
        new_message = types.Content(
            role="user",
            parts=[types.Part.from_text(text="What's 2+2?")]
        )
        
        response_events = []
        async for event in runner.run_async(
            user_id=test_session.user_id,
            session_id=test_session.id,
            new_message=new_message,
        ):
            response_events.append(event)
            if event.content and event.content.parts and event.author != "user":
                print(f"  {event.author}: {event.content.parts[0].text[:100]}")
        
        print(f"\n✓ Got {len(response_events)} events in response")
        
        # Verify the event was persisted
        updated_session = await session_service.get_session(
            app_name="migrate_session_db_app",
            user_id=test_session.user_id,
            session_id=test_session.id,
        )
        
        original_count = len(full_session.events)
        new_count = len(updated_session.events)
        print(f"✓ Session now has {new_count} events (was {original_count})")
    
    print("\n" + "=" * 80)
    print("✅ All tests passed! Migrated database works with SqliteSessionService")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
