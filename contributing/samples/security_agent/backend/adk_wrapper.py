import asyncio
import logging
import time
import traceback
from typing import Dict, Any
from pathlib import Path
from agents.adk_agent import root_agent as security_agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Database validation and performance monitoring imports
try:
    import sys
    from pathlib import Path
    # Add parent directory to sys.path for imports
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))
    from agents.tools.sqlite_tool import sqlite_tool_instance
    from backend.utils.performance import performance_monitor, log_agent_metrics
    DATABASE_VALIDATION_AVAILABLE = True
    PERFORMANCE_MONITORING_AVAILABLE = True
except ImportError:
    DATABASE_VALIDATION_AVAILABLE = False
    PERFORMANCE_MONITORING_AVAILABLE = False

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create dedicated logger for ADK operations
adk_logger = logging.getLogger("adk_operations")
adk_logger.setLevel(logging.DEBUG)

class ADKAgentWrapper:
    """
    A wrapper class to interact with the ADK security agent.
    """
    _runner = None
    _session_service = None

    @classmethod
    def _initialize(cls):
        """
        Initializes the runner and session service if they haven't been already.
        Includes database validation.
        """
        try:
            adk_logger.info("🔧 Starting ADK initialization...")

            # Validate database connection on startup
            if DATABASE_VALIDATION_AVAILABLE:
                cls._validate_database()

            if cls._session_service is None:
                adk_logger.info("📋 Creating InMemorySessionService instance...")
                cls._session_service = InMemorySessionService()
                adk_logger.info("✅ InMemorySessionService created successfully")

            if cls._runner is None and security_agent:
                adk_logger.info("🚀 Creating ADK Runner instance...")
                adk_logger.debug(f"Agent type: {type(security_agent)}")
                adk_logger.debug(f"Session service type: {type(cls._session_service)}")

                cls._runner = Runner(
                    agent=security_agent,
                    session_service=cls._session_service,
                    app_name="security_agent_test"
                )
                adk_logger.info("✅ ADK Runner created successfully")
            elif not security_agent:
                adk_logger.error("❌ Security agent is None - cannot initialize runner")

        except Exception as e:
            adk_logger.error(f"❌ ADK initialization failed: {e}")
            adk_logger.error(f"Error details: {traceback.format_exc()}")
            cls._runner = None
            cls._session_service = None

    @classmethod
    def _validate_database(cls):
        """Validate database connection and create tables if needed."""
        try:
            adk_logger.info("🗄️ Validating database connection...")

            # Check if database file exists
            db_path = sqlite_tool_instance.db_path
            if not Path(db_path).exists():
                adk_logger.warning(f"⚠️ Database file not found: {db_path}")
                adk_logger.info("📁 Creating empty database with basic tables...")
                sqlite_tool_instance._create_empty_database()

            # Test database connection
            tables = sqlite_tool_instance.get_tables()
            adk_logger.info(f"✅ Database connection validated. Tables found: {len(tables)}")
            adk_logger.debug(f"Available tables: {tables}")

            if not tables:
                adk_logger.warning("⚠️ No tables found in database - some queries may fail")
                adk_logger.info("💡 Run 'python populate_sqlite.py' to populate the database")

        except Exception as e:
            adk_logger.error(f"❌ Database validation failed: {e}")
            adk_logger.warning("⚠️ Continuing without database validation...")

    @classmethod
    async def query_agent(cls, message: str, session_id: str = "test_session", user_id: str = "test_user") -> Dict[str, Any]:
        """
        Sends a query to the ADK agent and returns the response.
        """
        start_time = time.time()

        adk_logger.info("=" * 80)
        adk_logger.info(f"🔍 NEW ADK QUERY STARTED")
        adk_logger.info(f"  📝 Message: '{message[:100]}{'...' if len(message) > 100 else ''}'")
        adk_logger.info(f"  👤 User ID: {user_id}")
        adk_logger.info(f"  🎭 Session ID: {session_id}")
        adk_logger.info(f"  ⏰ Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        adk_logger.info("=" * 80)

        cls._initialize()
        if not cls._runner:
            error_msg = "❌ ADK Runner not initialized - cannot process query"
            adk_logger.error(error_msg)
            return {"error": error_msg}

        app_name = "security_agent_test"
        adk_logger.info(f"🏠 App Name: {app_name}")

        # Create session using correct ADK API
        try:
            adk_logger.info(f"🆕 Creating session: {session_id}")
            session = cls._session_service.create_session_sync(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                state={}
            )
            adk_logger.info(f"✅ Session created successfully")
        except Exception as e:
            # Session errors are often not critical - continue with query
            adk_logger.warning(f"⚠️ Session creation warning: {e}")
            adk_logger.info("🔄 Continuing with query despite session warning...")

        # Prepare content for ADK
        try:
            adk_logger.info(f"📦 Preparing content for ADK...")
            content = types.Content(parts=[types.Part(text=message)])
            adk_logger.debug(f"Content structure: {type(content)}")
            adk_logger.debug(f"Parts count: {len(content.parts)}")
        except Exception as e:
            error_msg = f"❌ Error creating content: {e}"
            adk_logger.error(error_msg)
            return {"error": error_msg}

        response_text = ""
        tool_calls = []

        try:
            adk_logger.info(f"🚀 Sending message to ADK agent...")
            query_start = time.time()

            events = cls._runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=content
            )

            event_count = 0
            part_count = 0

            adk_logger.info(f"📨 Processing ADK response events...")

            # Add timeout handling for query processing
            timeout_seconds = 30  # 30 second timeout

            try:
                async for event in asyncio.wait_for(events, timeout=timeout_seconds):
                    event_count += 1
                    event_type = type(event).__name__

                    adk_logger.debug(f"📬 Event #{event_count}: {event_type}")
            except asyncio.TimeoutError:
                adk_logger.error(f"⏰ Query timeout after {timeout_seconds} seconds")
                return {
                    "error": f"Query timed out after {timeout_seconds} seconds",
                    "timeout": True
                }
            except Exception as event_error:
                # Continue processing for non-timeout errors
                adk_logger.warning(f"⚠️ Event processing warning: {event_error}")
                # Re-iterate without timeout for remaining events
                async for event in events:

                    # Log tool calls if present
                    if hasattr(event, 'tool_calls') and event.tool_calls:
                        for tool_call in event.tool_calls:
                            tool_info = {
                                "name": getattr(tool_call, 'name', 'unknown'),
                                "function": getattr(tool_call, 'function', 'unknown'),
                                "args": getattr(tool_call, 'args', {})
                            }
                            tool_calls.append(tool_info)
                            adk_logger.info(f"🔧 Tool call detected: {tool_info['name']}")
                            adk_logger.debug(f"   Function: {tool_info['function']}")
                            adk_logger.debug(f"   Args: {tool_info['args']}")

                    # Extract text content
                    if hasattr(event, 'content') and event.content and hasattr(event.content, 'parts'):
                        for part in event.content.parts:
                            part_count += 1
                            if hasattr(part, 'text'):
                                text_length = len(part.text)
                                response_text += part.text
                                adk_logger.debug(f"📄 Part #{part_count}: {text_length} chars")

                    elif hasattr(event, 'text'):
                        text_length = len(event.text)
                        response_text += event.text
                        adk_logger.debug(f"📄 Direct text: {text_length} chars")

            query_duration = time.time() - query_start
            total_duration = time.time() - start_time

            adk_logger.info("=" * 80)
            adk_logger.info(f"✅ ADK QUERY COMPLETED SUCCESSFULLY")
            adk_logger.info(f"  📊 Events processed: {event_count}")
            adk_logger.info(f"  📄 Text parts: {part_count}")
            adk_logger.info(f"  🔧 Tool calls: {len(tool_calls)}")
            adk_logger.info(f"  📝 Response length: {len(response_text)} chars")
            adk_logger.info(f"  ⏱️  Query duration: {query_duration:.2f}s")
            adk_logger.info(f"  ⏰ Total duration: {total_duration:.2f}s")
            adk_logger.info("=" * 80)

        except Exception as e:
            error_duration = time.time() - start_time
            error_msg = f"❌ ADK query error: {e}"

            adk_logger.error("=" * 80)
            adk_logger.error(f"❌ ADK QUERY FAILED")
            adk_logger.error(f"  Error: {error_msg}")
            adk_logger.error(f"  Duration: {error_duration:.2f}s")
            adk_logger.error(f"  Exception type: {type(e).__name__}")
            adk_logger.error("=" * 80)
            adk_logger.error(f"Full traceback: {traceback.format_exc()}")

            return {"error": error_msg}

        # Provide graceful fallback for empty responses
        if not response_text and not tool_calls:
            adk_logger.warning("⚠️ Empty response from ADK agent")
            response_text = "I apologize, but I couldn't process your request. Please ensure the database is populated and try again."

        # Log performance metrics
        if PERFORMANCE_MONITORING_AVAILABLE:
            try:
                log_agent_metrics(
                    session_id=session_id,
                    user_id=user_id,
                    query_text=message,
                    metadata={
                        "event_count": event_count,
                        "part_count": part_count,
                        "tool_calls": tool_calls,
                        "query_duration": query_duration,
                        "total_duration": total_duration,
                        "response_length": len(response_text)
                    }
                )
            except Exception as e:
                adk_logger.warning(f"Performance logging failed: {e}")

        return {
            "response": response_text,
            "metadata": {
                "event_count": event_count,
                "part_count": part_count,
                "tool_calls": tool_calls,
                "query_duration": query_duration,
                "total_duration": total_duration,
                "response_length": len(response_text),
                "database_available": DATABASE_VALIDATION_AVAILABLE,
                "performance_monitoring": PERFORMANCE_MONITORING_AVAILABLE
            }
        }

    @classmethod
    async def cleanup(cls):
        """
        Cleans up the runner and session service.
        """
        adk_logger.info("🧹 Starting ADK cleanup...")

        if cls._runner:
            try:
                adk_logger.info("🏃 Closing ADK Runner...")
                await cls._runner.close()
                cls._runner = None
                adk_logger.info("✅ ADK Runner closed successfully")
            except Exception as e:
                adk_logger.error(f"❌ Error closing ADK Runner: {e}")
                adk_logger.error(f"Cleanup error details: {traceback.format_exc()}")

        if cls._session_service:
            try:
                adk_logger.info("📋 Cleaning up session service...")
                cls._session_service = None
                adk_logger.info("✅ Session service cleaned up")
            except Exception as e:
                adk_logger.error(f"❌ Error cleaning session service: {e}")

        adk_logger.info("🧹 ADK cleanup completed")
