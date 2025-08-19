# Session Management API Documentation

## Overview

The Session Management API provides persistent conversation storage using SQLite, enabling context retention across interactions. This API is part of STORY-013: Session Management Service.

**Base URL**: `/api/v1/sessions`

## Endpoints

### 1. Create Session

Create a new conversation session with persistent storage.

**Endpoint**: `POST /api/v1/sessions/create`

**Request Body**:
```json
{
  "user_id": "user123",
  "metadata": {
    "agent": "gcp_security_agent",
    "project": "my-project"
  }
}
```

**Response**:
```json
{
  "id": "sess-abc123",
  "user_id": "user123",
  "created_at": "2024-01-15T10:00:00Z",
  "updated_at": "2024-01-15T10:00:00Z",
  "expires_at": "2024-01-16T10:00:00Z",
  "is_active": true,
  "context": {},
  "metadata": {
    "agent": "gcp_security_agent",
    "project": "my-project"
  }
}
```

**Status Codes**:
- `200 OK`: Session created successfully
- `500 Internal Server Error`: Creation failed

---

### 2. Get Session

Retrieve session details by ID.

**Endpoint**: `GET /api/v1/sessions/{session_id}`

**Response**:
```json
{
  "id": "sess-abc123",
  "user_id": "user123",
  "created_at": "2024-01-15T10:00:00Z",
  "updated_at": "2024-01-15T10:30:00Z",
  "expires_at": "2024-01-16T10:00:00Z",
  "is_active": true,
  "context": {
    "last_action": "security_scan",
    "findings": 5
  },
  "metadata": {}
}
```

**Status Codes**:
- `200 OK`: Session found
- `404 Not Found`: Session not found or expired

---

### 3. Update Session

Update session context and metadata.

**Endpoint**: `PUT /api/v1/sessions/{session_id}/update`

**Request Body**:
```json
{
  "context": {
    "last_action": "remediation",
    "vulnerabilities_fixed": 3
  },
  "metadata": {
    "priority": "high"
  }
}
```

**Response**:
```json
{
  "success": true,
  "session_id": "sess-abc123"
}
```

---

### 4. Add Message

Add a message to the conversation history.

**Endpoint**: `POST /api/v1/sessions/{session_id}/messages`

**Request Body**:
```json
{
  "role": "user",
  "content": "What are my security vulnerabilities?",
  "metadata": {
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

**Response**:
```json
{
  "id": "msg-xyz789",
  "session_id": "sess-abc123",
  "role": "user",
  "content": "What are my security vulnerabilities?",
  "timestamp": "2024-01-15T10:30:00Z",
  "metadata": {
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

---

### 5. Get Conversation History

Retrieve conversation history for a session.

**Endpoint**: `GET /api/v1/sessions/{session_id}/messages`

**Query Parameters**:
- `limit` (optional): Maximum number of messages to return

**Response**:
```json
[
  {
    "id": "msg-001",
    "session_id": "sess-abc123",
    "role": "user",
    "content": "Hello, analyze my GCP security",
    "timestamp": "2024-01-15T10:00:00Z",
    "metadata": {}
  },
  {
    "id": "msg-002",
    "session_id": "sess-abc123",
    "role": "assistant",
    "content": "I'll analyze your GCP security posture...",
    "timestamp": "2024-01-15T10:00:05Z",
    "metadata": {
      "tool_used": "analyze_security"
    }
  }
]
```

---

### 6. Get User Sessions

Get all sessions for a specific user.

**Endpoint**: `GET /api/v1/sessions/user/{user_id}`

**Query Parameters**:
- `active_only` (boolean, default: true): Return only active sessions

**Response**:
```json
[
  {
    "id": "sess-abc123",
    "user_id": "user123",
    "created_at": "2024-01-15T10:00:00Z",
    "updated_at": "2024-01-15T11:00:00Z",
    "expires_at": "2024-01-16T10:00:00Z",
    "is_active": true,
    "context": {},
    "metadata": {}
  },
  {
    "id": "sess-def456",
    "user_id": "user123",
    "created_at": "2024-01-14T09:00:00Z",
    "updated_at": "2024-01-14T10:00:00Z",
    "expires_at": "2024-01-15T09:00:00Z",
    "is_active": true,
    "context": {},
    "metadata": {}
  }
]
```

---

### 7. Expire Session

Mark a session as expired/inactive.

**Endpoint**: `DELETE /api/v1/sessions/{session_id}/expire`

**Response**:
```json
{
  "success": true,
  "session_id": "sess-abc123"
}
```

---

### 8. Cleanup Expired Sessions

Clean up expired sessions (usually called by a scheduled job).

**Endpoint**: `POST /api/v1/sessions/cleanup`

**Response**:
```json
{
  "success": true,
  "expired_sessions": 5,
  "timestamp": "2024-01-15T12:00:00Z"
}
```

---

### 9. Get Session Summary

Get a summary of a session with statistics.

**Endpoint**: `GET /api/v1/sessions/{session_id}/summary`

**Response**:
```json
{
  "session_id": "sess-abc123",
  "user_id": "user123",
  "created_at": "2024-01-15T10:00:00Z",
  "updated_at": "2024-01-15T11:00:00Z",
  "expires_at": "2024-01-16T10:00:00Z",
  "is_active": true,
  "message_count": 12,
  "user_messages": 6,
  "assistant_messages": 6,
  "context": {
    "last_action": "security_scan"
  },
  "metadata": {}
}
```

---

### 10. Search Messages

Search messages across sessions.

**Endpoint**: `POST /api/v1/sessions/search`

**Request Body**:
```json
{
  "query": "vulnerability",
  "session_id": null,
  "user_id": "user123"
}
```

**Response**:
```json
{
  "success": true,
  "query": "vulnerability",
  "results": [
    {
      "message_id": "msg-001",
      "session_id": "sess-abc123",
      "user_id": "user123",
      "role": "user",
      "content": "Find my security vulnerabilities",
      "timestamp": "2024-01-15T10:00:00Z",
      "metadata": {}
    },
    {
      "message_id": "msg-002",
      "session_id": "sess-abc123",
      "user_id": "user123",
      "role": "assistant",
      "content": "I found 5 critical vulnerabilities...",
      "timestamp": "2024-01-15T10:00:05Z",
      "metadata": {}
    }
  ],
  "count": 2
}
```

## Data Models

### Session
```typescript
interface Session {
  id: string;                    // Unique session identifier
  user_id?: string;              // Optional user identifier
  created_at: datetime;          // Creation timestamp
  updated_at: datetime;          // Last update timestamp
  expires_at?: datetime;         // Expiration timestamp
  is_active: boolean;            // Active status
  context: object;               // Session context data
  metadata: object;              // Additional metadata
}
```

### Message
```typescript
interface Message {
  id: string;                    // Unique message identifier
  session_id: string;            // Parent session ID
  role: string;                  // 'user' or 'assistant'
  content: string;               // Message content
  timestamp: datetime;           // Message timestamp
  metadata?: object;             // Optional metadata
}
```

## Session Lifecycle

1. **Creation**: Sessions are created with a 24-hour TTL by default
2. **Updates**: Context and metadata can be updated throughout the session
3. **Messages**: Messages are appended to maintain conversation history
4. **Expiration**: Sessions expire after TTL or can be manually expired
5. **Cleanup**: Expired sessions are marked inactive; old data is purged after 30 days

## ADK Agent Integration

The session management is fully integrated with the ADK agent. Example commands:

```
"Create a new session for me"
"Show my conversation history"
"Save this security finding to my session"
"Search for our discussion about IAM permissions"
"List all my active sessions"
```

## Best Practices

1. **Session Creation**: Create a session at the start of each conversation
2. **Context Updates**: Update context after significant actions
3. **Message Storage**: Store both user queries and agent responses
4. **Cleanup**: Run cleanup periodically to manage database size
5. **Search**: Use search to find previous discussions and findings

## Security Considerations

- Sessions are stored in SQLite with proper indexing
- No sensitive credentials are stored in session data
- Sessions expire automatically after 24 hours
- Old session data is purged after 30 days
- Access control should be implemented at the API gateway level

## Performance

- SQLite provides fast local storage
- Indexes on session_id, user_id, and timestamps
- Conversation history is paginated
- Search uses SQL LIKE for simple text matching

## Error Codes

| Code | Description |
|------|-------------|
| `SESS001` | Session not found |
| `SESS002` | Session expired |
| `SESS003` | Invalid session data |
| `SESS004` | Database error |
| `SESS005` | Message save failed |

## Rate Limits

- Create session: 10 per minute per user
- Add message: 60 per minute per session
- Search: 30 per minute per user
- Cleanup: 1 per hour (system-wide)