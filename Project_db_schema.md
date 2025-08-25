# Database Schema Documentation

This document describes the database schema for the AI Personal Assistant system. It includes tables, columns, constraints, and relationships.

---

## **Tables**

### 1. `conversations`
Stores metadata about conversation sessions.

| Column         | Type             | Default                  | Description                           |
|---------------|-----------------|-------------------------|---------------------------------------|
| `id`          | INTEGER (PK)    | auto-increment         | Unique identifier for the conversation |
| `started_at`  | TIMESTAMP       | CURRENT_TIMESTAMP       | When the conversation started         |
| `ended_at`    | TIMESTAMP       |                         | When the conversation ended           |
| `title`       | VARCHAR(255)    | 'Untitled Conversation' | Conversation title                    |
| `session_id`  | VARCHAR(100)    |                         | Session identifier                    |
| `metadata`    | JSONB           | '{}'                   | Additional metadata                   |
| `created_at`  | TIMESTAMP       | CURRENT_TIMESTAMP       | Creation time                         |
| `updated_at`  | TIMESTAMP       | CURRENT_TIMESTAMP       | Last update time                      |
| `summary`     | TEXT            |                         | Conversation summary                   |
| `tool_usage_summary` | TEXT     |                         | Summary of tools used in the conversation |

**Indexes:**
- `idx_conversations_session_id`
- `idx_conversations_started_at`

---

### 2. `messages`
Stores individual messages exchanged during conversations.

| Column              | Type            | Default           | Description                                  |
|---------------------|---------------|-----------------|----------------------------------------------|
| `id`               | INTEGER (PK)  | auto-increment  | Unique identifier for the message           |
| `conversation_id`  | INTEGER (FK)  |                 | References `conversations.id`               |
| `role`             | VARCHAR(20)   |                 | Role of the sender (`user`, `assistant`, `system`, `tool`) |
| `content`          | TEXT          |                 | Message content                              |
| `tool_name`        | VARCHAR(100)  |                 | Name of the tool used (if any)             |
| `tool_result`      | JSONB         |                 | Result from the tool execution             |
| `is_correction`    | BOOLEAN       | false           | Whether this message corrects a previous one |
| `parent_message_id`| INTEGER (FK)  |                 | References previous message (threading)    |
| `sequence_number`  | INTEGER       | 1               | Order of message within the conversation   |
| `metadata`         | JSONB         | '{}'            | Additional metadata                         |
| `created_at`       | TIMESTAMP     | CURRENT_TIMESTAMP| Creation time                               |
| `updated_at`       | TIMESTAMP     | CURRENT_TIMESTAMP| Last update time                            |
| `tool_id`          | INTEGER (FK)  |                 | References `tools.id`                      |

**Constraints:**
- `messages_conversation_id_fkey` → `conversations(id)` (ON DELETE CASCADE)
- `messages_parent_message_id_fkey` → `messages(id)`
- `messages_tool_id_fkey` → `tools(id)`
- Role must be one of `user`, `assistant`, `system`, `tool`

**Indexes:**
- `idx_messages_conversation_id`
- `idx_messages_created_at`
- `idx_messages_role`
- `idx_messages_sequence`

---

### 3. `tools`
Stores available tools that the assistant can call.

| Column            | Type           | Default           | Description                             |
|-------------------|--------------|------------------|-----------------------------------------|
| `id`             | INTEGER (PK) | auto-increment  | Unique identifier for the tool         |
| `name`           | VARCHAR(255) |                 | Tool name                              |
| `description`    | TEXT          |                 | Tool description                        |
| `python_function`| VARCHAR(255) |                 | Name of the Python function implementing the tool |
| `active`         | BOOLEAN       | true            | Whether the tool is active             |
| `created_at`     | TIMESTAMP     | now()           | Creation time                           |
| `updated_at`     | TIMESTAMP     | now()           | Last update time                        |
| `query_examples` | TEXT[]        |                 | Example queries for the tool           |

---

## **Relationships**
- **conversations (1) → (N) messages**
- **messages (optional) → tools**  
- **messages (optional) → parent message** (self-relation for threading)

---

## **Views**
### `conversation_summary`
Provides an aggregated view of conversations:
- `id`, `title`, `summary`, `tool_usage_summary`
- `message_count` (number of messages in the conversation)
- `last_message_at` (latest message timestamp)

---

## **Example Queries**

- Get all messages in a conversation:
```sql
SELECT * FROM messages WHERE conversation_id = 123 ORDER BY sequence_number;
