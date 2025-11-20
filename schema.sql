-- Enable pgcrypto extension to generate UUIDs, a best practice for primary keys
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Table for storing individual chat sessions
CREATE TABLE chat_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(255) NOT NULL DEFAULT 'New Chat',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Table for storing all messages (user, assistant, system)
CREATE TABLE chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for fast retrieval of messages for a specific session, ordered by time
CREATE INDEX idx_chat_messages_session_id ON chat_messages(session_id, created_at);