import uuid
import datetime
from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from typing import List

# Use the modern DeclarativeBase
class Base(DeclarativeBase):
    pass

class ChatSession(Base):
    """
    ORM Model for the chat_sessions table.
    """
    __tablename__ = "chat_sessions"
    
    # Mapped columns for modern SQLAlchemy
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    title: Mapped[str] = mapped_column(String(255), default="New Chat", nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.datetime.now, nullable=False
    )
    
    # One-to-Many relationship: A session has many messages
    # cascade="all, delete-orphan" means messages are deleted when the session is.
    messages: Mapped[List["ChatMessage"]] = relationship(
        "ChatMessage", 
        back_populates="session", 
        cascade="all, delete-orphan", 
        lazy="select"
    )

    def __repr__(self):
        return f"<ChatSession(id={self.id}, title='{self.title}')>"

class ChatMessage(Base):
    """
    ORM Model for the chat_messages table.
    """
    __tablename__ = "chat_messages"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chat_sessions.id"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(50), nullable=False)  # 'user', 'assistant', 'system'
    
    # Use JSONB for flexible, structured content storage
    content: Mapped[dict] = mapped_column(JSONB, nullable=False)
    
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.datetime.now, nullable=False
    )
    
    # Many-to-One relationship: A message belongs to one session
    session: Mapped["ChatSession"] = relationship("ChatSession", back_populates="messages")

    def __repr__(self):
        return f"<ChatMessage(id={self.id}, role='{self.role}')>"