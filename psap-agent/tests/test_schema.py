"""Tests for the schema module."""

from psap_agent.src.schema import (
    ChatHistoryResponse,
    ChatMessage,
    FeedbackRequest,
    FeedbackResponse,
    StreamRequest,
    ToolCall,
    UserInput,
)


class TestUserInput:
    """Test cases for UserInput model."""

    def test_user_input_creation(self):
        """Test creating UserInput with required fields."""
        user_input = UserInput(message="Hello world")
        assert user_input.message == "Hello world"
        assert user_input.thread_id is None
        assert user_input.session_id is None
        assert user_input.user_id is None

    def test_user_input_with_optional_fields(self):
        """Test creating UserInput with all fields."""
        user_input = UserInput(
            message="Hello world",
            thread_id="thread_123",
            session_id="session_456",
            user_id="user_789",
        )
        assert user_input.message == "Hello world"
        assert user_input.thread_id == "thread_123"
        assert user_input.session_id == "session_456"
        assert user_input.user_id == "user_789"


class TestStreamRequest:
    """Test cases for StreamRequest model."""

    def test_stream_request_creation(self):
        """Test creating StreamRequest with default stream_tokens."""
        stream_request = StreamRequest(message="Hello world")
        assert stream_request.message == "Hello world"
        assert stream_request.stream_tokens is True

    def test_stream_request_with_custom_stream_tokens(self):
        """Test creating StreamRequest with custom stream_tokens."""
        stream_request = StreamRequest(message="Hello world", stream_tokens=False)
        assert stream_request.message == "Hello world"
        assert stream_request.stream_tokens is False


class TestToolCall:
    """Test cases for ToolCall TypedDict."""

    def test_tool_call_creation(self):
        """Test creating ToolCall with required fields."""
        tool_call = ToolCall(name="test_tool", args={"param": "value"}, id="call_123")
        assert tool_call["name"] == "test_tool"
        assert tool_call["args"] == {"param": "value"}
        assert tool_call["id"] == "call_123"

    def test_tool_call_with_type(self):
        """Test creating ToolCall with type field."""
        tool_call = ToolCall(
            name="test_tool", args={"param": "value"}, id="call_123", type="tool_call"
        )
        assert tool_call["type"] == "tool_call"


class TestChatMessage:
    """Test cases for ChatMessage model."""

    def test_chat_message_human(self):
        """Test creating human ChatMessage."""
        message = ChatMessage(type="human", content="Hello")
        assert message.type == "human"
        assert message.content == "Hello"
        assert message.tool_calls == []
        assert message.tool_call_id is None
        assert message.run_id is None
        assert message.response_metadata == {}
        assert message.custom_data == {}

    def test_chat_message_ai(self):
        """Test creating AI ChatMessage."""
        message = ChatMessage(
            type="ai",
            content="Hello",
            tool_calls=[{"name": "test_tool", "args": {}, "id": "call_123"}],
        )
        assert message.type == "ai"
        assert message.content == "Hello"
        assert len(message.tool_calls) == 1
        assert message.tool_calls[0]["name"] == "test_tool"

    def test_chat_message_tool(self):
        """Test creating tool ChatMessage."""
        message = ChatMessage(
            type="tool", content="Tool result", tool_call_id="call_123"
        )
        assert message.type == "tool"
        assert message.content == "Tool result"
        assert message.tool_call_id == "call_123"

    def test_chat_message_custom(self):
        """Test creating custom ChatMessage."""
        message = ChatMessage(type="custom", content="", custom_data={"key": "value"})
        assert message.type == "custom"
        assert message.content == ""
        assert message.custom_data == {"key": "value"}


class TestFeedbackRequest:
    """Test cases for FeedbackRequest model."""

    def test_feedback_request_creation(self):
        """Test creating FeedbackRequest with required fields."""
        feedback = FeedbackRequest(run_id="run_123", key="response_quality", score=4.5)
        assert feedback.run_id == "run_123"
        assert feedback.key == "response_quality"
        assert feedback.score == 4.5
        assert feedback.kwargs == {}

    def test_feedback_request_with_kwargs(self):
        """Test creating FeedbackRequest with kwargs."""
        feedback = FeedbackRequest(
            run_id="run_123",
            key="response_quality",
            score=4.5,
            kwargs={"comment": "Great response"},
        )
        assert feedback.kwargs == {"comment": "Great response"}


class TestFeedbackResponse:
    """Test cases for FeedbackResponse model."""

    def test_feedback_response_creation(self):
        """Test creating FeedbackResponse."""
        response = FeedbackResponse()
        assert response.status == "success"


class TestChatHistoryResponse:
    """Test cases for ChatHistoryResponse model."""

    def test_chat_history_response_creation(self):
        """Test creating ChatHistoryResponse."""
        messages = [
            ChatMessage(type="human", content="Hello"),
            ChatMessage(type="ai", content="Hi there"),
        ]
        response = ChatHistoryResponse(messages=messages)
        assert len(response.messages) == 2
        assert response.messages[0].type == "human"
        assert response.messages[1].type == "ai"
