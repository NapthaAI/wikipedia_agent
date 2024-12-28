from pydantic import BaseModel
from typing import Dict

class InputSchema(BaseModel):
    tool_name: str
    tool_input_data: Dict

class SystemPromptSchema(BaseModel):
    """Schema for system prompts."""
    role: str = "You are a helpful AI assistant."