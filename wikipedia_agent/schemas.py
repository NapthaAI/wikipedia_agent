from pydantic import BaseModel
from typing import Dict

class InputSchema(BaseModel):
    function_name: str
    query: str
    question: str

class SystemPromptSchema(BaseModel):
    """Schema for system prompts."""
    role: str = "You are a helpful AI assistant."