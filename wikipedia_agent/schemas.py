from pydantic import BaseModel
from typing import Dict, Optional, Union

class InputSchema(BaseModel):
    query: str
    question: str
