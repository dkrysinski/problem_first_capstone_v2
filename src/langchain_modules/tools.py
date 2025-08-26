from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field

class CustomToolInput(BaseModel):
    query: str = Field(description="Input query for the tool")

class CustomTool(BaseTool):
    name = "custom_tool"
    description = "A custom tool for the AI agent"
    args_schema: Type[BaseModel] = CustomToolInput

    def _run(self, query: str) -> str:
        """Execute the tool"""
        return f"Tool executed with query: {query}"

    async def _arun(self, query: str) -> str:
        """Async execution of the tool"""
        return self._run(query)