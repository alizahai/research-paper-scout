"""Pydantic schemas for request and response payloads."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
