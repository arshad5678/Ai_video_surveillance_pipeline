"""ASGI entrypoint.

Run with: uvicorn src.main:app --reload
"""

from src.surveillance.api.app import create_app

app = create_app()
