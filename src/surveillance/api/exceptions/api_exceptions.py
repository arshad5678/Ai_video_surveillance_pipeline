"""Exception hierarchy for the FastAPI Backend module.

Routers/services raise these; `handlers.py` maps them to HTTP responses.
Framework-free by design (no `fastapi`/`starlette` import here) so this
hierarchy could in principle be raised from a non-HTTP context too.
"""


class ApiError(Exception):
    """Base exception for all API-layer failures. Maps to HTTP 400 by default."""

    status_code = 400

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ResourceNotFoundError(ApiError):
    """Raised when a requested resource (event, output file, ...) does not exist. Maps to HTTP 404."""

    status_code = 404


class ConfigurationReloadError(ApiError):
    """Raised when reloading config.yaml/zones.yaml fails. Maps to HTTP 400."""

    status_code = 400
