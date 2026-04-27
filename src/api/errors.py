from __future__ import annotations

from typing import NoReturn

from fastapi import HTTPException, status


def bad_request(message: str) -> HTTPException:
    """Build a 400 error for invalid request semantics."""

    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=message,
    )


def service_unavailable(message: str) -> HTTPException:
    """Build a 503 error for unavailable runtime dependencies."""

    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=message,
    )


def gateway_timeout(message: str) -> HTTPException:
    """Build a 504 error for slow backend operations."""

    return HTTPException(
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        detail=message,
    )


def internal_server_error() -> HTTPException:
    """Build a generic 500 error without exposing internal details."""

    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Internal API error.",
    )


def to_http_exception(exc: Exception) -> HTTPException:
    """Map expected runtime exceptions to API-facing HTTP errors."""

    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, ValueError):
        return bad_request(str(exc))
    if isinstance(exc, TimeoutError):
        return gateway_timeout(str(exc) or "RAG backend operation timed out.")
    if isinstance(exc, RuntimeError):
        return service_unavailable(str(exc) or "RAG backend is unavailable.")
    return service_unavailable(str(exc) or "RAG backend is unavailable.")


def raise_http_error(exc: Exception) -> NoReturn:
    """Raise the HTTP equivalent of an expected runtime exception."""

    raise to_http_exception(exc)
