"""Braintrust production tracing (logger + auto-instrumentation)."""

from __future__ import annotations

_CONFIGURED = False

PROJECT_NAME = "psych755-ca-personas"


def configure_tracing() -> None:
    """Initialize the Braintrust logger and auto-instrument supported AI libraries."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    try:
        import braintrust
    except ImportError:
        return
    braintrust.init_logger(project=PROJECT_NAME)
    braintrust.auto_instrument()
    _CONFIGURED = True
