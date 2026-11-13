"""Exception types shared across modules, and the result shape of a model call.

They live here so that modules built on top of the eBay and Crusoe clients (the funnel,
the poller, the CLI) can catch these without importing the clients themselves.
Messages must never contain a secret.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---- Crusoe ------------------------------------------------------------------

class CrusoeError(Exception):
    """No valid answer could be obtained from a model."""


class CrusoeAuthError(CrusoeError):
    """401 or 403: the key is missing, wrong or expired. Never retried."""


class CrusoeUnavailable(CrusoeError):
    """Rate limited or unavailable on every candidate model, after retries."""


@dataclass
class ChatResult:
    """A validated JSON answer from a model, with what is needed for the funnel trace."""
    data: Dict[str, Any]
    model_requested: str
    model_served: str
    swapped: bool = False          # the endpoint answered with a different model than requested
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


# ---- eBay --------------------------------------------------------------------

class EbayError(Exception):
    """An eBay API call failed."""

    def __init__(self, message: str, status: Optional[int] = None, error_ids: Optional[List[int]] = None) -> None:
        super().__init__(message)
        self.status = status
        self.error_ids = list(error_ids or [])


class EbayAuthError(EbayError):
    """401 or 403: the keyset is invalid, disabled or lacks access."""


class EbayNotFound(EbayError):
    """The listing does not exist (404, error 11001 or 11003). Not always permanent."""


class EbayUnavailable(EbayError):
    """Temporary: error 11004, 429, 5xx or a network failure. Try again later."""


class EbayGroupRequired(EbayError):
    """Error 11006: the id is a multi-variation parent; read its item group instead."""

    def __init__(self, message: str, item_group_id: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.item_group_id = item_group_id


class EbayBudgetExceeded(EbayError):
    """The daily call cap was reached. No request was sent."""
