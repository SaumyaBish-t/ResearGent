"""Authentication + session management (Phase 16)."""

from src.auth.deps import current_user, current_user_optional
from src.auth.users import User

__all__ = ["User", "current_user", "current_user_optional"]
