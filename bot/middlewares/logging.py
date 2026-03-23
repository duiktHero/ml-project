from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware
from aiogram.types import TelegramObject

from bot.db import SessionLocal, get_or_create_user, log_activity

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseMiddleware):
    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        user = data.get("event_from_user")
        if user:
            text = (
                getattr(event, "text", None)
                or getattr(event, "caption", None)
                or "<media>"
            )
            logger.info("[%s] @%s → %s", user.id, user.username, text)

            try:
                async with SessionLocal() as session:
                    db_user = await get_or_create_user(
                        session, user.id, user.username, user.first_name
                    )
                    await log_activity(session, db_user.id, text[:128])
                    await session.commit()
            except Exception:
                logger.exception("DB logging error (non-fatal)")

        return await handler(event, data)
