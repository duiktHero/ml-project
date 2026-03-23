from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from api.models import BotActivity, User
from bot.config import settings

_engine = create_async_engine(settings.database_url, echo=False, pool_pre_ping=True)
SessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    _engine, class_=AsyncSession, expire_on_commit=False
)


async def get_or_create_user(
    session: AsyncSession,
    telegram_id: int,
    username: str | None,
    first_name: str | None,
) -> User:
    result = await session.execute(
        select(User).where(User.telegram_id == telegram_id)
    )
    user = result.scalar_one_or_none()
    if user is None:
        user = User(
            telegram_id=telegram_id,
            username=username,
            first_name=first_name,
        )
        session.add(user)
        await session.flush()
    return user


async def log_activity(session: AsyncSession, user_id: int, command: str) -> None:
    session.add(BotActivity(user_id=user_id, command=command))
