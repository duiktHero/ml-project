from __future__ import annotations

import asyncio
import logging
import os
from logging.handlers import TimedRotatingFileHandler

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from bot.config import settings
from bot.middlewares.logging import LoggingMiddleware
from bot.routers import admin, benchmark, classify, start, stylize


def setup_logging() -> None:
    os.makedirs("logs", exist_ok=True)
    file_handler = TimedRotatingFileHandler(
        "logs/bot.log",
        when="midnight",
        backupCount=7,
        encoding="utf-8",
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[file_handler, logging.StreamHandler()],
    )


async def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)

    bot = Bot(
        token=settings.bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher()

    dp.message.middleware(LoggingMiddleware())

    dp.include_router(start.router)
    dp.include_router(classify.router)
    dp.include_router(stylize.router)
    dp.include_router(benchmark.router)
    dp.include_router(admin.router)

    logger.info("Bot starting — polling…")
    await dp.start_polling(bot, allowed_updates=["message", "callback_query"])


if __name__ == "__main__":
    asyncio.run(main())
