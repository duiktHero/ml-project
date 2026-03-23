from __future__ import annotations

import os

import aiohttp
from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import CallbackQuery, Message

from bot.config import settings
from bot.keyboards.keyboards import get_admin_keyboard

router = Router()

_TIMEOUT = aiohttp.ClientTimeout(total=10)


def _is_admin(user_id: int) -> bool:
    return user_id in settings.admin_id_list


@router.message(Command("admin"))
async def cmd_admin(message: Message) -> None:
    if not _is_admin(message.from_user.id):
        await message.answer("⛔ Доступ заборонено")
        return

    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(
                f"{settings.api_base_url}/api/health", timeout=_TIMEOUT
            ) as resp:
                health = await resp.json() if resp.status == 200 else {}
    except Exception:
        health = {}

    models_ok = health.get("models_loaded", False)
    status = "✅ Завантажені" if models_ok else "❌ Не знайдені (запустіть навчання)"

    await message.answer(
        "🔧 <b>Адмін-панель</b>\n\n"
        f"🧠 Моделі: {status}\n"
        f"🌐 API: <code>{settings.api_base_url}</code>",
        reply_markup=get_admin_keyboard(),
    )


@router.callback_query(F.data == "admin:model_status")
async def cb_model_status(callback: CallbackQuery) -> None:
    if not _is_admin(callback.from_user.id):
        await callback.answer("⛔ Доступ заборонено", show_alert=True)
        return

    classifier_ok = os.path.exists(settings.model_classifier)
    colorizer_ok = os.path.exists(settings.model_colorizer)

    await callback.answer()
    await callback.message.answer(
        f"🧠 Класифікатор (CIFAR-10): {'✅ знайдено' if classifier_ok else '❌ не знайдено'}\n"
        f"🎨 Колоризатор: {'✅ знайдено' if colorizer_ok else '❌ не знайдено'}\n\n"
        "Для навчання: <code>docker compose --profile train up</code>"
    )


@router.callback_query(F.data == "admin:last_benchmark")
async def cb_last_benchmark(callback: CallbackQuery) -> None:
    if not _is_admin(callback.from_user.id):
        await callback.answer("⛔ Доступ заборонено", show_alert=True)
        return

    await callback.answer()
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(
                f"{settings.api_base_url}/api/benchmark/results", timeout=_TIMEOUT
            ) as resp:
                runs = await resp.json() if resp.status == 200 else []

        if not runs:
            await callback.message.answer("📭 Результатів ще немає")
            return

        last = runs[0]
        count = len(last["results"].get("results", []))
        await callback.message.answer(
            f"📊 Бенчмарк {last['created_at'][:10]}: "
            f"датасет <code>{last['dataset']}</code>, {count} моделі"
        )
    except Exception as exc:
        await callback.message.answer(f"❌ {exc}")
