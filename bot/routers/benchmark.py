from __future__ import annotations

import aiohttp
from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from bot.config import settings

router = Router()

_TIMEOUT = aiohttp.ClientTimeout(total=10)


@router.message(Command("benchmark"))
async def cmd_benchmark(message: Message) -> None:
    await message.answer("⚙️ Запускаю бенчмарк PyTorch vs Sklearn у фоновому режимі…")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{settings.api_base_url}/api/benchmark/run",
                json={"dataset": "breast_cancer", "epochs": 100},
                timeout=_TIMEOUT,
            ) as resp:
                if resp.status == 200:
                    await message.answer(
                        "✅ Бенчмарк запущено.\n"
                        "Результати з'являться через кілька хвилин — "
                        "перегляньте їх командою /results"
                    )
                else:
                    await message.answer("❌ Не вдалося запустити бенчмарк")
    except aiohttp.ClientError as exc:
        await message.answer(f"❌ {exc}")


@router.message(Command("results"))
async def cmd_results(message: Message) -> None:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{settings.api_base_url}/api/benchmark/results",
                timeout=_TIMEOUT,
            ) as resp:
                if resp.status != 200:
                    await message.answer("❌ Не вдалося отримати результати")
                    return
                runs = await resp.json()

        if not runs:
            await message.answer("📭 Результатів ще немає. Запустіть /benchmark")
            return

        last = runs[0]
        lines = [
            f"📊 <b>Бенчмарк</b> {last['created_at'][:10]}",
            f"Датасет: <code>{last['dataset']}</code>\n",
        ]
        for r in last["results"].get("results", []):
            lines.append(
                f"<b>{r['model']}</b> ({r['framework']})\n"
                f"  Accuracy: {r['accuracy']:.3f} | F1: {r['f1_score']:.3f} | "
                f"Час: {r['train_time_sec']:.2f}s\n"
            )
        await message.answer("\n".join(lines))
    except aiohttp.ClientError as exc:
        await message.answer(f"❌ {exc}")
