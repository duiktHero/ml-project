from __future__ import annotations

import base64

import aiohttp
from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message

from bot.config import settings
from bot.keyboards.keyboards import get_classify_result_keyboard, get_classifier_model_keyboard
from bot.states.states import ClassifyStates

router = Router()

_TIMEOUT = aiohttp.ClientTimeout(total=30)


async def _fetch_classifier_models() -> list[dict]:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{settings.api_base_url}/api/classify/models",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
    except aiohttp.ClientError:
        pass
    return []


@router.message(Command("classify"))
async def cmd_classify(message: Message, state: FSMContext) -> None:
    await message.answer("📸 Надішли зображення для класифікації")
    await state.set_state(ClassifyStates.waiting_image)


@router.message(ClassifyStates.waiting_image, F.photo)
async def process_classify_image(message: Message, state: FSMContext) -> None:
    photo = message.photo[-1]
    await state.update_data(photo_file_id=photo.file_id)
    await state.set_state(ClassifyStates.waiting_model)

    models = await _fetch_classifier_models()
    await state.update_data(classifier_models=[m["path"] for m in models])
    await message.answer("Оберіть модель:", reply_markup=get_classifier_model_keyboard(models))


@router.callback_query(ClassifyStates.waiting_model, F.data.startswith("clf_model:"))
async def process_classifier_model_choice(callback: CallbackQuery, state: FSMContext) -> None:
    value = callback.data.split(":", 1)[1]

    if value == "cancel":
        await state.clear()
        await callback.answer("Скасовано")
        await callback.message.edit_text("❌ Класифікацію скасовано")
        return

    await callback.answer()
    await callback.message.edit_text("⏳ Класифікую…")

    data = await state.get_data()
    await state.clear()

    file = await callback.bot.get_file(data["photo_file_id"])
    img_io = await callback.bot.download_file(file.file_path)
    b64 = base64.b64encode(img_io.read()).decode()

    model_path = None
    if value != "imagenet":
        model_path = data["classifier_models"][int(value)]

    try:
        async with aiohttp.ClientSession() as session:
            if value == "imagenet":
                endpoint = f"{settings.api_base_url}/api/classify/imagenet"
                payload = {"image": b64}
            else:
                endpoint = f"{settings.api_base_url}/api/classify"
                payload = {"image": b64, "model_path": model_path}

            async with session.post(endpoint, json=payload, timeout=_TIMEOUT) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    label = result["label"]
                    confidence = result["confidence"]
                    await state.update_data(last_photo_file_id=data["photo_file_id"])
                    await callback.message.answer(
                        f"🏷 Клас: <b>{label}</b>\n📊 Впевненість: {confidence:.1%}",
                        reply_markup=get_classify_result_keyboard(),
                    )
                elif resp.status == 503:
                    d = await resp.json()
                    await callback.message.answer(f"⚠️ {d['detail']}")
                else:
                    await callback.message.answer("❌ Помилка класифікації")
    except aiohttp.ClientError as exc:
        await callback.message.answer(f"❌ Помилка з'єднання з API: {exc}")


@router.message(ClassifyStates.waiting_image)
async def wrong_classify_input(message: Message) -> None:
    await message.answer("⚠️ Очікую зображення. Надішли фото або /start для скасування.")


@router.callback_query(F.data == "classify:retry")
async def retry_classify(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    await callback.message.answer("📸 Надішли нове зображення для класифікації")
    await state.set_state(ClassifyStates.waiting_image)
