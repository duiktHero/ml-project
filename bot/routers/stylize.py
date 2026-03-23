from __future__ import annotations

import base64

import aiohttp
from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import BufferedInputFile, CallbackQuery, Message

from bot.config import settings
from bot.keyboards.keyboards import get_colorizer_model_keyboard, get_neural_style_keyboard, get_style_keyboard
from bot.states.states import ColorizeStates, NeuralStyleStates, StylizeStates

router = Router()

_TIMEOUT = aiohttp.ClientTimeout(total=60)


async def _fetch_colorizer_models() -> list[dict]:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{settings.api_base_url}/api/colorize/models",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
    except aiohttp.ClientError:
        pass
    return []


# ── /colorize ─────────────────────────────────────────────────────────────────


@router.message(Command("colorize"))
async def cmd_colorize(message: Message, state: FSMContext) -> None:
    await message.answer("🖼 Надішли grayscale-фото для колоризації")
    await state.set_state(ColorizeStates.waiting_grayscale)


@router.message(ColorizeStates.waiting_grayscale, F.photo)
async def process_colorize(message: Message, state: FSMContext) -> None:
    photo = message.photo[-1]
    await state.update_data(photo_file_id=photo.file_id)
    await state.set_state(ColorizeStates.waiting_model)

    models = await _fetch_colorizer_models()
    await state.update_data(colorizer_models=[m["path"] for m in models])
    await message.answer("Оберіть модель:", reply_markup=get_colorizer_model_keyboard(models))


@router.callback_query(ColorizeStates.waiting_model, F.data.startswith("clr_model:"))
async def process_colorizer_model_choice(callback: CallbackQuery, state: FSMContext) -> None:
    value = callback.data.split(":", 1)[1]

    if value == "cancel":
        await state.clear()
        await callback.answer("Скасовано")
        await callback.message.edit_text("❌ Колоризацію скасовано")
        return

    await callback.answer()
    await callback.message.edit_text("⏳ Колоризую…")

    data = await state.get_data()
    await state.clear()

    model_path = data["colorizer_models"][int(value)]

    file = await callback.bot.get_file(data["photo_file_id"])
    img_io = await callback.bot.download_file(file.file_path)
    b64 = base64.b64encode(img_io.read()).decode()

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{settings.api_base_url}/api/colorize",
                json={"image": b64, "model_path": model_path},
                timeout=_TIMEOUT,
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    result_bytes = base64.b64decode(result["result_image"])
                    await callback.message.answer_photo(
                        BufferedInputFile(result_bytes, filename="colorized.jpg"),
                        caption="✅ Колоризоване зображення",
                    )
                elif resp.status == 503:
                    d = await resp.json()
                    await callback.message.answer(f"⚠️ {d['detail']}")
                else:
                    await callback.message.answer("❌ Помилка колоризації")
    except aiohttp.ClientError as exc:
        await callback.message.answer(f"❌ {exc}")


@router.message(ColorizeStates.waiting_grayscale)
async def colorize_wrong_input(message: Message) -> None:
    await message.answer("⚠️ Очікую зображення")


# ── /stylize ──────────────────────────────────────────────────────────────────


@router.message(Command("stylize"))
async def cmd_stylize(message: Message, state: FSMContext) -> None:
    await message.answer("🎨 Надішли фото для стилізації")
    await state.set_state(StylizeStates.waiting_image)


@router.message(StylizeStates.waiting_image, F.photo)
async def stylize_got_image(message: Message, state: FSMContext) -> None:
    photo = message.photo[-1]
    await state.update_data(photo_file_id=photo.file_id)
    await state.set_state(StylizeStates.waiting_style)
    await message.answer("Оберіть стиль:", reply_markup=get_style_keyboard())


@router.callback_query(StylizeStates.waiting_style, F.data.startswith("style:"))
async def process_style_choice(callback: CallbackQuery, state: FSMContext) -> None:
    style = callback.data.split(":", 1)[1]

    if style == "cancel":
        await state.clear()
        await callback.answer("Скасовано")
        await callback.message.edit_text("❌ Стилізацію скасовано")
        return

    await callback.answer()
    await callback.message.edit_text(f"⏳ Стилізую у стилі «{style}»…")

    data = await state.get_data()
    await state.clear()

    file = await callback.bot.get_file(data["photo_file_id"])
    img_io = await callback.bot.download_file(file.file_path)
    b64 = base64.b64encode(img_io.read()).decode()

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{settings.api_base_url}/api/stylize",
                json={"image": b64, "style": style},
                timeout=_TIMEOUT,
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    result_bytes = base64.b64decode(result["result_image"])
                    await callback.message.answer_photo(
                        BufferedInputFile(result_bytes, filename="stylized.jpg"),
                        caption=f"✅ Стиль: <b>{style}</b>",
                    )
                else:
                    await callback.message.answer("❌ Помилка стилізації")
    except aiohttp.ClientError as exc:
        await callback.message.answer(f"❌ {exc}")


@router.message(StylizeStates.waiting_image)
async def stylize_wrong_input(message: Message) -> None:
    await message.answer("⚠️ Очікую зображення")


# ── /neural_stylize ───────────────────────────────────────────────────────────

NEURAL_STYLE_LABELS = {
    "starry_night":    "Starry Night (Van Gogh)",
    "great_wave":      "Great Wave (Hokusai)",
    "the_scream":      "The Scream (Munch)",
    "composition_viii": "Composition VIII (Kandinsky)",
}

_NST_TIMEOUT = aiohttp.ClientTimeout(total=120)  # NST takes longer than CV filters


@router.message(Command("neural_stylize"))
async def cmd_neural_stylize(message: Message, state: FSMContext) -> None:
    await message.answer("🧠 Надішли фото для Neural Style Transfer")
    await state.set_state(NeuralStyleStates.waiting_image)


@router.message(NeuralStyleStates.waiting_image, F.photo)
async def neural_stylize_got_image(message: Message, state: FSMContext) -> None:
    photo = message.photo[-1]
    await state.update_data(photo_file_id=photo.file_id)
    await state.set_state(NeuralStyleStates.waiting_style)
    await message.answer("Оберіть художній стиль:", reply_markup=get_neural_style_keyboard())


@router.callback_query(NeuralStyleStates.waiting_style, F.data.startswith("nstyle:"))
async def process_neural_style_choice(callback: CallbackQuery, state: FSMContext) -> None:
    style = callback.data.split(":", 1)[1]

    if style == "cancel":
        await state.clear()
        await callback.answer("Скасовано")
        await callback.message.edit_text("❌ Neural Style Transfer скасовано")
        return

    label = NEURAL_STYLE_LABELS.get(style, style)
    await callback.answer()
    await callback.message.edit_text(
        f"⏳ Застосовую нейро-стиль «{label}»…\n"
        f"<i>Це займає 5–15 с на GPU. Зачекайте.</i>"
    )

    data = await state.get_data()
    await state.clear()

    file = await callback.bot.get_file(data["photo_file_id"])
    img_io = await callback.bot.download_file(file.file_path)
    b64 = base64.b64encode(img_io.read()).decode()

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{settings.api_base_url}/api/neural-stylize",
                json={"image": b64, "style": style},
                timeout=_NST_TIMEOUT,
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    result_bytes = base64.b64decode(result["result_image"])
                    await callback.message.answer_photo(
                        BufferedInputFile(result_bytes, filename="neural_style.jpg"),
                        caption=f"✅ Нейро-стиль: <b>{label}</b>",
                    )
                elif resp.status == 503:
                    d = await resp.json()
                    await callback.message.answer(f"⚠️ {d['detail']}")
                else:
                    await callback.message.answer("❌ Помилка Neural Style Transfer")
    except aiohttp.ClientError as exc:
        await callback.message.answer(f"❌ {exc}")


@router.message(NeuralStyleStates.waiting_image)
async def neural_stylize_wrong_input(message: Message) -> None:
    await message.answer("⚠️ Очікую зображення")


# ── Inline button from classify result ────────────────────────────────────────


@router.callback_query(F.data == "stylize_from_last")
async def stylize_from_classify(callback: CallbackQuery, state: FSMContext) -> None:
    data = await state.get_data()
    file_id = data.get("last_photo_file_id")
    if not file_id:
        await callback.answer("Фото не знайдено, надішли знову")
        return
    await state.update_data(photo_file_id=file_id)
    await state.set_state(StylizeStates.waiting_style)
    await callback.answer()
    await callback.message.answer("Оберіть стиль:", reply_markup=get_style_keyboard())


# ── Shared helper ─────────────────────────────────────────────────────────────


async def _send_to_endpoint(
    message: Message,
    endpoint: str,
    payload_key: str,
    caption: str,
) -> None:
    photo = message.photo[-1]
    file = await message.bot.get_file(photo.file_id)
    img_io = await message.bot.download_file(file.file_path)
    b64 = base64.b64encode(img_io.read()).decode()

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{settings.api_base_url}{endpoint}",
                json={payload_key: b64},
                timeout=_TIMEOUT,
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    result_bytes = base64.b64decode(result["result_image"])
                    await message.answer_photo(
                        BufferedInputFile(result_bytes, filename="result.jpg"),
                        caption=caption,
                    )
                elif resp.status == 503:
                    d = await resp.json()
                    await message.answer(f"⚠️ {d['detail']}")
                else:
                    await message.answer("❌ Помилка обробки")
    except aiohttp.ClientError as exc:
        await message.answer(f"❌ {exc}")
