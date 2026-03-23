from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder


def get_classifier_model_keyboard(models: list[dict]) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    for i, m in enumerate(models):
        builder.row(InlineKeyboardButton(
            text=m["name"],
            callback_data=f"clf_model:{i}",
        ))
    builder.row(InlineKeyboardButton(text="🌐 ImageNet", callback_data="clf_model:imagenet"))
    builder.row(InlineKeyboardButton(text="❌ Скасувати", callback_data="clf_model:cancel"))
    return builder.as_markup()


def get_colorizer_model_keyboard(models: list[dict]) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    for i, m in enumerate(models):
        builder.row(InlineKeyboardButton(
            text=m["name"],
            callback_data=f"clr_model:{i}",
        ))
    builder.row(InlineKeyboardButton(text="❌ Скасувати", callback_data="clr_model:cancel"))
    return builder.as_markup()


def get_style_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.row(
        InlineKeyboardButton(text="🎨 Van Gogh", callback_data="style:vangogh"),
        InlineKeyboardButton(text="🖤 Чорно-білий", callback_data="style:bw"),
    )
    builder.row(
        InlineKeyboardButton(text="🌊 Hokusai", callback_data="style:hokusai"),
        InlineKeyboardButton(text="🎭 Cartoon", callback_data="style:cartoon"),
    )
    builder.row(InlineKeyboardButton(text="❌ Скасувати", callback_data="style:cancel"))
    return builder.as_markup()


def get_neural_style_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.row(
        InlineKeyboardButton(text="🌃 Starry Night", callback_data="nstyle:starry_night"),
        InlineKeyboardButton(text="🌊 Great Wave",   callback_data="nstyle:great_wave"),
    )
    builder.row(
        InlineKeyboardButton(text="😱 The Scream",        callback_data="nstyle:the_scream"),
        InlineKeyboardButton(text="🎨 Composition VIII",  callback_data="nstyle:composition_viii"),
    )
    builder.row(InlineKeyboardButton(text="❌ Скасувати", callback_data="nstyle:cancel"))
    return builder.as_markup()


def get_classify_result_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.row(
        InlineKeyboardButton(text="🔄 Спробувати ще раз", callback_data="classify:retry"),
        InlineKeyboardButton(text="🎨 Стилізувати це фото", callback_data="stylize_from_last"),
    )
    return builder.as_markup()


def get_admin_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.row(
        InlineKeyboardButton(text="✅ Статус моделей", callback_data="admin:model_status")
    )
    builder.row(
        InlineKeyboardButton(text="📊 Останній бенчмарк", callback_data="admin:last_benchmark")
    )
    return builder.as_markup()
