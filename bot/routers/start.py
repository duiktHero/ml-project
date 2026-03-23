from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

router = Router()

HELP_TEXT = """\
<b>ML Sandbox Bot</b> 🤖

<b>Команди:</b>
/classify  — класифікувати зображення (CIFAR-10, 10 класів)
/colorize  — колоризувати grayscale-фото
/stylize   — стилізувати фото (Van Gogh, Hokusai, Cartoon, Ч/Б)
/benchmark — запустити PyTorch vs Sklearn бенчмарк
/results   — переглянути результати бенчмарку
/admin     — панель адміністратора
/help      — ця довідка"""


@router.message(Command("start"))
async def cmd_start(message: Message) -> None:
    await message.answer(
        f"Привіт, <b>{message.from_user.first_name}</b>! 👋\n\n"
        "Це <b>ML Sandbox</b> — стенд для порівняння ML-фреймворків "
        "та обробки зображень за допомогою нейронних мереж.\n\n"
        + HELP_TEXT
    )


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(HELP_TEXT)
