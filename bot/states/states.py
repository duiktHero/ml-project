from aiogram.fsm.state import State, StatesGroup


class ClassifyStates(StatesGroup):
    waiting_image = State()
    waiting_model = State()


class ColorizeStates(StatesGroup):
    waiting_grayscale = State()
    waiting_model = State()


class StylizeStates(StatesGroup):
    waiting_image = State()
    waiting_style = State()


class NeuralStyleStates(StatesGroup):
    waiting_image = State()
    waiting_style = State()
