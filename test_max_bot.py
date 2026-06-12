import asyncio
import logging
from maxapi import Bot, Dispatcher, F
from maxapi.types import MessageCreated

# --- 1. Настройка логирования ---
# Включаем логирование, чтобы видеть, что происходит и отлавливать ошибки.
logging.basicConfig(level=logging.INFO)

# --- 2. Инициализация бота и диспетчера ---
# ВСТАВЬТЕ СЮДА ВАШ ТОКЕН! Получить его можно у @MasterBot в MAX.
TOKEN = 'ВАШ_ТОКЕН_БОТА'
bot = Bot(token=TOKEN)
dp = Dispatcher()

# --- 3. Обработчики событий ---

# Событие при нажатии на кнопку "Начать"
@dp.bot_started()
async def on_start(event):
    await event.bot.send_message(
        chat_id=event.chat_id,
        text="Привет! Я тестовый бот. Отправь мне любое сообщение."
    )

# Команда /start
@dp.message_created(F.message.body.text)
async def hello(event: MessageCreated):
    # Проверяем, является ли сообщение командой /start
    if event.message.body.text == '/start':
        await event.message.answer("Бот запущен! Просто отправь мне сообщение.")
    else:
        # Отправляем ответ 'Эхо' на любое другое текстовое сообщение
        await event.message.answer(f"Эхо: {event.message.body.text}")

# --- 4. Запуск бота (main) ---
async def main():
    # Удаляем старые вебхуки, если они были установлены
    await bot.delete_webhook()
    # Запускаем получение обновлений методом Long Polling
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())