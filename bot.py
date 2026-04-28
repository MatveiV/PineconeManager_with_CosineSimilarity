import os
import telebot
import logging
from dotenv import load_dotenv
from pinecone_manager import PineconeManager

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TelegramBot")

# Инициализация бота
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN не найден в .env")

bot = telebot.TeleBot(BOT_TOKEN)

# Инициализация менеджера Pinecone
# Параметры будут взяты из .env автоматически
pm = PineconeManager()

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    logger.info(f"Команда {message.text} от пользователя {message.from_user.id}")
    help_text = (
        "Привет! Я бот-помощник с долговременной памятью.\n\n"
        "Я запоминаю всё, что ты мне пишешь, и проверяю на дубликаты.\n"
        "Команды:\n"
        "/search <текст> - поиск по моей памяти\n"
        "/stats - статистика хранилища\n"
        "/clear - очистить всю память (осторожно!)"
    )
    bot.reply_to(message, help_text)

@bot.message_handler(commands=['stats'])
def show_stats(message):
    logger.info(f"Запрос статистики от {message.from_user.id}")
    try:
        stats = pm.describe_index_stats()
        text = f"Статистика индекса:\n`{stats}`"
        bot.reply_to(message, text, parse_mode="Markdown")
    except Exception as e:
        bot.reply_to(message, f"Ошибка при получении статистики: {str(e)}")

@bot.message_handler(commands=['search'])
def search_memory(message):
    query = message.text.replace("/search", "").strip()
    if not query:
        bot.reply_to(message, "Пожалуйста, укажите текст для поиска после команды /search")
        return
    
    bot.send_chat_action(message.chat.id, 'find_location')
    try:
        results = pm.query_by_text(query, top_k=3)
        if not results['matches']:
            bot.reply_to(message, "Ничего похожего в памяти не найдено.")
            return
        
        response = "Вот что я нашел в памяти:\n\n"
        for i, match in enumerate(results['matches'], 1):
            text = match['metadata'].get('text', 'Без текста')
            score = round(match['score'], 3)
            response += f"{i}. {text} (сходство: {score})\n"
        
        bot.reply_to(message, response)
    except Exception as e:
        bot.reply_to(message, f"Ошибка при поиске: {str(e)}")

@bot.message_handler(commands=['clear'])
def clear_memory(message):
    try:
        pm.delete_all()
        bot.reply_to(message, "Вся память была успешно очищена.")
    except Exception as e:
        bot.reply_to(message, f"Ошибка при очистке памяти: {str(e)}")

@bot.message_handler(func=lambda message: True)
def handle_all_messages(message):
    # Игнорируем команды
    if message.text.startswith('/'):
        return

    # Подготовка ID и текста (берем только чистый текст сообщения)
    vector_id = f"msg_{message.chat.id}_{message.message_id}"
    text = message.text

    try:
        logger.info(f"Обработка сообщения от {message.from_user.id}: {text[:50]}...")
        # Запись в Pinecone с проверкой на сходство. 
        # Метаданные со служебными полями больше не передаем, 
        # менеджер сохранит только сам текст.
        result = pm.upsert_document(vector_id, text)
        
        action = result['action']
        score = result['similarity_score']
        
        if action == "inserted":
            logger.info(f"Сообщение {vector_id} сохранено как новое.")
            response = "✅ Запомнил новую информацию."
        elif action == "updated":
            logger.info(f"Сообщение {vector_id} обновило существующий вектор {result['existing_id']} (score: {score})")
            response = f"🔄 Похожее уже было (сходство: {round(score, 3)}). Обновил существующую запись."
        else:
            response = "ℹ️ Информация пропущена."
            
        bot.reply_to(message, response)
        
    except Exception as e:
        logger.error(f"Error upserting to Pinecone: {e}")
        bot.reply_to(message, f"Произошла ошибка при сохранении в память: {str(e)}")

if __name__ == "__main__":
    print("Бот запущен...")
    bot.infinity_polling()
