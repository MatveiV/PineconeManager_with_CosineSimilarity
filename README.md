# Pinecone Manager with Cosine Similarity

Система управления долговременной памятью на базе Pinecone и OpenAI для Telegram-ботов. 
Основная особенность — автоматическая проверка входящих сообщений на дубликаты с использованием косинусного сходства (Cosine Similarity).

## 🚀 Основные возможности

- **Интеллектуальный Upsert**: Перед сохранением система проверяет, нет ли уже похожей информации в базе. Если сходство выше порога, запись обновляется, а не дублируется.
- **Чистые метаданные**: В векторном хранилище сохраняется только оригинальный текст сообщения пользователя без лишнего служебного шума (ID, таймстампы и т.д.).
- **Поддержка ProxyAPI**: Возможность указать `OPENAI_BASE_URL` для работы через прокси-серверы.
- **OpenAI Embeddings**: Использование современной модели `text-embedding-3-small`.
- **Telegram Интеграция**: Готовый бот-ассистент с обработкой исключений (например, отсутствие username у пользователя).
- **Логирование и Тестирование**: Встроенный блок для ручной проверки работы ядра системы.

## ⚙️ Настройка (Environment Variables)

Создайте файл `.env` в корне проекта:

```env
# Pinecone
PINECONE_API_KEY=ваш_ключ
PINECONE_INDEX_NAME=имя_индекса
PINECONE_ENVIRONMENT=us-east-1

# OpenAI / ProxyAPI
OPENAI_API_KEY=ваш_ключ
OPENAI_BASE_URL=https://api.proxyapi.ru/openai/v1  # Опционально
EMBEDDING_MODEL=text-embedding-3-small

# Telegram
TELEGRAM_BOT_TOKEN=токен_от_BotFather
```

## 🛠 Архитектура

### C4 System Context Diagram

<div align="center">
<div style="background-color: white; padding: 20px; border-radius: 10px;">

```mermaid
C4Context
    title System Context diagram for PineconeManager System
    
    Person(user, "Пользователь Telegram", "Общается с ботом, отправляет сообщения.")
    System(bot_system, "Pinecone Manager Bot", "Интерфейс чата и управление памятью.")
    
    System_Ext(telegram_api, "Telegram Bot API", "Платформа мессенджера.")
    System_Ext(openai_api, "OpenAI API", "Создание эмбеддингов.")
    System_Ext(pinecone_db, "Pinecone DB", "Векторная база данных.")

    Rel(user, telegram_api, "Отправляет", "HTTPS")
    Rel_D(telegram_api, bot_system, "Передает", "HTTPS")
    Rel_R(bot_system, openai_api, "Запрашивает", "REST")
    Rel_L(bot_system, pinecone_db, "Ищет/Сохраняет", "gRPC")
    Rel_U(bot_system, telegram_api, "Отвечает", "HTTPS")
```

</div>
</div>

### UML Sequence Diagram: Логика "Умной памяти"

<div align="center">
<div style="background-color: white; padding: 20px; border-radius: 10px;">

```mermaid
sequenceDiagram
    participant U as Пользователь
    participant B as Telegram Bot
    participant PM as PineconeManager
    participant OA as OpenAI API
    participant P as Pinecone DB

    U->>B: Сообщение "Хочу на Марс"
    B->>PM: upsert_document("Хочу на Марс")
    PM->>OA: create_embedding
    OA-->>PM: vector
    
    PM->>P: query (найти самый похожий)
    P-->>PM: match (score: 0.94)
    
    Note over PM: THRESHOLD = 0.9
    
    alt score > 0.9
        PM->>P: upsert (ID существующего вектора)
        PM-->>B: {action: "updated", score: 0.94}
        B->>U: "🔄 Похожее уже было, обновил запись"
    else score <= 0.9
        PM->>P: upsert (новый ID)
        PM-->>B: {action: "inserted"}
        B->>U: "✅ Запомнил новую информацию"
    end
```

</div>
</div>

## 📦 Установка и Запуск

1. **Установите зависимости**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Проверьте ядро (PineconeManager)**:
   ```bash
   python pinecone_manager.py
   ```
   *Запустится встроенный тест: проверка связи, запись документа и проверка дубликата.*

3. **Запустите бота**:
   ```bash
   python bot.py
   ```

## 📂 Структура проекта

- [pinecone_manager.py](file:///c:/GitHub/PineconeManagerCosSim/pinecone_manager.py): Класс `PineconeManager` с логикой косинусного сходства и поддержкой OpenAI.
- [bot.py](file:///c:/GitHub/PineconeManagerCosSim/bot.py): Реализация Telegram-бота на `pyTelegramBotAPI`.
- [requirements.txt](file:///c:/GitHub/PineconeManagerCosSim/requirements.txt): Зависимости проекта.
- [.env](file:///c:/GitHub/PineconeManagerCosSim/.env): Конфигурация (необходимо создать).

## 🖼 Скриншоты работы

### Интерфейс Telegram-бота
| Похожее сообщение | Обновление записи | Статистика |
|:---:|:---:|:---:|
| ![Похожее сообщение](Screenshots/Похожее%20сообщение.png) | ![Обновление записи](Screenshots/Обновление%20записи%20похожим.png) | ![Stats](Screenshots/Stats%20command.png) |

### Работа в терминале (Логирование)
![Terminal](Screenshots/Terminal%20Sunny%20Update.png)

### Панель управления Pinecone
![Pinecone Index](Screenshots/Pinecone%20Index.png)
