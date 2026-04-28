import os
import logging
from typing import List, Optional, Dict, Any, Union
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

# Порог косинусного сходства для определения дубликатов
# 0.9+ обычно означает очень высокую схожесть
SIMILARITY_THRESHOLD = 0.9

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PineconeManager")

class PineconeManager:
    """Класс для управления операциями с векторной базой данных Pinecone."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
        index_name: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_model: str = "text-embedding-3-small",
        vector_dimension: Optional[int] = None,
        base_url: Optional[str] = None
    ):
        """
        Инициализация менеджера Pinecone.
        
        Args:
            api_key: API ключ Pinecone (если None, загружается из .env)
            environment: Окружение Pinecone (если None, загружается из .env)
            index_name: Имя индекса (если None, загружается из .env)
            openai_api_key: API ключ OpenAI для создания эмбеддингов (если None, загружается из .env)
            openai_model: Модель OpenAI для создания эмбеддингов
            vector_dimension: Размерность векторов (если None, загружается из .env)
            base_url: Базовый URL для OpenAI API (если None, загружается из .env)
        """
        # Загрузка переменных окружения
        load_dotenv()
        
        # Инициализация Pinecone
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.environment = environment or os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME")
        self.vector_dimension = vector_dimension or int(os.getenv("VECTOR_DIMENSION", "1536"))
        
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY не найден. Укажите в параметрах или в .env файле.")
        if not self.index_name:
            raise ValueError("PINECONE_INDEX_NAME не найден. Укажите в параметрах или в .env файле.")
        
        # Инициализация клиента Pinecone
        self.pc = Pinecone(api_key=self.api_key)
        
        # Инициализация OpenAI для создания эмбеддингов
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY") or os.getenv("PROXY_API_KEY")
        self.openai_model = openai_model
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.openai_client = None
        
        if self.openai_api_key:
            self.openai_client = OpenAI(
                api_key=self.openai_api_key,
                base_url=self.base_url
            )
        
        # Получение индекса
        self.index = self.pc.Index(self.index_name)
        logger.info(f"PineconeManager инициализирован для индекса: {self.index_name}")

    def create_embedding(self, text: str) -> List[float]:
        """
        Создание эмбеддинга для текста.
        
        Args:
            text: Текст для эмбеддинга
            
        Returns:
            Список float значений эмбеддинга
        """
        if not self.openai_client:
            raise ValueError("OpenAI client не инициализирован. Проверьте API ключ.")
        
        text = text.replace("\n", " ")
        response = self.openai_client.embeddings.create(
            input=[text],
            model=self.openai_model
        )
        return response.data[0].embedding

    def _check_similarity(self, vector: List[float]) -> Optional[Dict[str, Any]]:
        """
        Внутренний метод для проверки сходства вектора с существующими.
        
        Args:
            vector: Вектор для проверки
            
        Returns:
            Словарь с 'id' и 'score' самого похожего вектора, если score > SIMILARITY_THRESHOLD
        """
        results = self.query_by_vector(vector, top_k=1)
        if results and results.get("matches"):
            match = results["matches"][0]
            logger.debug(f"Наилучшее сходство: {match['score']} (ID: {match['id']})")
            if match["score"] >= SIMILARITY_THRESHOLD:
                return {
                    "id": match["id"],
                    "score": match["score"]
                }
        return None

    def upsert_vector(
        self, 
        vector_id: str, 
        vector: List[float], 
        metadata: Optional[Dict[str, Any]] = None,
        check_similarity: bool = True
    ) -> Dict[str, Any]:
        """
        Запись вектора в Pinecone с проверкой косинусного сходства.
        
        Args:
            vector_id: Уникальный идентификатор вектора
            vector: Вектор для записи
            metadata: Метаданные (опционально)
            check_similarity: Проверять ли сходство перед записью (по умолчанию True)
            
        Returns:
            Словарь с информацией о результате:
            - 'action': 'inserted' (новая запись), 'updated' (обновлена существующая), 'skipped' (пропущена)
            - 'similarity_score': значение сходства (если было найдено)
            - 'existing_id': ID существующего вектора (если был найден)
        """
        result = {
            "action": "inserted",
            "similarity_score": None,
            "existing_id": None
        }
        
        # Проверяем сходство перед записью
        if check_similarity:
            similar = self._check_similarity(vector)
            if similar:
                # Высокое сходство - обновляем существующий слот
                existing_id = similar["id"]
                result["action"] = "updated"
                result["similarity_score"] = similar["score"]
                result["existing_id"] = existing_id
                
                # Обновляем существующий вектор
                vectors_to_upsert = [{
                    "id": existing_id,
                    "values": vector,
                    "metadata": metadata or {}
                }]
                self.index.upsert(vectors=vectors_to_upsert)
                logger.info(f"Обновлен существующий вектор {existing_id} (сходство: {similar['score']:.4f})")
                return result
        
        # Низкое сходство - записываем как новую информацию
        vectors_to_upsert = [{
            "id": vector_id,
            "values": vector,
            "metadata": metadata or {}
        }]
        self.index.upsert(vectors=vectors_to_upsert)
        logger.info(f"Добавлен новый вектор: {vector_id}")
        return result

    def upsert_vectors(self, vectors: List[Dict[str, Any]]):
        """
        Запись нескольких векторов в Pinecone.
        
        Args:
            vectors: Список словарей с ключами 'id', 'values', 'metadata'
        """
        # Преобразуем формат если нужно, Pinecone принимает список кортежей или словарей
        self.index.upsert(vectors=vectors)

    def upsert_document(self, vector_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Запись документа (текст преобразуется в эмбеддинг).
        
        Args:
            vector_id: ID документа
            text: Текст документа
            metadata: Метаданные (текст будет добавлен в метаданные автоматически)
            
        Returns:
            Результат выполнения upsert_vector
        """
        embedding = self.create_embedding(text)
        if metadata is None:
            metadata = {}
        metadata["text"] = text
        return self.upsert_vector(vector_id, embedding, metadata)

    def upsert_documents(self, documents: List[Dict[str, Any]]):
        """
        Запись нескольких документов.
        
        Args:
            documents: Список словарей с 'id', 'text' и опционально 'metadata'
        """
        vectors_to_upsert = []
        for doc in documents:
            embedding = self.create_embedding(doc["text"])
            metadata = doc.get("metadata", {})
            metadata["text"] = doc["text"]
            vectors_to_upsert.append({
                "id": doc["id"],
                "values": embedding,
                "metadata": metadata
            })
        self.upsert_vectors(vectors_to_upsert)

    def query_by_vector(
        self, 
        vector: List[float], 
        top_k: int = 10, 
        filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Поиск по вектору.
        
        Args:
            vector: Вектор для поиска
            top_k: Количество результатов
            filter: Фильтр метаданных
            include_metadata: Включать ли метаданные в ответ
            
        Returns:
            Результаты поиска
        """
        return self.index.query(
            vector=vector,
            top_k=top_k,
            filter=filter,
            include_metadata=include_metadata
        )

    def query_by_text(
        self, 
        text: str, 
        top_k: int = 10, 
        filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Поиск по тексту.
        
        Args:
            text: Текст для поиска
            top_k: Количество результатов
            filter: Фильтр метаданных
            include_metadata: Включать ли метаданные в ответ
            
        Returns:
            Результаты поиска
        """
        embedding = self.create_embedding(text)
        return self.query_by_vector(
            vector=embedding,
            top_k=top_k,
            filter=filter,
            include_metadata=include_metadata
        )

    def fetch_vectors(self, ids: List[str]) -> Dict[str, Any]:
        """
        Получение векторов по их ID.
        
        Args:
            ids: Список ID векторов
            
        Returns:
            Словарь с найденными векторами
        """
        return self.index.fetch(ids=ids)

    def delete(self, ids: List[str], namespace: Optional[str] = None):
        """
        Удаление векторов по ID.
        
        Args:
            ids: Список ID для удаления
            namespace: Пространство имен (опционально)
        """
        self.index.delete(ids=ids, namespace=namespace)

    def delete_by_filter(self, filter: Dict[str, Any], namespace: Optional[str] = None):
        """
        Удаление векторов по фильтру.
        
        Args:
            filter: Фильтр для удаления
            namespace: Пространство имен (опционально)
        """
        self.index.delete(filter=filter, namespace=namespace)

    def delete_all(self, namespace: Optional[str] = None):
        """
        Удаление всех векторов в индексе или пространстве имен.
        
        Args:
            namespace: Пространство имен (опционально)
        """
        self.index.delete(delete_all=True, namespace=namespace)

    def describe_index_stats(self) -> Dict[str, Any]:
        """
        Получение статистики индекса.
        
        Returns:
            Статистика индекса
        """
        return self.index.describe_index_stats()

    def update_metadata(self, vector_id: str, metadata: Dict[str, Any], namespace: Optional[str] = None):
        """
        Обновление метаданных вектора.
        
        Args:
            vector_id: ID вектора
            metadata: Новые метаданные
            namespace: Пространство имен (опционально)
        """
        self.index.update(id=vector_id, set_metadata=metadata, namespace=namespace)

if __name__ == "__main__":
    # Тестовый блок для ручной проверки
    print("--- Тестирование PineconeManager ---")
    try:
        manager = PineconeManager()
        
        # 1. Проверка статистики
        print("\n1. Получение статистики индекса...")
        stats = manager.describe_index_stats()
        print(f"Статистика: {stats}")
        
        # 2. Тестовая запись документа
        test_id = "test_doc_1"
        test_text = "Сегодня солнечная погода, и я изучаю работу с векторными базами данных."
        print(f"\n2. Тестовая запись документа '{test_id}'...")
        res = manager.upsert_document(test_id, test_text, {"type": "test"})
        print(f"Результат записи: {res}")
        
        # 3. Проверка сходства (дубликат)
        print("\n3. Проверка записи дубликата (похожий текст)...")
        duplicate_text = "Сегодня очень солнечная погода, изучаю векторные БД."
        res_dup = manager.upsert_document("test_doc_2", duplicate_text, {"type": "test"})
        print(f"Результат (должен быть updated): {res_dup}")
        
        # 4. Поиск по тексту
        query = "какая сегодня погода?"
        print(f"\n4. Поиск по тексту: '{query}'...")
        search_res = manager.query_by_text(query, top_k=2)
        print("Результаты поиска:")
        for match in search_res['matches']:
            print(f" - ID: {match['id']}, Score: {round(match['score'], 3)}, Text: {match['metadata'].get('text')}")
            
        print("\n--- Тестирование успешно завершено ---")
        
    except Exception as e:
        print(f"\n❌ Ошибка при тестировании: {e}")
