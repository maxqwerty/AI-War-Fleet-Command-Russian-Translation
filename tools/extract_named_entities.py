import os
import json
import sqlite3
from pathlib import Path

import tqdm
from langchain_core.prompts import ChatPromptTemplate

from helpers import collect_translation_items, chunks, get_llm, export_db_to_csv, COMMON_SYSTEM_PROMPT_PREFIX

SQLITE_DB = Path(os.getenv("SQLITE_DB", "named_entities.sqlite3"))
OUTPUT_CSV = Path(os.getenv("OUTPUT_CSV", "named_entities.csv"))

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "20"))


SYSTEM_PROMPT = COMMON_SYSTEM_PROMPT_PREFIX + """
Тебе дают пары:
- original: английский оригинал
- translation: русский перевод

Нужно найти именованные сущности и сопоставить их перевод.

Именованные сущности:
- имена персонажей
- названия фракций
- названия мест
- названия предметов
- названия навыков
- названия классов
- названия существ
- уникальные игровые термины
- названия событий, квестов, механик

Не добавляй обычные слова.
Не добавляй глаголы, прилагательные и общие существительные, если они не являются названием.
Не добавляй сущность, если не можешь уверенно сопоставить её перевод.

Ответ верни строго в JSON-массиве.
Каждый элемент массива должен иметь поля:
- original
- translation

JSON response format:
```
{{
  "named_entities":
    [
      {{
        "original": <ORIGINAL NAMED ENTITY>,
        "translation": <TRANSLATION NAMED ENTITY>
      }}
    ]
}}
```

Если сущностей нет, верни [].
"""


USER_PROMPT = """
Извлеки именованные сущности из следующих строк:

{items_json}
"""


def init_db():
    with sqlite3.connect(SQLITE_DB) as conn:

        conn.execute("""
            DROP TABLE IF EXISTS named_entities
        """)

        conn.execute("""
            CREATE TABLE named_entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original TEXT NOT NULL,
                translation TEXT NOT NULL
            )
        """)

        conn.commit()


def insert_entities(entities):
    with sqlite3.connect(SQLITE_DB) as conn:
        conn.executemany(
            """
            INSERT INTO named_entities (original, translation)
            VALUES (?, ?)
            """,
            [
                (
                    entity["original"].strip(),
                    entity["translation"].strip(),
                )
                for entity in entities
                if entity.get("original") and entity.get("translation")
            ],
        )

        conn.commit()


def extract_entities_batch(llm, batch):
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT),
    ])

    chain = prompt | llm

    response = chain.invoke({
        "items_json": json.dumps(batch, ensure_ascii=False, indent=2)
    })

    content = response.content.strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        print("LLM returned invalid JSON:")
        print(content)
        return []

    if not isinstance(data, dict) or ("named_entities" not in data):
        # print("ERROR: response json format: ", data)
        return []

    data = data["named_entities"]

    entities = []

    for row in data:
        if not isinstance(row, dict):
            continue

        original = str(row.get("original", "")).strip()
        translation = str(row.get("translation", "")).strip()

        if original and translation:
            entities.append({
                "original": original,
                "translation": translation,
            })

    return entities


def main():
    init_db()

    items = collect_translation_items()

    llm = get_llm()
    total_entities = 0

    with tqdm.tqdm(total=len(items)) as pbar:
        for batch in chunks(items, BATCH_SIZE):
            entities = extract_entities_batch(llm, batch)
            insert_entities(entities)

            total_entities += len(entities)
            pbar.update(len(batch))

    if total_entities:
        export_db_to_csv(SQLITE_DB, OUTPUT_CSV)
        print(f"Done. Entities found: {total_entities}")
    else:
        print(f"No entities found.")

    print(f"SQLite database: {SQLITE_DB}")


if __name__ == "__main__":
    main()