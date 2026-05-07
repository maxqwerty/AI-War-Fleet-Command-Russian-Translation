import os
import csv
import json
from pathlib import Path

import tqdm
from langchain_core.prompts import ChatPromptTemplate

from helpers import XML_DIR, collect_translation_items, chunks, get_llm

OUTPUT_CSV = Path(os.getenv("OUTPUT_CSV", "sentence_structure_issues.csv"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "15"))


SYSTEM_PROMPT = """
Ты редактор русской локализации видеоигр.

Тебе дают пары:
- original: английский оригинал из XML-комментария
- translation: русский перевод из XML-тега

Нужно проверить ТОЛЬКО структуру предложений:
- сохранено ли количество предложений
- не объединены ли несколько предложений в одно
- не разделено ли одно предложение на несколько
- сохранён ли порядок смысловых предложений
- нет ли перестановки смысловых частей между предложениями

Не оценивай качество перевода.
Не проверяй стиль.
Не проверяй терминологию.
Не добавляй строку в отчёт, если структура допустимая.

Важно:
- Заголовки, короткие названия, одиночные фразы и строки без полноценного предложения обычно считаются корректными.
- Допускается небольшая грамматическая адаптация.
- Допускается замена точки на восклицательный или вопросительный знак, если структура не изменилась.
- Не считай ошибкой отсутствие точки в конце короткой строки.
- Проверяй именно соответствие структуры оригиналу.

Ответ верни строго в JSON-массиве.
Каждый элемент массива должен иметь поля:
- file
- id
- original
- translation
- error_description

Если проблем нет, верни [].
"""


USER_PROMPT = """
Проверь структуру предложений в следующих строках:

{items_json}
"""


def validate_batch(llm, batch):
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

    if not isinstance(data, list):
        return []

    rows = []

    for row in data:
        if not isinstance(row, dict):
            continue

        if all(key in row for key in ["file", "id", "original", "translation", "error_description"]):
            rows.append(row)

    return rows


def write_csv(rows):
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "tag_id",
                "original",
                "translation",
                "description",
            ],
        )

        writer.writeheader()

        for row in rows:
            writer.writerow({
                "file": row["file"],
                "tag_id": row["id"],
                "original": row["original"],
                "translation": row["translation"],
                "description": row["error_description"],
            })


def main():
    items = collect_translation_items()

    llm = get_llm()

    issues = []
    with tqdm.tqdm(total=len(items)) as pbar:
        for batch in chunks(items, BATCH_SIZE):
            issues.extend(validate_batch(llm, batch))
            pbar.update(len(batch))

    if issues:
        write_csv(issues)
        print(f"Done. Issues found: {len(issues)} cases were saved to {OUTPUT_CSV}")
    else:
        print(f"No issues found.")


if __name__ == "__main__":
    main()