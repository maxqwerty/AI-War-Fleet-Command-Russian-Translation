import os
import csv
import json
from pathlib import Path

import tqdm
from langchain_core.prompts import ChatPromptTemplate

from helpers import XML_DIR, collect_translation_items, chunks, get_llm

OUTPUT_CSV = Path(os.getenv("OUTPUT_CSV", "fix_suggestions.csv"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "20"))


SYSTEM_PROMPT = """
Ты профессиональный редактор русской локализации видеоигр.
Ты сейчас работаешь над переводом 

Тебе дают список строк:
- file: имя XML-файла
- id: id XML-тега
- original: оригинальный английский текст из XML-комментария
- translation: русский перевод из содержимого XML-тега

Твоя задача - найти ТОЛЬКО те переводы, которые требуют исправления.

Не добавляй строку в отчёт, если перевод нормальный, естественный и передаёт смысл.

Ищи такие проблемы:
- неверный смысл
- слишком вольный перевод
- пропущенная важная часть смысла
- неправильный игровой термин
- странная или неестественная русская формулировка
- перевод выглядит как другая сущность, предмет, класс, навык или действие
- перепутано единственное/множественное число
- перепутан род, стиль или тон
- оригинал является названием, а перевод звучит как описание
- перевод слишком общий по сравнению с оригиналом

Не придирайся к допустимым вариантам перевода.
Не исправляй хорошие переводы.
Не добавляй комментарии о стиле, если перевод приемлем.

Ответ верни строго в JSON-массиве.
Каждый элемент массива должен иметь поля:
- file
- id
- original
- translation
- error_description

Если ошибок нет, верни пустой массив [].
"""


USER_PROMPT = """
Проверь следующие строки перевода:

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

    valid_rows = []

    for row in data:
        if not isinstance(row, dict):
            continue

        if all(key in row for key in ["file", "id", "original", "translation", "error_description"]):
            valid_rows.append(row)

    return valid_rows


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

    suggestions = []
    with tqdm.tqdm(total=len(items)) as pbar:
        for batch in chunks(items, BATCH_SIZE):
            suggestions.extend(validate_batch(llm, batch))
            pbar.update(len(batch))

    if suggestions:
        write_csv(suggestions)
        print(f"Done. Translation issues: {len(suggestions)} cases were saved to {OUTPUT_CSV}")
    else:
        print(f"No suggestions found.")

if __name__ == "__main__":
    main()