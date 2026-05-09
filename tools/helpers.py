import csv
import os
import sqlite3
from pathlib import Path
from xml.etree import ElementTree as ET

from langchain_ollama import ChatOllama

XML_DIR = Path(os.getenv("XML_DIR", "../test_trans"))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")


COMMON_SYSTEM_PROMPT_PREFIX = """
Ты профессиональный редактор русской локализации видеоигр.
Ты сейчас работаешь над переводом игры под названием AI WAR: Fleet Command.
Тебе нужно обеспечить качество перевода.
""".strip()


if not XML_DIR.exists():
    raise FileNotFoundError(f"XML_DIR does not exist: {XML_DIR}")


def collect_translation_items():
    items = []

    for xml_path in XML_DIR.rglob("*.xml"):
        tree = parse_xml_keep_comments(xml_path)
        root = tree.getroot()

        previous_comment = None

        for node in list(root):
            if node.tag is ET.Comment:
                previous_comment = extract_original_from_comment(node.text)
                continue

            tag_id = node.attrib.get("id")
            translation = (node.text or "").strip()

            if previous_comment and tag_id and translation:
                items.append({
                    "file": str(xml_path.relative_to(XML_DIR)),
                    "id": tag_id,
                    "original": previous_comment,
                    "translation": translation,
                })

            previous_comment = None

    return items


def chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def get_llm():
    return ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0,
        format="json",
    )


def parse_xml_keep_comments(path: Path) -> ET.ElementTree:
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    return ET.parse(path, parser=parser)


def extract_original_from_comment(comment_text: str) -> str:
    text = (comment_text or "").strip()

    if text.startswith("EN:"):
        return text[3:].strip()

    return text


def export_db_to_csv(db_name, output_csv):
    with sqlite3.connect(db_name) as conn, output_csv.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["original", "translation"])

        rows = conn.execute("""
            SELECT original, translation
            FROM named_entities
            ORDER BY id
        """)

        writer.writerows(rows)

    print(f"CSV exported to: {output_csv}")
