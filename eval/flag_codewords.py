"""Flag golden-set rows whose question contains code-like words.

Usage: python eval/flag_codewords.py [path]   (default: eval/golden_draft.jsonl)
"""
import json
import re
import sys

# code-ish tokens: calls, attributes, literals, identifiers with underscores, type hints, flags
CODE = re.compile(
    r"`|->|==|!=|\(\)|=|;|\.(py|json|md)\b|"
    r"\b(async|def|await|yield|None|True|False|str|int|float|bool|dict|list|"
    r"response_model|Field|Query|Path|Body|Header|Cookie|Form|File|UploadFile|"
    r"Depends|HTTPException|APIRouter|SQLModel|Annotated)\b|"
    r"\b[a-zA-Z]+_[a-zA-Z_]+\b|\bX-[A-Za-z-]+\b"
)


def main(path="eval/golden_draft.jsonl"):
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    flagged = [r for r in rows if CODE.search(r["question"])]
    print(f"{len(flagged)} of {len(rows)} rows contain code-like words")
    for r in flagged[:3]:
        print(f"  {r['id']}: {r['question']}")


if __name__ == "__main__":
    main(*sys.argv[1:])
