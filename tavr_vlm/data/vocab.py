from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[\.-][A-Za-z0-9]+)*|[^\s]")


class WordVocab:
    pad = "<pad>"
    bos = "<bos>"
    eos = "<eos>"
    unk = "<unk>"

    def __init__(self, token_to_id: dict[str, int]):
        self.token_to_id = dict(token_to_id)
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        self.pad_id = self.token_to_id[self.pad]
        self.bos_id = self.token_to_id[self.bos]
        self.eos_id = self.token_to_id[self.eos]
        self.unk_id = self.token_to_id[self.unk]

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return TOKEN_RE.findall(text.lower())

    @classmethod
    def build(cls, texts: Iterable[str], min_freq: int = 1, max_size: int | None = None) -> "WordVocab":
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(cls.tokenize(text))
        specials = [cls.pad, cls.bos, cls.eos, cls.unk]
        tokens = [tok for tok, c in counter.most_common() if c >= min_freq and tok not in specials]
        if max_size is not None:
            tokens = tokens[: max(0, max_size - len(specials))]
        token_to_id = {tok: idx for idx, tok in enumerate(specials + tokens)}
        return cls(token_to_id)

    def encode(self, text: str, add_special: bool = True, max_len: int | None = None) -> list[int]:
        ids = [self.token_to_id.get(tok, self.unk_id) for tok in self.tokenize(text)]
        if add_special:
            ids = [self.bos_id] + ids + [self.eos_id]
        if max_len is not None:
            ids = ids[:max_len]
            if add_special and ids and ids[-1] != self.eos_id:
                ids[-1] = self.eos_id
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        specials = {self.pad, self.bos, self.eos, self.unk} if skip_special else set()
        toks = [self.id_to_token.get(int(i), self.unk) for i in ids]
        toks = [t for t in toks if t not in specials]
        text = " ".join(toks)
        text = text.replace(" ,", ",").replace(" .", ".").replace(" ;", ";").replace(" :", ":")
        return text

    def __len__(self) -> int:
        return len(self.token_to_id)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump({"token_to_id": self.token_to_id}, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "WordVocab":
        with Path(path).open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return cls(obj["token_to_id"])

    def ids_for_terms(self, terms: Iterable[str]) -> set[int]:
        ids: set[int] = set()
        for term in terms:
            for tok in self.tokenize(term):
                ids.add(self.token_to_id.get(tok, self.unk_id))
        ids.discard(self.unk_id)
        return ids
