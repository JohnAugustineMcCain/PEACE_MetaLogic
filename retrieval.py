
# retrieval.py
# Lightweight bag-of-words retriever for PEACE-C demos (no external deps).

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from collections import Counter
import math
import re

TOKEN = re.compile(r"[A-Za-z0-9_']+")

def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN.findall(text)]

def cosine(a: Dict[str,int], b: Dict[str,int]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a.get(k,0)*b.get(k,0) for k in set(a)|set(b))
    na = math.sqrt(sum(v*v for v in a.values()))
    nb = math.sqrt(sum(v*v for v in b.values()))
    return 0.0 if na==0 or nb==0 else dot/(na*nb)

@dataclass
class Doc:
    doc_id: str
    text: str
    meta: Dict[str, str] = field(default_factory=dict)

@dataclass
class Retriever:
    docs: List[Doc] = field(default_factory=list)

    def index(self, docs: List[Tuple[str,str,Dict[str,str]]]) -> None:
        for did, txt, meta in docs:
            self.docs.append(Doc(doc_id=did, text=txt, meta=meta))

    def search(self, query: str, k: int = 5) -> List[Doc]:
        qv = Counter(tokenize(query))
        scored = []
        for d in self.docs:
            dv = Counter(tokenize(d.text))
            scored.append((cosine(qv, dv), d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for s, d in scored[:k]]
