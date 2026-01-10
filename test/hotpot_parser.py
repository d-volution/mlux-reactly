from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class HotpotDocument:
    title: str
    sentences: List[str]

    def text(self) -> str:
        return "\n".join(self.sentences)


@dataclass
class HotpotDatapoint:
    id: str
    question: str
    answer: str
    documents: List[HotpotDocument]


def parse_hotspot_file(file_data: List[Dict[str, Any]]) -> List[HotpotDatapoint]:
    datapoints = []
    for dp in file_data:
        datapoints.append(HotpotDatapoint(
            dp.get('_id'),
            dp.get('question'),
            dp.get('answer'),
            [HotpotDocument(doc[0], doc[1]) for doc in dp.get('context')]
        ))
    return datapoints