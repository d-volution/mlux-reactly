from typing import List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from io import TextIOWrapper
import json
from .diagnostics import Diagnostics



@dataclass
class QueryRecord:
    session: str
    query: str
    response: Optional[str]
    diagnostics: Diagnostics

    def as_dict(self):
        data = asdict(self)
        data['diagnostics'] = self.diagnostics.as_dict()
        return data


class Recorder:
    session: str
    file: Optional[TextIOWrapper]

    queries: List[QueryRecord] = []

    def __init__(self, session: str = "", file=None):
        self.session = session=str(datetime.now().timestamp()) + session
        self.file = file

    def record_query(self, query: str) -> QueryRecord:
        now = datetime.now()
        record = QueryRecord(self.session, query, None, Diagnostics())
        record.diagnostics.set_timepoint('started_at', now)
        self.queries.append(record)
        return record

    def on_response(self, response: str):
        record = self.queries[-1]
        record.diagnostics.set_timepoint('finished_at', datetime.now())
        record.response = response

        if self.file != None:
            self.file.write(json.dumps(record.as_dict()) + "\n")



class ZeroRecorder(Recorder):
    "Recorder for not recording anything."
    
    def record_query(self, query: str):
        return

    def on_response(self, response: str):
        return