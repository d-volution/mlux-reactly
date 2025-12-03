from typing import List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from io import TextIOWrapper
import json


class Diagnostics:
    counters: dict[str, int]
    timepoints: dict[str, float]

    def __init__(self):
        self.counters = {}
        self.timepoints = {}

    def get_timepoint(self, key) -> datetime|None:
        if not key in self.timepoints:
            return None
        else:
            return datetime.fromtimestamp(self.timepoints[key])

    def set_timepoint(self, key: str, timepoint: datetime):
        self.timepoints[key] = timepoint.timestamp()

    def as_dict(self):
        return {
            'counters': self.counters,
            'timepoints': self.timepoints
        }

@dataclass
class QueryRecord:
    session: str
    query: str
    response: str
    diagnostics: Diagnostics

    def as_dict(self):
        data = asdict(self)
        data['diagnostics'] = self.diagnostics.as_dict()
        return data


class Recorder:
    session: str
    file: Optional[TextIOWrapper]

    queries: List[QueryRecord] = []

    def __init__(self, session=str(datetime.now().timestamp()), file=None):
        self.session = session
        self.file = file

    def record_query(self, query: str):
        now = datetime.now()
        record = QueryRecord(self.session, query, None, Diagnostics())
        record.diagnostics.set_timepoint('started_at', now)
        self.queries.append(record)

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