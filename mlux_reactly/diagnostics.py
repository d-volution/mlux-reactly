from datetime import datetime




class Diagnostics:
    counters: dict[str, int]
    timepoints: dict[str, float]

    def __init__(self):
        self.counters = {}
        self.timepoints = {}

    def as_dict(self):
        return {
            'counters': self.counters,
            'timepoints': self.timepoints
        }
    
    def get_timepoint(self, key) -> datetime|None:
        if not key in self.timepoints:
            return None
        else:
            return datetime.fromtimestamp(self.timepoints[key])

    def set_timepoint(self, key: str, timepoint: datetime):
        self.timepoints[key] = timepoint.timestamp()
    
    def increment_counter(self, key: str):
        if key not in self.counters:
            self.counters[key] = 0
        self.counters[key] += 1
