from .types import DiagnosticHandler


class DiagnosticHandlerDefault(DiagnosticHandler):
    def event(name: str):
        print(f"got event {name}")