from abc import ABC, abstractmethod

class Agent(ABC):
    @abstractmethod
    def query(user_question: str) -> str:
        pass
    
class ResponseStream(ABC):
    @abstractmethod
    def write(self, text: str):
        pass