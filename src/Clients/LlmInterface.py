from abc import ABC, abstractmethod

class LlmInterface(ABC):
    
    @abstractmethod
    def start_dialog(self) -> None:
        pass