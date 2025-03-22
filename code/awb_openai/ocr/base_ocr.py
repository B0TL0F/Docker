from abc import ABC, abstractmethod

class BaseOCR(ABC):
    """Abstract base class for OCR implementations."""

    @abstractmethod
    def extract_text(self, file_bytes):
        """Extract text from an image file."""
        pass
