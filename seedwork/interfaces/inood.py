from typing import Protocol, Any

class INood(Protocol):
    def __init__(self, *args: Any, **kwargs: Any):
        ...
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ...
        
