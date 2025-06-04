from typing import Protocol, Dict, List, Optional, Any, Tuple
from pydantic import BaseModel

class IDatabaseClient(Protocol):
    """
    Interface defining the contract for database clients.
    Any database client implementation must conform to this interface.
    """
    
    def __init__(self, config: Any) -> None:
        """
        Initialize the database client with configuration.
        
        Args:
            config: Configuration for database connection
        """
        ...
    
    def connect(self) -> bool:
        """
        Establish connection to the database.
        
        Returns:
            Success status
        """
        ...
    
    def disconnect(self) -> bool:
        """
        Close the database connection.
        
        Returns:
            Success status
        """
        ...
    
    def query(self, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute a query against the database.
        
        Args:
            query_params: Query parameters
            
        Returns:
            List of result records
        """
        ...
    
    def insert(self, data: Dict[str, Any]) -> bool:
        """
        Insert data into the database.
        
        Args:
            data: Data to insert
            
        Returns:
            Success status
        """
        ...
    
    def update(self, query_params: Dict[str, Any], data: Dict[str, Any]) -> bool:
        """
        Update records in the database.
        
        Args:
            query_params: Query parameters to identify records
            data: Data to update
            
        Returns:
            Success status
        """
        ...
    
    def delete(self, query_params: Dict[str, Any]) -> bool:
        """
        Delete records from the database.
        
        Args:
            query_params: Query parameters to identify records
            
        Returns:
            Success status
        """
        ... 