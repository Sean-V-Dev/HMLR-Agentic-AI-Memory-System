"""
Core Exception Classes for HMLR
"""

class HMLRError(Exception):
    """Base exception for all HMLR errors"""
    pass

class ApiConnectionError(HMLRError):
    """Raised when external API connection fails"""
    pass

class ModelNotAvailableError(HMLRError):
    """Raised when a specific requested model is not found"""
    pass

class ConfigurationError(HMLRError):
    """Raised when there is a configuration issue"""
    pass

class RetrievalError(HMLRError):
    """Base exception for retrieval system failures"""
    pass

class VectorDatabaseError(RetrievalError):
    """Raised when vector storage/search fails"""
    pass

class LatticeGovernorError(RetrievalError):
    """Raised when the Governor logic fails"""
    pass


class StorageError(HMLRError):
    """Base exception for storage layer failures"""
    pass


class StorageWriteError(StorageError):
    """Raised when writes to persistent storage fail"""
    pass
