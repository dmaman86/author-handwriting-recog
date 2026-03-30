
class SerializationError(Exception):
    """Exception raised for errors in the serialization process."""
    pass

class DeserializationError(Exception):
    """Exception raised for errors in the deserialization process."""
    pass

class UnsupportedFormatError(ValueError):
    """Exception raised for unsupported file formats."""
    pass