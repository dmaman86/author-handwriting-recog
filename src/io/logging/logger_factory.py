import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class LoggerFactory:
    """
    Centralized logger factory with multiple output handlers.

    Features:
    - Console output (colorized, optional)
    - File output (rotating, with timestamps)
    - Configurable log levels for handler
    - Singleton pattern per logger name
    """

    _loggers: dict[str, logging.Logger] = {}
    _default_config = {
        "level": logging.INFO,
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "date_format": "%Y-%m-%d %H:%M:%S",
        "console": True,
        "file": None,  # Path or None
        "file_level": logging.DEBUG,
        "console_level": logging.INFO,
    }

    @classmethod
    def get_logger(
        cls,
        name: str,
        level: int = logging.INFO,
        log_dir: Optional[str | Path] = None,
        console: bool = True,
        file_prefix: str = "app",
    ) -> logging.Logger:
        """
        Get or create a logger with the specified configuration.

        Args:
            name: Logger name (usually __name__)
            level: Global logging level
            log_dir: Directory for log files (None = no file logging)
            console: Enable console output
            file_prefix: Prefix for log filename

        Returns:
            Configured logger instance

        Example:
            >>> logger = LoggerFactory.get_logger(__name__, log_dir="./logs")
            >>> logger.info("Processing started")
        """
        # Return existing logger if already created
        if name in cls._loggers:
            return cls._loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.handlers.clear()  # Remove any existing handlers

        if console:
            console_handler = cls._create_console_handler()
            logger.addHandler(console_handler)

        if log_dir:
            file_handler = cls._create_file_handler(log_dir, file_prefix)
            logger.addHandler(file_handler)

        # Prevent log propagation to root logger
        logger.propagate = False

        cls._loggers[name] = logger
        return logger

    @classmethod
    def _create_console_handler(cls) -> logging.StreamHandler:
        """Create a colorized console handler."""
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)

        formatter = ColorizedFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        return handler

    @classmethod
    def _create_file_handler(
        cls, log_dir: str | Path, prefix: str
    ) -> logging.FileHandler:
        """Create a file handler with timestamped filename."""
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{prefix}_{timestamp}.log"

        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setLevel(logging.DEBUG)  # File gets everything

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        return handler

    @classmethod
    def get_notebook_logger(
        cls, name: str, log_dir: Optional[str | Path] = None
    ) -> logging.Logger:
        """
         Get a logger optimized for Jupyter notebooks.

        - Shorter format
        - No duplicate output in notebooks
        - Optional file logging
        """
        if name in cls._loggers:
            return cls._loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        if log_dir:
            file_handler = cls._create_file_handler(log_dir, "notebook")
            logger.addHandler(file_handler)

        logger.propagate = False
        cls._loggers[name] = logger
        return logger

    @classmethod
    def reset(cls) -> None:
        """Reset all loggers (useful for testing)."""
        for logger in cls._loggers.values():
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
        cls._loggers.clear()


class ColorizedFormatter(logging.Formatter):
    """
    Custom formatter with color codes for terminal output.
    """

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",
    }

    def format(self, record: logging.LogRecord) -> str:
        # Add color to levelname
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
            )

        return super().format(record)

