import datetime as dt
import json
import logging

from rich.console import Console
from rich.table import Table
from typing_extensions import override

LOG_RECORD_BUILTIN_ATTRS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}


class MyJSONFormatter(logging.Formatter):
    def __init__(
        self,
        *,
        fmt_keys: dict[str, str] | None = None,
    ):
        super().__init__()
        self.fmt_keys = fmt_keys if fmt_keys is not None else {}

    @override
    def format(self, record: logging.LogRecord) -> str:
        message = self._prepare_log_dict(record)
        return json.dumps(message, default=str)

    def _prepare_log_dict(self, record: logging.LogRecord):
        # Base structure of the log message
        log_dict = {
            "timestamp": dt.datetime.fromtimestamp(
                record.created, tz=dt.timezone.utc
            ).isoformat(),
        }

        # If the log message is a dictionary, handle it specially to ensure it stays in JSON format
        if isinstance(record.msg, dict):
            # Directly assign the dictionary to a specific key or merge with log_dict
            log_dict["message"] = record.msg
        else:
            # For non-dict messages, just use the message as is
            log_dict["message"] = record.getMessage()

        # Add exception info if present
        if record.exc_info:
            log_dict["exc_info"] = self.formatException(record.exc_info)

        # Add stack info if present
        if record.stack_info:
            log_dict["stack_info"] = self.formatStack(record.stack_info)

        # Add any custom fields defined in fmt_keys
        for key, val in self.fmt_keys.items():
            if key == "message" and isinstance(log_dict.get("message"), dict):
                # Potentially merge or handle the dictionary message differently here
                # For example, you might want to nest additional info under a different key
                continue
            if hasattr(record, val):
                log_dict[key] = getattr(record, val)

        # Include any additional custom attributes added to the log record
        for attr in record.__dict__:
            if attr not in LOG_RECORD_BUILTIN_ATTRS and attr not in log_dict:
                log_dict[attr] = record.__dict__[attr]

        return log_dict


from rich.logging import RichHandler


class RichJSONHandler(logging.Handler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.console = Console()

    def emit(self, record):
        try:
            # Parse the log message assuming it's JSON formatted
            log_msg = self.format(record)
            log_entry = json.loads(log_msg)

            # Create timestamp
            time_str = dt.utcfromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]

            # Create the log level and message string
            level_str = f"{record.levelname:<5}"
            message_str = log_entry.get("message", "")

            # Optionally, add other fields if needed
            additional_info = " ".join(
                f"{key}: {value}"
                for key, value in log_entry.items()
                if key not in ["message", "timestamp", "level"]
            )

            # Combine all parts and print
            self.console.print(
                f"[{time_str}] {level_str} {record.process:d}: {message_str} {additional_info}"
            )

        except Exception as e:
            self.console.print(
                f"An error occurred while formatting the log message: {e}"
            )
