from loguru import logger
import sys
import os
from datetime import datetime

# Custom level between INFO (20) and WARNING (30) — always shown in terminal
RESULT_LEVEL = "RESULT"
RESULT_LEVEL_NO = 25
logger.level(RESULT_LEVEL, no=RESULT_LEVEL_NO, color="<bold>")

def _file_format(record):
    msg = record["message"]
    # Escape curly braces to prevent loguru from treating them as placeholders
    msg = msg.replace("{", "{{").replace("}", "}}")
    if len(msg) > 90:
        import textwrap
        msg = "\n    ".join(textwrap.wrap(msg, width=90))
    return "{time:HH:mm:ss} | {level:<8} | " + msg + "\n"

def setup_logging(level="INFO", log_file=None, verbose=True):
    if log_file is None:
        log_file = os.getenv("LOG_FILE_PATH", "logs/agent.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    stderr_level = level if verbose else RESULT_LEVEL

    logger.remove()
    logger.add(
        sys.stderr,
        level=stderr_level,
        format="<green>{time:HH:mm:ss}</green> | "
               "<level>{level}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
               "- <level>{message}</level>",
    )
    logger.add(
        log_file,
        level=level,
        rotation="10 MB",
        encoding="utf-8",
        format=_file_format,
    )

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 120 + "\n")
        f.write("Agent session started {}\n".format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        f.write("=" * 120 + "\n\n")

def get_logger():
    return logger
