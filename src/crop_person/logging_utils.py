import structlog
import logging


def get_logger(name: str = "crop_person"):
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,  # <-- Set your desired level here!
    )

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),  # or JSONRenderer in prod
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.INFO
        ),  # <-- match your level here!
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    return structlog.get_logger(name)
