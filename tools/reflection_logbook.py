"""
GPT-style logbook for recording reflections and experiences.
"""
import datetime
import logging

logger = logging.getLogger(__name__)

class ReflectionLogbook:
    def __init__(self, file_path):
        self.file_path = file_path

    def record(self, entry):
        """Append a markdown entry with timestamp"""
        ts = datetime.datetime.now().isoformat()
        try:
            with open(self.file_path, 'a') as f:
                f.write(f"## [{ts}]\n{entry}\n\n")
        except Exception as e:
            logger.warning(f"Failed logging reflection: {e}")

    def record_introspection(self, report):
        text = """Introspection Report:\n"""
        for module, loss in report.items():
            loss_str = f"loss={loss}" if loss is not None else "no data"
            text += f"- **{module}**: {loss_str}\n"
        self.record(text)
