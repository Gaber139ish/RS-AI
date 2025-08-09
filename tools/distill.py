import copy
import numpy as np


class EMATeacher:
    def __init__(self, model, decay: float = 0.99):
        self.decay = float(decay)
        self.teacher = copy.deepcopy(model)

    def update(self, student) -> None:
        # Update teacher params as EMA of student
        if hasattr(self.teacher, 'W') and hasattr(student, 'W'):
            self.teacher.W = self.decay * self.teacher.W + (1.0 - self.decay) * student.W
        if hasattr(self.teacher, 'b') and hasattr(student, 'b'):
            self.teacher.b = self.decay * self.teacher.b + (1.0 - self.decay) * student.b

    def target(self, inputs: np.ndarray) -> np.ndarray:
        if hasattr(self.teacher, 'process'):
            return self.teacher.process(inputs)
        return inputs