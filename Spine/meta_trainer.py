import numpy as np
from collections import deque

class MetaTrainer:
    def __init__(self, inner_lr=0.001, outer_lr=0.01, evolution_schedule=True, memory_size=100):
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.evolution_schedule = evolution_schedule
        self.memory = deque(maxlen=memory_size)
        self.performance_log = []

    def meta_train(self, model, data, labels, epoch):
        """
        Perform one step of meta-training. Adjust internal hyperparameters if evolution is enabled.
        """
        predictions = model.forward(data)
        loss = self._compute_loss(predictions, labels)

        if self.evolution_schedule and epoch % 5 == 0:
            self._adjust_learning_rates(epoch)

        gradient = model.backward(data, labels)
        model.update_weights(gradient, self.inner_lr)

        self.memory.append((loss, data, labels))
        self.performance_log.append((epoch, loss))

        return loss

    def _compute_loss(self, predictions, labels):
        return np.mean((predictions - labels) ** 2)

    def _adjust_learning_rates(self, epoch):
        """
        Adjust learning rates over time (exponential decay for outer, cosine schedule for inner).
        """
        decay = 0.95 ** (epoch // 5)
        self.outer_lr *= decay
        self.inner_lr = 0.001 * (1 + np.cos(epoch / 50 * np.pi)) / 2

    def curiosity_reward(self, loss, previous_loss, threshold=0.01):
        """
        Return a curiosity reward based on how much the loss changes.
        """
        delta = abs(previous_loss - loss)
        if delta > threshold:
            return min(delta, 1.0)
        return 0.0

    def introspect(self):
        if not self.performance_log:
            return {}

        avg_loss = np.mean([loss for _, loss in self.performance_log[-10:]])
        trend = "improving" if self.performance_log[-1][1] < self.performance_log[0][1] else "declining"
        return {
            "average_recent_loss": avg_loss,
            "performance_trend": trend,
            "outer_lr": self.outer_lr,
            "inner_lr": self.inner_lr
        }

def build_meta_trainer(config):
    inner = config.get("inner_lr", 0.001)
    outer = config.get("outer_lr", 0.01)
    evolve = config.get("evolution_schedule", True)
    memory = config.get("memory_size", 100)
    return MetaTrainer(inner_lr=inner, outer_lr=outer, evolution_schedule=evolve, memory_size=memory)

def create(config=None):
    if config is None:
        config = {}
    return build_meta_trainer(config)
