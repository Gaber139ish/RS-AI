from typing import Dict
import threading


class Wallet:
    def __init__(self, initial_balance: float = 0.0):
        self._balance = float(initial_balance)
        self._staked = 0.0
        self._lock = threading.Lock()

    def balance(self) -> float:
        with self._lock:
            return float(self._balance)

    def stake(self, amount: float) -> float:
        with self._lock:
            amount = float(amount)
            amount = min(amount, self._balance)
            self._balance -= amount
            self._staked += amount
            return self._staked

    def unstake(self, amount: float) -> float:
        with self._lock:
            amount = float(amount)
            amount = min(amount, self._staked)
            self._staked -= amount
            self._balance += amount
            return self._staked

    def staked(self) -> float:
        with self._lock:
            return float(self._staked)

    def deposit(self, amount: float) -> float:
        with self._lock:
            self._balance += float(amount)
            return self._balance

    def transfer_to(self, other: 'Wallet', amount: float) -> bool:
        with self._lock:
            if amount > self._balance:
                return False
            self._balance -= amount
        other.deposit(amount)
        return True