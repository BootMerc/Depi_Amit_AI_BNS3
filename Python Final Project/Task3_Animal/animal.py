from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def make_sound(self):
        """Abstract method that must be implemented by subclasses"""
        pass
    def describe(self):
        print("This is an animal.")
