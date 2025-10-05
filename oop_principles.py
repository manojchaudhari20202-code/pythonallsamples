"""
Demonstration of Object-Oriented Programming Principles in Python
1. Encapsulation
2. Inheritance
3. Polymorphism
4. Abstraction
"""

from abc import ABC, abstractmethod

# Encapsulation Example
class BankAccount:
    def __init__(self, account_number, balance):
        self.__account_number = account_number  # Private attribute
        self.__balance = balance  # Private attribute
        
    def get_balance(self):
        return self.__balance
    
    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
            return True
        return False
    
    def withdraw(self, amount):
        if 0 < amount <= self.__balance:
            self.__balance -= amount
            return True
        return False

# Abstraction Example
class Shape(ABC):
    @abstractmethod
    def area(self):
        pass
    
    @abstractmethod
    def perimeter(self):
        pass

# Inheritance Example
class Rectangle(Shape):
    def __init__(self, length, width):
        self.length = length
        self.width = width
    
    def area(self):
        return self.length * self.width
    
    def perimeter(self):
        return 2 * (self.length + self.width)

class Square(Rectangle):
    def __init__(self, side):
        super().__init__(side, side)

# Polymorphism Example
class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
        self.pi = 3.14159
    
    def area(self):
        return self.pi * self.radius ** 2
    
    def perimeter(self):
        return 2 * self.pi * self.radius

# Multiple Inheritance Example
class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        pass

class Flyable:
    def fly(self):
        return "I can fly!"

class Bird(Animal, Flyable):
    def speak(self):
        return "Tweet!"

# Example Usage
if __name__ == "__main__":
    # Encapsulation
    print("=== Encapsulation Example ===")
    account = BankAccount("123456", 1000)
    print(f"Initial balance: ${account.get_balance()}")
    account.deposit(500)
    print(f"After deposit: ${account.get_balance()}")
    account.withdraw(200)
    print(f"After withdrawal: ${account.get_balance()}")
    
    # Inheritance and Polymorphism
    print("\n=== Inheritance and Polymorphism Example ===")
    shapes = [Rectangle(5, 3), Square(4), Circle(3)]
    
    for shape in shapes:
        print(f"{type(shape).__name__}:")
        print(f"Area: {shape.area()}")
        print(f"Perimeter: {shape.perimeter()}\n")
    
    # Multiple Inheritance
    print("=== Multiple Inheritance Example ===")
    parrot = Bird("Rio")
    print(f"{parrot.name} says: {parrot.speak()}")
    print(f"{parrot.name}: {parrot.fly()}")