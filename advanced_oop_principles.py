"""
🚀 Advanced Python Object-Oriented Programming and Design Principles
==================================================================
This file demonstrates comprehensive implementation of:
1. OOP Principles
2. Software Design Principles
3. Design Patterns
4. Core Python Coding Practices

Author: GitHub Copilot
Date: October 5, 2025
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import logging
import threading
from datetime import datetime
from decimal import Decimal
import uuid
import queue
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from typing import Callable, Tuple
from collections import OrderedDict
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================
# 1. OOP PRINCIPLES
# =============================================

# Interface (Abstract Base Class)
class PaymentProcessor(ABC):
    """Abstract base class demonstrating interface definition."""
    
    @abstractmethod
    def process_payment(self, amount: Decimal) -> bool:
        """Process a payment transaction."""
        pass
    
    @abstractmethod
    def refund_payment(self, transaction_id: str) -> bool:
        """Refund a processed payment."""
        pass

# Encapsulation Example
class BankAccount:
    """Demonstrates encapsulation with private attributes and public methods."""
    
    def __init__(self, account_number: str, initial_balance: Decimal):
        self.__account_number = account_number
        self.__balance = initial_balance
        self.__transaction_history: List[str] = []
    
    def deposit(self, amount: Decimal) -> bool:
        """Public method to deposit money."""
        if amount > 0:
            self.__balance += amount
            self.__log_transaction("deposit", amount)
            return True
        return False
    
    def withdraw(self, amount: Decimal) -> bool:
        """Public method to withdraw money."""
        if 0 < amount <= self.__balance:
            self.__balance -= amount
            self.__log_transaction("withdrawal", amount)
            return True
        return False
    
    def get_balance(self) -> Decimal:
        """Public method to check balance."""
        return self.__balance
    
    def __log_transaction(self, type_: str, amount: Decimal) -> None:
        """Private method to log transactions."""
        timestamp = datetime.now().isoformat()
        self.__transaction_history.append(f"{timestamp} - {type_}: {amount}")

# Inheritance and Polymorphism Example
class Vehicle(ABC):
    """Base class demonstrating inheritance and polymorphism."""
    
    def __init__(self, make: str, model: str, year: int):
        self.make = make
        self.model = model
        self.year = year
    
    @abstractmethod
    def calculate_rental_price(self) -> Decimal:
        """Calculate daily rental price."""
        pass
    
    def get_info(self) -> str:
        """Get vehicle information."""
        return f"{self.year} {self.make} {self.model}"

class Car(Vehicle):
    """Car class demonstrating inheritance."""
    
    def __init__(self, make: str, model: str, year: int, seats: int):
        super().__init__(make, model, year)
        self.seats = seats
    
    def calculate_rental_price(self) -> Decimal:
        """Implementation of abstract method."""
        base_price = Decimal('50.00')
        age_factor = Decimal(str(max(1, (2025 - self.year) * 0.1)))
        return base_price - (base_price * age_factor)

# Association Example
class Driver:
    """Demonstrates association relationship."""
    
    def __init__(self, name: str, license_number: str):
        self.name = name
        self.license_number = license_number
        self.current_vehicle: Optional[Vehicle] = None
    
    def assign_vehicle(self, vehicle: Vehicle) -> None:
        """Assign a vehicle to driver."""
        self.current_vehicle = vehicle
        logger.info(f"Vehicle {vehicle.get_info()} assigned to driver {self.name}")

# Composition Example
class Engine:
    """Component class for composition."""
    
    def __init__(self, horsepower: int, fuel_type: str):
        self.horsepower = horsepower
        self.fuel_type = fuel_type
        self.__running = False
    
    def start(self) -> None:
        """Start the engine."""
        self.__running = True
        logger.info("Engine started")
    
    def stop(self) -> None:
        """Stop the engine."""
        self.__running = False
        logger.info("Engine stopped")

class ElectricCar(Car):
    """Demonstrates composition - ElectricCar has-a Engine."""
    
    def __init__(self, make: str, model: str, year: int, seats: int,
                 battery_capacity: int):
        super().__init__(make, model, year, seats)
        self.engine = Engine(horsepower=300, fuel_type="electric")
        self.battery_capacity = battery_capacity
    
    def calculate_rental_price(self) -> Decimal:
        """Override with electric car specific logic."""
        base_price = super().calculate_rental_price()
        # Electric cars get a green discount
        green_discount = Decimal('0.15')
        return base_price * (1 - green_discount)

# Aggregation Example
class Fleet:
    """Demonstrates aggregation - Fleet has vehicles but vehicles can exist independently."""
    
    def __init__(self, name: str):
        self.name = name
        self.vehicles: List[Vehicle] = []
    
    def add_vehicle(self, vehicle: Vehicle) -> None:
        """Add a vehicle to the fleet."""
        self.vehicles.append(vehicle)
        logger.info(f"Added {vehicle.get_info()} to fleet {self.name}")
    
    def remove_vehicle(self, vehicle: Vehicle) -> None:
        """Remove a vehicle from the fleet."""
        if vehicle in self.vehicles:
            self.vehicles.remove(vehicle)
            logger.info(f"Removed {vehicle.get_info()} from fleet {self.name}")

# =============================================
# 2. SOFTWARE DESIGN PRINCIPLES
# =============================================

# Single Responsibility Principle
class PaymentCalculator:
    """Class with single responsibility of calculating payments."""
    
    @staticmethod
    def calculate_total_with_tax(amount: Decimal, tax_rate: Decimal) -> Decimal:
        """Calculate total amount including tax."""
        return amount * (1 + tax_rate)

# Open/Closed Principle
class PaymentValidator(ABC):
    """Abstract class for payment validation - open for extension, closed for modification."""
    
    @abstractmethod
    def validate(self, payment: Dict[str, Any]) -> bool:
        """Validate a payment."""
        pass

class CreditCardValidator(PaymentValidator):
    """Concrete implementation of payment validator."""
    
    def validate(self, payment: Dict[str, Any]) -> bool:
        """Validate credit card payment."""
        # Implementation of credit card validation logic
        return True

# Liskov Substitution Principle
class Bird(ABC):
    """Abstract base class for birds."""
    
    @abstractmethod
    def eat(self) -> None:
        """All birds can eat."""
        pass

class FlyingBird(Bird):
    """Abstract class for flying birds."""
    
    @abstractmethod
    def fly(self) -> None:
        """Flying birds can fly."""
        pass

class Sparrow(FlyingBird):
    """Concrete implementation of a flying bird."""
    
    def eat(self) -> None:
        """Implementation of eat method."""
        logger.info("Sparrow is eating")
    
    def fly(self) -> None:
        """Implementation of fly method."""
        logger.info("Sparrow is flying")

class Penguin(Bird):
    """Concrete implementation of a non-flying bird."""
    
    def eat(self) -> None:
        """Implementation of eat method."""
        logger.info("Penguin is eating")

# Interface Segregation Principle
class Worker(ABC):
    """Abstract base class demonstrating interface segregation."""
    
    @abstractmethod
    def work(self) -> None:
        """All workers can work."""
        pass

class Eater(ABC):
    """Separate interface for eating behavior."""
    
    @abstractmethod
    def eat(self) -> None:
        """All eaters can eat."""
        pass

class Robot(Worker):
    """Robot only implements Worker interface."""
    
    def work(self) -> None:
        """Implementation of work method."""
        logger.info("Robot is working")

class Human(Worker, Eater):
    """Human implements both Worker and Eater interfaces."""
    
    def work(self) -> None:
        """Implementation of work method."""
        logger.info("Human is working")
    
    def eat(self) -> None:
        """Implementation of eat method."""
        logger.info("Human is eating")

# =============================================
# 3. DESIGN PATTERNS
# =============================================

# Singleton Pattern
class DatabaseConnection:
    """Singleton pattern implementation."""
    
    _instance: Optional[DatabaseConnection] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> DatabaseConnection:
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.connection_string = "mongodb://localhost:27017"
    
    def connect(self) -> None:
        """Simulate database connection."""
        logger.info(f"Connecting to database: {self.connection_string}")

# Factory Pattern
class PaymentMethod(ABC):
    """Abstract base class for payment methods."""
    
    @abstractmethod
    def process(self, amount: Decimal) -> bool:
        """Process payment."""
        pass

class CreditCardPayment(PaymentMethod):
    """Concrete implementation of credit card payment."""
    
    def process(self, amount: Decimal) -> bool:
        """Process credit card payment."""
        logger.info(f"Processing credit card payment of {amount}")
        return True

class PayPalPayment(PaymentMethod):
    """Concrete implementation of PayPal payment."""
    
    def process(self, amount: Decimal) -> bool:
        """Process PayPal payment."""
        logger.info(f"Processing PayPal payment of {amount}")
        return True

class PaymentMethodFactory:
    """Factory pattern implementation."""
    
    @staticmethod
    def create_payment_method(method_type: str) -> PaymentMethod:
        """Create payment method based on type."""
        if method_type.lower() == "credit_card":
            return CreditCardPayment()
        elif method_type.lower() == "paypal":
            return PayPalPayment()
        raise ValueError(f"Unknown payment method: {method_type}")

# Observer Pattern
class Subject(ABC):
    """Abstract subject for observer pattern."""
    
    def __init__(self):
        self._observers: List[Observer] = []
    
    def attach(self, observer: Observer) -> None:
        """Attach an observer."""
        self._observers.append(observer)
    
    def detach(self, observer: Observer) -> None:
        """Detach an observer."""
        self._observers.remove(observer)
    
    def notify(self) -> None:
        """Notify all observers."""
        for observer in self._observers:
            observer.update(self)

class Observer(ABC):
    """Abstract observer."""
    
    @abstractmethod
    def update(self, subject: Subject) -> None:
        """Update method called by subject."""
        pass

class OrderStatus(Subject):
    """Concrete subject implementation."""
    
    def __init__(self):
        super().__init__()
        self._status = ""
    
    @property
    def status(self) -> str:
        """Get current status."""
        return self._status
    
    @status.setter
    def status(self, value: str) -> None:
        """Set status and notify observers."""
        self._status = value
        self.notify()

class CustomerNotifier(Observer):
    """Concrete observer implementation."""
    
    def __init__(self, customer_email: str):
        self.customer_email = customer_email
        # test hook: store last notification message for unit tests
        self.last_notification: Optional[str] = None
    
    def update(self, subject: Subject) -> None:
        """Send notification to customer."""
        if isinstance(subject, OrderStatus):
            message = (
                f"Sending email to {self.customer_email}: "
                f"Order status changed to {subject.status}"
            )
            logger.info(message)
            # store last notification for tests / debugging
            self.last_notification = message

# ---------------------------------------------
# Structural Patterns: Adapter, Composite, Decorator, Proxy
# ---------------------------------------------

# Adapter: adapt PaymentMethod to PaymentProcessor interface
class PaymentProcessorAdapter(PaymentProcessor):
    """Adapter that wraps a PaymentMethod and exposes PaymentProcessor API."""

    def __init__(self, payment_method: PaymentMethod):
        self._payment_method = payment_method

    def process_payment(self, amount: Decimal) -> bool:
        return self._payment_method.process(amount)

    def refund_payment(self, transaction_id: str) -> bool:
        # Simple simulated refund behavior
        logger.info(f"Refunding {transaction_id} via adapter")
        return True

# Composite: composite pattern for menu of services
class ServiceComponent(ABC):
    @abstractmethod
    def render(self) -> str:
        pass

class ServiceLeaf(ServiceComponent):
    def __init__(self, name: str):
        self.name = name

    def render(self) -> str:
        return f"Service: {self.name}\n"

class ServiceComposite(ServiceComponent):
    def __init__(self, name: str):
        self.name = name
        self.children: List[ServiceComponent] = []

    def add(self, component: ServiceComponent) -> None:
        self.children.append(component)

    def render(self) -> str:
        output = f"{self.name}:\n"
        for c in self.children:
            output += "  " + c.render()
        return output

# Decorator: add logging around payment processing
def logging_decorator(func):
    def wrapper(*args, **kwargs):
        logger.info(f"Calling {func.__name__}")
        result = func(*args, **kwargs)
        logger.info(f"Finished {func.__name__}")
        return result
    return wrapper

class PaymentService:
    @logging_decorator
    def make_payment(self, processor: PaymentProcessor, amount: Decimal) -> bool:
        return processor.process_payment(amount)

# Proxy: simple access control proxy for DatabaseConnection
class DatabaseConnectionProxy:
    def __init__(self, user_role: str):
        self._db = DatabaseConnection()
        self._role = user_role

    def connect(self) -> None:
        if self._role != 'admin':
            logger.warning("Only admin can open direct DB connections")
            return
        self._db.connect()

# ---------------------------------------------
# Behavioral Patterns: Strategy, Command, State
# ---------------------------------------------

# Strategy: different tax calculation strategies
class TaxStrategy(ABC):
    @abstractmethod
    def calculate(self, amount: Decimal) -> Decimal:
        pass

class NoTax(TaxStrategy):
    def calculate(self, amount: Decimal) -> Decimal:
        return Decimal('0')

class StandardTax(TaxStrategy):
    def __init__(self, rate: Decimal):
        self.rate = rate

    def calculate(self, amount: Decimal) -> Decimal:
        return amount * self.rate

class Checkout:
    def __init__(self, tax_strategy: TaxStrategy):
        self.tax_strategy = tax_strategy

    def total(self, amount: Decimal) -> Decimal:
        tax = self.tax_strategy.calculate(amount)
        return amount + tax

# Command: encapsulate actions
class Command(ABC):
    @abstractmethod
    def execute(self) -> None:
        pass

class StartEngineCommand(Command):
    def __init__(self, engine: Engine):
        self.engine = engine

    def execute(self) -> None:
        self.engine.start()

class StopEngineCommand(Command):
    def __init__(self, engine: Engine):
        self.engine = engine

    def execute(self) -> None:
        self.engine.stop()

# State: simple order state machine
class OrderState(ABC):
    @abstractmethod
    def next(self, order: 'SimpleOrder') -> None:
        pass

class NewState(OrderState):
    def next(self, order: 'SimpleOrder') -> None:
        order.state = ProcessingState()

class ProcessingState(OrderState):
    def next(self, order: 'SimpleOrder') -> None:
        order.state = ShippedState()

class ShippedState(OrderState):
    def next(self, order: 'SimpleOrder') -> None:
        order.state = DeliveredState()

class DeliveredState(OrderState):
    def next(self, order: 'SimpleOrder') -> None:
        logger.info('Order already delivered')

class SimpleOrder:
    def __init__(self):
        self.state: OrderState = NewState()

    def advance(self) -> None:
        self.state.next(self)

# ---------------------------------------------
# Creational: Builder
# ---------------------------------------------

class CarBuilder:
    def __init__(self):
        self._make = ''
        self._model = ''
        self._year = 0

    def set_make(self, make: str) -> 'CarBuilder':
        self._make = make
        return self

    def set_model(self, model: str) -> 'CarBuilder':
        self._model = model
        return self

    def set_year(self, year: int) -> 'CarBuilder':
        self._year = year
        return self

    def build(self) -> Car:
        return Car(self._make, self._model, self._year, seats=4)

# ---------------------------------------------
# Concurrency: Producer-Consumer using queue and ThreadPoolExecutor
# ---------------------------------------------

def producer(q: queue.Queue, items: List[int]) -> None:
    for item in items:
        logger.info(f"Producing {item}")
        q.put(item)
    q.put(None)  # sentinel

def consumer(q: queue.Queue) -> List[int]:
    consumed = []
    while True:
        item = q.get()
        if item is None:
            break
        logger.info(f"Consuming {item}")
        consumed.append(item)
    return consumed

# ---------------------------------------------
# Functional: currying and function composition
# ---------------------------------------------

def curry_add(a: int):
    def inner(b: int) -> int:
        return a + b
    return inner

def compose(f, g):
    def composed(x):
        return f(g(x))
    return composed

# ---------------------------------------------
# Integration: DTO, Repository, Service Locator
# ---------------------------------------------

@dataclass
class OrderDTO:
    id: str
    amount: Decimal

class OrderRepository:
    def __init__(self):
        self._store: Dict[str, OrderDTO] = {}

    def save(self, dto: OrderDTO) -> None:
        self._store[dto.id] = dto

    def get(self, id_: str) -> Optional[OrderDTO]:
        return self._store.get(id_)

class ServiceLocator:
    _services: Dict[str, Any] = {}

    @classmethod
    def register(cls, name: str, instance: Any) -> None:
        cls._services[name] = instance

    @classmethod
    def resolve(cls, name: str) -> Any:
        return cls._services.get(name)

# ---------------------------------------------
# Small MVC Demonstration (very lightweight)
# ---------------------------------------------

class Model:
    def __init__(self):
        self.data: Dict[str, Any] = {}

class View:
    def render(self, data: Dict[str, Any]) -> str:
        return f"View: {data}"

class Controller:
    def __init__(self, model: Model, view: View):
        self.model = model
        self.view = view

    def update_model(self, key: str, value: Any) -> str:
        self.model.data[key] = value
        return self.view.render(self.model.data)


# =============================================
# Second batch of pattern implementations
# - Flyweight
# - Bridge
# - Visitor
# - Interpreter
# - DAO (Data Access Object)
# - Unit of Work
# - Retry decorator
# - Circuit Breaker (simple)
# - Mediator
# =============================================


# Flyweight: share engine specs to reduce memory
class EngineSpec:
    def __init__(self, horsepower: int, fuel_type: str):
        self.horsepower = horsepower
        self.fuel_type = fuel_type

    def __repr__(self):
        return f"EngineSpec(hp={self.horsepower}, fuel={self.fuel_type})"

class EngineFlyweightFactory:
    _cache: Dict[Tuple[int, str], EngineSpec] = {}

    @classmethod
    def get_spec(cls, horsepower: int, fuel_type: str) -> EngineSpec:
        key = (horsepower, fuel_type)
        if key not in cls._cache:
            cls._cache[key] = EngineSpec(horsepower, fuel_type)
        return cls._cache[key]


# Bridge: separate reporting abstraction from rendering implementation
class Renderer(ABC):
    @abstractmethod
    def render(self, title: str, content: str) -> str:
        pass

class ConsoleRenderer(Renderer):
    def render(self, title: str, content: str) -> str:
        return f"=== {title} ===\n{content}\n"

class JSONRenderer(Renderer):
    def render(self, title: str, content: str) -> str:
        import json
        return json.dumps({"title": title, "content": content})

class ReportAbstraction:
    def __init__(self, renderer: Renderer):
        self.renderer = renderer

    def produce(self, data: Dict[str, Any]) -> str:
        title = data.get('title', 'Report')
        content = data.get('content', '')
        return self.renderer.render(title, content)


# Visitor pattern: allow operations across vehicle objects
class VehicleVisitor(ABC):
    @abstractmethod
    def visit(self, vehicle: Vehicle) -> Any:
        pass

class OdometerVisitor(VehicleVisitor):
    def visit(self, vehicle: Vehicle) -> str:
        # For demo, return a description
        return f"Visited {vehicle.get_info()}"

# Add accept method to Vehicle classes via mixin-style addition
def vehicle_accept(self, visitor: VehicleVisitor) -> Any:
    return visitor.visit(self)

# Monkey-patch accept onto Vehicle and subclasses (simple educational approach)
Vehicle.accept = vehicle_accept
Car.accept = vehicle_accept
ElectricCar.accept = vehicle_accept


# Interpreter: very small expression language: 'add 1 2', 'sub 5 3'
class Expression(ABC):
    @abstractmethod
    def interpret(self) -> int:
        pass

class Number(Expression):
    def __init__(self, value: int):
        self.value = value

    def interpret(self) -> int:
        return self.value

class AddExpression(Expression):
    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def interpret(self) -> int:
        return self.left.interpret() + self.right.interpret()

def parse_expression(text: str) -> Expression:
    parts = text.split()
    if parts[0] == 'add' and len(parts) == 3:
        return AddExpression(Number(int(parts[1])), Number(int(parts[2])))
    raise ValueError('Unsupported expression')


# DAO: Data Access Object wrapping repository operations
class OrderDAO:
    def __init__(self, repository: OrderRepository):
        self.repo = repository

    def create_order(self, amount: Decimal) -> OrderDTO:
        dto = OrderDTO(id=uuid.uuid4().hex, amount=amount)
        self.repo.save(dto)
        return dto

    def find(self, id_: str) -> Optional[OrderDTO]:
        return self.repo.get(id_)


# Unit of Work: track new objects and commit to repositories in one go
class UnitOfWork:
    def __init__(self, order_repo: OrderRepository):
        self._order_repo = order_repo
        self.new_orders: List[OrderDTO] = []

    def register_new_order(self, dto: OrderDTO) -> None:
        self.new_orders.append(dto)

    def commit(self) -> None:
        for o in self.new_orders:
            self._order_repo.save(o)
        self.new_orders.clear()


# Retry decorator: retry transient failures
def retry(times: int = 3, exceptions: Tuple[type, ...] = (Exception,)):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for _ in range(times):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
            raise last_exc
        return wrapper
    return decorator


# Circuit Breaker: simple implementation
class CircuitBreaker:
    def __init__(self, max_failures: int = 3):
        self.failures = 0
        self.max_failures = max_failures
        self.open = False

    def call(self, func: Callable, *args, **kwargs):
        if self.open:
            raise RuntimeError('Circuit is open')
        try:
            result = func(*args, **kwargs)
            self.failures = 0
            return result
        except Exception:
            self.failures += 1
            if self.failures >= self.max_failures:
                self.open = True
            raise


# Mediator: coordinate simple interactions
class Mediator:
    def __init__(self):
        self.participants: Dict[str, Any] = {}

    def register(self, name: str, participant: Any) -> None:
        self.participants[name] = participant

    def notify(self, sender: str, event: str, data: Any = None) -> None:
        logger.info(f"Mediator: {sender} -> event {event}")
        # simple broadcast to others
        for name, p in self.participants.items():
            if name != sender and hasattr(p, 'receive'):
                p.receive(event, data)

class MediatedDriver(Driver):
    def __init__(self, name: str, license_number: str, mediator: Mediator):
        super().__init__(name, license_number)
        self.mediator = mediator
        mediator.register(name, self)

    def receive(self, event: str, data: Any) -> None:
        logger.info(f"Driver {self.name} received {event} with {data}")


# =============================================
# Third batch of patterns
# - Event Sourcing (EventStore)
# - Saga Coordinator (simple)
# - Data Mapper
# - CQRS (CommandBus / QueryBus)
# - Publish-Subscribe EventBus
# - Health Check Registry
# - Object Pool (simple)
# - Feature Toggle
# =============================================


# Event Sourcing: in-memory event store and simple replay
class Event:
    def __init__(self, aggregate_id: str, type_: str, payload: Dict[str, Any]):
        self.aggregate_id = aggregate_id
        self.type = type_
        self.payload = payload
        self.timestamp = datetime.now()


class EventStore:
    def __init__(self):
        self._events: List[Event] = []

    def append(self, event: Event) -> None:
        self._events.append(event)

    def get_events(self, aggregate_id: str) -> List[Event]:
        return [e for e in self._events if e.aggregate_id == aggregate_id]

    def replay(self, aggregate_id: str) -> Dict[str, Any]:
        # simplistic replay building a dict state
        state: Dict[str, Any] = {}
        for e in self.get_events(aggregate_id):
            state[e.type] = e.payload
        return state


# Saga: naive coordinator that reacts to events and dispatches commands
class SagaCoordinator:
    def __init__(self, event_store: EventStore):
        self.event_store = event_store
        self.handlers: Dict[str, Callable[[Event], None]] = {}

    def register(self, event_type: str, handler: Callable[[Event], None]) -> None:
        self.handlers[event_type] = handler

    def handle(self, event: Event) -> None:
        self.event_store.append(event)
        h = self.handlers.get(event.type)
        if h:
            h(event)


# Data Mapper: map between DTO and domain object
class OrderDomain:
    def __init__(self, id_: str, amount: Decimal):
        self.id = id_
        self.amount = amount


class OrderDataMapper:
    @staticmethod
    def to_dto(domain: OrderDomain) -> OrderDTO:
        return OrderDTO(id=domain.id, amount=domain.amount)

    @staticmethod
    def to_domain(dto: OrderDTO) -> OrderDomain:
        return OrderDomain(id_=dto.id, amount=dto.amount)


# CQRS: Command and Query buses
class CommandBus:
    def __init__(self):
        self._handlers: Dict[str, Callable[[Any], Any]] = {}

    def register(self, name: str, handler: Callable[[Any], Any]) -> None:
        self._handlers[name] = handler

    def handle(self, name: str, payload: Any) -> Any:
        h = self._handlers.get(name)
        if not h:
            raise ValueError('Handler not found')
        return h(payload)


class QueryBus:
    def __init__(self):
        self._handlers: Dict[str, Callable[[Any], Any]] = {}

    def register(self, name: str, handler: Callable[[Any], Any]) -> None:
        self._handlers[name] = handler

    def handle(self, name: str, payload: Any) -> Any:
        h = self._handlers.get(name)
        if not h:
            raise ValueError('Handler not found')
        return h(payload)


# Publish-Subscribe Event Bus
class EventBus:
    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[Event], None]]] = {}

    def subscribe(self, event_type: str, handler: Callable[[Event], None]) -> None:
        self._subscribers.setdefault(event_type, []).append(handler)

    def publish(self, event: Event) -> None:
        for handler in self._subscribers.get(event.type, []):
            handler(event)


# Health Check Registry
class HealthCheckRegistry:
    def __init__(self):
        self._checks: Dict[str, Callable[[], bool]] = {}

    def register(self, name: str, check: Callable[[], bool]) -> None:
        self._checks[name] = check

    def run_all(self) -> Dict[str, bool]:
        return {name: check() for name, check in self._checks.items()}


# Object Pool: simplistic pool for DatabaseConnection
class ObjectPool:
    def __init__(self, factory: Callable[[], Any], max_size: int = 5):
        self._factory = factory
        self._pool: List[Any] = []
        self._max_size = max_size

    def acquire(self) -> Any:
        if self._pool:
            return self._pool.pop()
        return self._factory()

    def release(self, obj: Any) -> None:
        if len(self._pool) < self._max_size:
            self._pool.append(obj)


# Feature Toggle: enable/disable features at runtime
class FeatureToggle:
    def __init__(self):
        self._flags: Dict[str, bool] = {}

    def enable(self, name: str) -> None:
        self._flags[name] = True

    def disable(self, name: str) -> None:
        self._flags[name] = False

    def is_enabled(self, name: str) -> bool:
        return self._flags.get(name, False)


# =============================================
# Fourth batch of patterns and utilities
# - LRU Cache (caching)
# - Lazy / Virtual Proxy (lazy instantiation)
# - Backpressure Queue (bounded producer)
# - Double Dispatch example
# - Dynamic Proxy (method call logging)
# - Event Aggregator (grouped pub-sub)
# - Throttling decorator
# =============================================


class LRUCache:
    """Simple LRU cache using OrderedDict."""

    def __init__(self, capacity: int = 128):
        self.capacity = capacity
        self.cache: OrderedDict = OrderedDict()

    def get(self, key: Any) -> Any:
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: Any, value: Any) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


class LazyProxy:
    """Virtual proxy that delays object creation until first attribute access."""

    def __init__(self, factory: Callable[[], Any]):
        self._factory = factory
        self._obj: Optional[Any] = None

    def _ensure(self):
        if self._obj is None:
            self._obj = self._factory()

    def __getattr__(self, item):
        self._ensure()
        return getattr(self._obj, item)


class BackpressureQueue:
    """Wrap queue.Queue to provide a non-blocking put that signals backpressure."""

    def __init__(self, maxsize: int = 10):
        self._q = queue.Queue(maxsize=maxsize)

    def try_put(self, item: Any) -> bool:
        try:
            self._q.put_nowait(item)
            return True
        except queue.Full:
            return False

    def get(self, timeout: Optional[float] = None) -> Any:
        return self._q.get(timeout=timeout)


# Double Dispatch example for vehicle interactions
def double_dispatch_interact(a: Vehicle, b: Vehicle) -> str:
    # call a.interact(b), which will call a.interact_with_<ClassName> if present
    return a.interact(b) if hasattr(a, 'interact') else f"No interaction {a} and {b}"

def vehicle_interact(self, other: Vehicle) -> str:
    method_name = f"interact_with_{other.__class__.__name__}"
    method = getattr(self, method_name, None)
    if method:
        return method(other)
    return f"{self.get_info()} interacts with {other.get_info()} generically"

def car_interact_with_ElectricCar(self, other: Vehicle) -> str:
    return f"{self.get_info()} hums alongside electric {other.get_info()}"

# Attach double dispatch helpers
Vehicle.interact = vehicle_interact
Car.interact_with_ElectricCar = car_interact_with_ElectricCar


# Dynamic Proxy: wrap an object and log calls dynamically
class DynamicProxy:
    def __init__(self, target: Any):
        self._target = target

    def __getattr__(self, item):
        attr = getattr(self._target, item)
        if callable(attr):
            def wrapper(*args, **kwargs):
                logger.info(f"DynamicProxy: calling {item}")
                return attr(*args, **kwargs)
            return wrapper
        return attr


# Event Aggregator: batch events and dispatch to subscribers
class EventAggregator:
    def __init__(self):
        self._subs: Dict[str, List[Callable[[Event], None]]] = {}
        self._buffer: List[Event] = []

    def subscribe(self, event_type: str, handler: Callable[[Event], None]) -> None:
        self._subs.setdefault(event_type, []).append(handler)

    def publish(self, event: Event) -> None:
        self._buffer.append(event)

    def dispatch(self) -> None:
        while self._buffer:
            e = self._buffer.pop(0)
            for h in self._subs.get(e.type, []):
                h(e)


# Throttling decorator: allow N calls per sec
def throttle(calls_per_sec: float):
    interval = 1.0 / calls_per_sec
    def decorator(func):
        last_call = {'time': 0.0}
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            elapsed = now - last_call['time']
            if elapsed < interval:
                # simple throttle: sleep remaining time
                time.sleep(interval - elapsed)
            last_call['time'] = time.time()
            return func(*args, **kwargs)
        return wrapper
    return decorator





# =============================================
# 4. EXAMPLE USAGE
# =============================================

def main() -> None:
    """Main function demonstrating usage of all implemented patterns and principles."""
    try:
        # Create vehicles
        tesla = ElectricCar("Tesla", "Model 3", 2023, 5, 75)
        toyota = Car("Toyota", "Camry", 2022, 5)
        
        # Create and use fleet (Aggregation)
        fleet = Fleet("Main Fleet")
        fleet.add_vehicle(tesla)
        fleet.add_vehicle(toyota)
        
        # Create driver and assign vehicle (Association)
        driver = Driver("John Doe", "DL123456")
        driver.assign_vehicle(tesla)
        
        # Demonstrate polymorphism
        vehicles = [tesla, toyota]
        for vehicle in vehicles:
            price = vehicle.calculate_rental_price()
            logger.info(f"Rental price for {vehicle.get_info()}: ${price:.2f}")
        
        # Demonstrate Singleton
        db1 = DatabaseConnection()
        db2 = DatabaseConnection()
        assert id(db1) == id(db2), "Singleton pattern failed"
        
        # Demonstrate Factory Pattern
        payment_factory = PaymentMethodFactory()
        credit_card = payment_factory.create_payment_method("credit_card")
        paypal = payment_factory.create_payment_method("paypal")
        
        # Demonstrate Observer Pattern
        order_status = OrderStatus()
        customer_notifier = CustomerNotifier("customer@example.com")
        order_status.attach(customer_notifier)
        order_status.status = "Processing"
        order_status.status = "Shipped"
        
        logger.info("All demonstrations completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()


# =============================================
# ADDITIONAL: DESIGN PRINCIPLES EXAMPLES
# Short, focused examples for many common software design principles.
# These are intentionally tiny — each demonstrates the core idea.
# =============================================

def design_principles_examples() -> Dict[str, str]:
    """Return short textual/demo evidence for each design principle.

    Each entry is a one-line demonstration or hint + (where applicable)
    a tiny code example executed or described.
    """
    examples: Dict[str, str] = {}

    # KISS: Keep It Simple, Stupid — prefer simple direct solutions
    examples['KISS'] = (
        "Prefer a simple function over complex abstractions - e.g. use a single-"
        "responsibility helper instead of a 10-class hierarchy for a small task."
    )

    # YAGNI: You Aren't Gonna Need It — avoid adding premature features
    examples['YAGNI'] = (
        "Don't implement optional caching or plugins until a real need appears."
    )

    # Do The Simplest Thing That Could Possibly Work
    examples['SimpleThing'] = (
        "Start with a straightforward implementation and iterate when needed."
    )

    # Separation of Concerns
    examples['SeparationOfConcerns'] = (
        "Split logic: data access, business rules, and presentation are separate."
    )

    # DRY
    def _format_currency(amount: Decimal) -> str:
        return f"${amount:.2f}"

    examples['DRY'] = (
        "Use shared helpers to avoid duplication. Example helper: _format_currency"
    )

    # Code For The Maintainer
    examples['CodeForMaintainer'] = (
        "Write clear names, docstrings, and small functions — future-you will thank you."
    )

    # Avoid Premature Optimization
    examples['AvoidPrematureOptimization'] = (
        "Measure first; optimize hot paths later. Keep readable code initially."
    )

    # Minimise Coupling
    examples['MinimiseCoupling'] = (
        "Depend on interfaces/ABCs rather than concrete classes (e.g., PaymentProcessor)."
    )

    # Law of Demeter (only talk to immediate friends)
    examples['LawOfDemeter'] = (
        "Prefer driver.assign_vehicle(vehicle) over driver.vehicle.engine.start()")

    # Composition Over Inheritance
    examples['CompositionOverInheritance'] = (
        "ElectricCar has an Engine (composition) rather than inheriting engine behavior."
    )

    # Orthogonality
    examples['Orthogonality'] = (
        "Keep components independent so changes in one area don't ripple."
    )

    # Robustness Principle (Be conservative in what you send...) - Postel's
    examples['RobustnessPrinciple'] = (
        "A function should accept slightly malformed input but produce clear errors."
    )

    # Inversion of Control / Dependency Injection
    examples['InversionOfControl'] = (
        "Pass dependencies (e.g., tax strategy) into constructors instead of creating them inside."
    )

    # Maximise Cohesion
    examples['MaximiseCohesion'] = (
        "Group related functions into the same module/class (e.g., PaymentCalculator)."
    )

    # Liskov Substitution Principle
    examples['LiskovSubstitution'] = (
        "Subtypes (Sparrow) should work wherever base types (Bird) are expected."
    )

    # Open/Closed Principle
    examples['OpenClosed'] = (
        "Use abstract validators (PaymentValidator) so new validators can be added."
    )

    # Single Responsibility Principle
    examples['SingleResponsibility'] = (
        "PaymentCalculator does only payment calculations; repository handles storage."
    )

    # Hide Implementation Details
    examples['HideImplementationDetails'] = (
        "Use private attributes/methods (e.g., BankAccount.__log_transaction)."
    )

    # Curly's Law (You ain't gonna need it variant / keep small)
    examples['CurlysLaw'] = (
        "Prefer small, composable functions instead of giant monoliths."
    )

    # Encapsulate What Changes
    examples['EncapsulateWhatChanges'] = (
        "Wrap volatile code behind an interface (e.g., DatabaseConnection)."
    )

    # Interface Segregation Principle
    examples['InterfaceSegregation'] = (
        "Provide small, focused interfaces (Worker vs Eater) so clients only depend on needed methods."
    )

    # Boy-Scout Rule (leave code cleaner than you found it)
    examples['BoyScoutRule'] = (
        "Refactor small things when you touch code to improve readability and reduce debt."
    )

    # Command Query Separation
    examples['CommandQuerySeparation'] = (
        "Commands modify state (StartEngineCommand.execute), queries return data (get_balance)."
    )

    # Murphy's Law
    examples['MurphysLaw'] = (
        "Plan for failure: add defensive checks and tests for edge cases."
    )

    # Brooks's Law
    examples['BrooksLaw'] = (
        "Adding people to a late project makes it later — prefer small, focused teams."
    )

    # Linus's Law
    examples['LinusLaw'] = (
        "Given enough eyeballs, all bugs are shallow — use code review and open collaboration."
    )

    return examples


# =============================================
# DESIGN PATTERNS INDEX (stubs + short examples)
# Provide a mapping of many pattern names -> short description or small stub
# This keeps the file readable while offering pointers for future expansion.
# =============================================

design_patterns_index: Dict[str, str] = {
    # Creational
    'Abstract Document': 'Pattern for dynamic documents; stub description',
    'Abstract Factory': 'Create families of related objects; stub',
    'Active Object': 'Concurrency pattern; stub',
    'Actor Model': 'Concurrency model using actors; stub',
    'Builder': 'Fluent builder example (see CarBuilder)',
    'Factory': 'Factory method (see PaymentMethodFactory)',
    'Factory Method': 'Factory method variant; stub',
    'Singleton': 'Single instance accessor (see DatabaseConnection)',
    'Multiton': 'Multiple named singletons; stub',
    'Prototype': 'Clone prototypical instances; stub',

    # Structural
    'Adapter': 'See PaymentProcessorAdapter',
    'Ambassador': 'Remote proxy pattern; stub',
    'Bridge': 'Separate abstraction from implementation; stub',
    'Composite': 'See ServiceComposite/ServiceLeaf',
    'Decorator': 'See logging_decorator & PaymentService',
    'Facade': 'Provide a simplified interface over complex subsystems',
    'Flyweight': 'Share fine-grained objects to save memory; stub',
    'Proxy': 'See DatabaseConnectionProxy',
    'Virtual Proxy': 'Lazy-loading proxy; stub',

    # Behavioral
    'Observer': 'See OrderStatus/CustomerNotifier',
    'Strategy': 'See TaxStrategy/Checkout',
    'Command': 'See Command and Start/StopEngineCommand',
    'State': 'See SimpleOrder and state classes',
    'Chain of Responsibility': 'Chain handlers until one handles the request; stub',
    'Iterator': 'Traverse collections; Python iterator protocol applies',
    'Mediator': 'Coordinate communication between objects; stub',
    'Memento': 'Capture and restore object state; stub',
    'Template Method': 'Define skeleton algorithm in base class; stub',
    'Visitor': 'Add operations to object structures without changing them; stub',

    # Architectural
    'MVC': 'Model-View-Controller (see lightweight Model/View/Controller)',
    'Layered Architecture': 'Separate concerns into layers; stub',
    'Clean Architecture': 'Use boundaries and dependency rules; stub',
    'Hexagonal Architecture': 'Ports-and-adapters; stub',

    # Concurrency
    'Producer-Consumer': 'See producer/consumer using queue',
    'Thread-Pool Executor': 'See concurrent.futures usage',
    'Leader-Followers': 'Concurrency coordination pattern; stub',
    'Poison Pill': 'Shutdown sentinel in queues (we use None sentinel)',

    # Microservices & Integration
    'API Gateway': 'Gateway pattern for microservices; stub',
    'Event Sourcing': 'Persist events as source of truth; stub',
    'Saga': 'Manage long-running transactions; stub',
    'Repository': 'See OrderRepository',
    'Service Locator': 'See ServiceLocator',
    'Data Transfer Object (DTO)': 'See OrderDTO dataclass',

    # Functional / Utilities
    'Currying': 'See curry_add',
    'Function Composition': 'See compose',
    'Monad': 'Functional abstraction - conceptual stub',

    # Integration / Data Access
    'DAO': 'Data Access Object pattern; stub',
    'Data Mapper': 'Map between domain objects and DB; stub',

    # Resilience / Reliability
    'Circuit Breaker': 'Protect remote calls; stub',
    'Retry': 'Retry transient failures; stub',
    'Backpressure': 'Regulate load between producers and consumers; stub',

    # Misc / Other
    'Decorator (Structural)': 'See logging_decorator example',
    'Proxy (Dynamic Proxy)': 'Use Python dynamic proxies / wrappers; stub',
    'Null Object': 'Provide neutral object instead of None; stub',
    'Strategy (Behavioral)': 'See TaxStrategy',
    'Observer (Publish-Subscribe)': 'See OrderStatus',

    # Large list mapping: mark many additional patterns as "listed - stub" so
    # they appear in the index and can be expanded later.
    'Abstract Factory (listed)': 'listed - stub',
    'Abstract Document (listed)': 'listed - stub',
    'Acyclic Visitor': 'listed - stub',
    'Anti-Corruption Layer': 'listed - stub',
    'Async Method Invocation': 'listed - stub',
    'Balking': 'listed - stub',
    'Bloc': 'listed - stub',
    'Business Delegate': 'listed - stub',
    'Caching': 'listed - stub',
    'Callback': 'listed - stub',
    'Circuit Breaker (listed)': 'listed - stub',
    'Clean Architecture (listed)': 'listed - stub',
    'Command Query Responsibility Segregation (CQRS)': 'listed - stub',
    'Context Object': 'listed - stub',
    'Curiously Recurring Template Pattern (CRTP)': 'listed - stub',
    'Data Bus': 'listed - stub',
    'Data Locality': 'listed - stub',
    'Data Mapper (listed)': 'listed - stub',
    'Decorator (listed)': 'listed - stub',
    'Dependency Injection': 'listed - stub (IoC via constructor injection)',
    'Dirty Flag': 'listed - stub',
    'Domain Model': 'listed - stub',
    'Double-Checked Locking': 'listed - stub',
    'Double Dispatch': 'listed - stub',
    'Dynamic Proxy': 'listed - stub',
    'Event Aggregator': 'listed - stub',
    'Event-Driven Architecture': 'listed - stub',
    'Event Queue': 'listed - stub',
    'Event Sourcing (listed)': 'listed - stub',
    'Execute Around': 'listed - stub',
    'Extension Objects': 'listed - stub',
    'Factory Kit': 'listed - stub',
    'Feature Toggle': 'listed - stub',
    'Fluent Interface': 'listed - stub',
    'Flux': 'listed - stub',
    'Front Controller': 'listed - stub',
    'Game Loop': 'listed - stub',
    'Gateway': 'listed - stub',
    'Guarded Suspension': 'listed - stub',
    'Half-Sync/Half-Async': 'listed - stub',
    'Health Check': 'listed - stub',
    'Hexagonal Architecture (listed)': 'listed - stub',
    'Identity Map': 'listed - stub',
    'Intercepting Filter': 'listed - stub',
    'Interpreter': 'listed - stub',
    'Iterator (listed)': 'listed - stub',
    'Layered Architecture (listed)': 'listed - stub',
    'Lazy Loading': 'listed - stub',
    'Leader Election': 'listed - stub',
    'Leader-Followers (listed)': 'listed - stub',
    'Lockable Object': 'listed - stub',
    'MapReduce': 'listed - stub',
    'Marker Interface': 'listed - stub',
    'Master-Worker': 'listed - stub',
    'Mediator (listed)': 'listed - stub',
    'Memento (listed)': 'listed - stub',
    'Microservices Aggregator': 'listed - stub',
    'Microservices API Gateway': 'listed - stub',
    'Model-View-Intent (MVI)': 'listed - stub',
    'Model-View-Presenter (MVP)': 'listed - stub',
    'Model-View-ViewModel': 'listed - stub',
    'Money': 'listed - stub',
    'Monitor': 'listed - stub',
    'Monolithic Architecture': 'listed - stub',
    'Monostate': 'listed - stub',
    'Mute Idiom': 'listed - stub',
    'Naked Objects': 'listed - stub',
    'Notification': 'listed - stub',
    'Null Object (listed)': 'listed - stub',
    'Object Pool': 'listed - stub',
    'Object Mother': 'listed - stub',
    'Optimistic Offline Lock': 'listed - stub',
    'Page Controller': 'listed - stub',
    'Page Object': 'listed - stub',
    'Parameter Object': 'listed - stub',
    'Pipeline': 'listed - stub',
    'Poison Pill (listed)': 'listed - stub',
    'Presentation Model': 'listed - stub',
    'Private Class Data': 'listed - stub',
    'Producer-Consumer (listed)': 'listed - stub',
    'Promise': 'listed - stub',
    'Property': 'listed - stub',
    'Prototype (listed)': 'listed - stub',
    'Publish-Subscribe': 'listed - stub',
    'Queue-Based Load Leveling': 'listed - stub',
    'Reactor': 'listed - stub',
    'Registry': 'listed - stub',
    'Resource Acquisition Is Initialization (RAII)': 'listed - stub',
    'Role Object': 'listed - stub',
    'Saga (listed)': 'listed - stub',
    'Separated Interface': 'listed - stub',
    'Servant': 'listed - stub',
    'Service Layer': 'listed - stub',
    'Service Locator (listed)': 'listed - stub',
    'Session Facade': 'listed - stub',
    'Sharding': 'listed - stub',
    'Single Table Inheritance': 'listed - stub',
    'Singleton (listed)': 'listed - stub',
    'Special Case': 'listed - stub',
    'Specification': 'listed - stub',
    'Step Builder': 'listed - stub',
    'Strangler': 'listed - stub',
    'Strategy (listed)': 'listed - stub',
    'Subclass Sandbox': 'listed - stub',
    'Table Module': 'listed - stub',
    'Template View': 'listed - stub',
    'Throttling': 'listed - stub',
    'Tolerant Reader': 'listed - stub',
    'Transaction Script': 'listed - stub',
    'Twin': 'listed - stub',
    'Type Object': 'listed - stub',
    'Unit of Work': 'listed - stub',
    'Value Object': 'listed - stub',
    'Visitor (listed)': 'listed - stub',
}


def list_design_patterns() -> List[str]:
    """Return a sorted list of patterns included (index keys)."""
    return sorted(design_patterns_index.keys())


# =============================================
# Additional pattern implementations (first batch)
# - Prototype
# - Facade
# - Chain of Responsibility
# - Iterator for Fleet
# - Null Object
# - Template Method
# - Memento (snapshot/restore)
# =============================================

# Prototype Pattern: shallow clone of vehicles
import copy

class Prototype(ABC):
    @abstractmethod
    def clone(self):
        pass

class CarPrototype(Car, Prototype):
    def __init__(self, make: str, model: str, year: int, seats: int):
        super().__init__(make, model, year, seats)
        # add a unique id to show cloning difference
        self.prototype_id = uuid.uuid4().hex

    def clone(self) -> 'CarPrototype':
        cloned = copy.deepcopy(self)
        cloned.prototype_id = uuid.uuid4().hex
        return cloned


# Facade Pattern: simplify rental workflow
class RentalFacade:
    """Simplified interface for renting a vehicle."""

    def __init__(self, fleet: Fleet, payment_processor: PaymentProcessor):
        self.fleet = fleet
        self.processor = payment_processor

    def rent_vehicle(self, vehicle_index: int, amount: Decimal) -> bool:
        try:
            vehicle = self.fleet.vehicles[vehicle_index]
        except IndexError:
            logger.error("Vehicle not available")
            return False
        # Charge customer
        success = self.processor.process_payment(amount)
        if success:
            logger.info(f"Rented {vehicle.get_info()} for {amount}")
            return True
        return False


# Chain of Responsibility: simple request handlers
class Handler(ABC):
    def __init__(self, successor: Optional['Handler'] = None):
        self.successor = successor

    @abstractmethod
    def handle(self, request: Dict[str, Any]) -> bool:
        pass

class AuthHandler(Handler):
    def handle(self, request: Dict[str, Any]) -> bool:
        if not request.get('user'):
            logger.info('AuthHandler: no user')
            return False
        logger.info('AuthHandler: passed')
        return self.successor.handle(request) if self.successor else True

class ValidateHandler(Handler):
    def handle(self, request: Dict[str, Any]) -> bool:
        if 'amount' not in request:
            logger.info('ValidateHandler: missing amount')
            return False
        logger.info('ValidateHandler: passed')
        return self.successor.handle(request) if self.successor else True

class FinalHandler(Handler):
    def handle(self, request: Dict[str, Any]) -> bool:
        logger.info('FinalHandler: processing request')
        return True


# Iterator for Fleet
class FleetIterator:
    def __init__(self, fleet: Fleet):
        self._fleet = fleet
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self) -> Vehicle:
        if self._index >= len(self._fleet.vehicles):
            raise StopIteration
        v = self._fleet.vehicles[self._index]
        self._index += 1
        return v

def iter_fleet(fleet: Fleet):
    return FleetIterator(fleet)


# Null Object: payment method that does nothing
class NullPaymentMethod(PaymentMethod):
    def process(self, amount: Decimal) -> bool:
        logger.info('NullPaymentMethod: no-op')
        return False


# Template Method: report generation skeleton
class Report(ABC):
    def generate(self) -> str:
        parts = [self.header(), self.body(), self.footer()]
        return '\n'.join(parts)

    @abstractmethod
    def header(self) -> str:
        pass

    @abstractmethod
    def body(self) -> str:
        pass

    def footer(self) -> str:
        return f"Generated at {datetime.now().isoformat()}"

class SalesReport(Report):
    def header(self) -> str:
        return "Sales Report"

    def body(self) -> str:
        # dummy content for example
        return "Total Sales: $1000"


# Memento: capture and restore BankAccount state
class AccountMemento:
    def __init__(self, balance: Decimal):
        self.balance = balance

class AccountCaretaker:
    def __init__(self):
        self._mementos: List[AccountMemento] = []

    def save(self, account: BankAccount) -> None:
        self._mementos.append(AccountMemento(account.get_balance()))

    def restore_last(self, account: BankAccount) -> bool:
        if not self._mementos:
            return False
        m = self._mementos.pop()
        # direct access is not possible (balance is private). We simulate
        # restoration by using withdraw/deposit to reach target balance.
        current = account.get_balance()
        target = m.balance
        if current < target:
            account.deposit(target - current)
        elif current > target:
            account.withdraw(current - target)
        return True

