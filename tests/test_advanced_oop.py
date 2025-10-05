import queue
from decimal import Decimal
import pytest

from advanced_oop_principles import (
    CustomerNotifier,
    OrderStatus,
    DatabaseConnection,
    PaymentMethodFactory,
    CarBuilder,
    producer,
    consumer,
)


def test_observer_notification():
    status = OrderStatus()
    notifier = CustomerNotifier("test@example.com")
    status.attach(notifier)
    status.status = "Done"
    assert notifier.last_notification is not None
    assert "Done" in notifier.last_notification


def test_singleton_db():
    db1 = DatabaseConnection()
    db2 = DatabaseConnection()
    assert id(db1) == id(db2)


def test_factory_creates_methods():
    factory = PaymentMethodFactory()
    cc = factory.create_payment_method("credit_card")
    pp = factory.create_payment_method("paypal")
    assert cc is not None
    assert pp is not None


def test_builder_creates_car():
    builder = CarBuilder()
    car = builder.set_make("Honda").set_model("Civic").set_year(2020).build()
    assert car.make == "Honda"
    assert car.model == "Civic"


def test_producer_consumer():
    q = queue.Queue()
    items = [1, 2, 3]
    # run producer and consumer sequentially for test simplicity
    producer(q, items)
    consumed = consumer(q)
    assert consumed == items
