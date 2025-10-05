from decimal import Decimal
import queue

from advanced_oop_principles import (
    CarPrototype,
    Fleet,
    ElectricCar,
    PaymentMethodFactory,
    PaymentProcessorAdapter,
    NullPaymentMethod,
    RentalFacade,
    AuthHandler,
    ValidateHandler,
    FinalHandler,
    iter_fleet,
    SalesReport,
    BankAccount,
    AccountCaretaker,
)


def test_prototype_clone():
    car = CarPrototype('Honda', 'Accord', 2020, 5)
    clone = car.clone()
    assert car is not clone
    assert car.make == clone.make


def test_rental_facade_with_null_payment():
    fleet = Fleet('Test')
    fleet.add_vehicle(ElectricCar('Tesla', 'S', 2021, 5, 75))
    null_processor = NullPaymentMethod()
    facade = RentalFacade(fleet, PaymentProcessorAdapter(null_processor))
    result = facade.rent_vehicle(0, Decimal('100'))
    assert result is False


def test_chain_of_responsibility_success():
    chain = AuthHandler(ValidateHandler(FinalHandler()))
    req = {'user': 'u1', 'amount': 10}
    assert chain.handle(req) is True


def test_fleet_iterator():
    fleet = Fleet('Iter')
    fleet.add_vehicle(ElectricCar('Tesla', '3', 2023, 5, 75))
    fleet.add_vehicle(ElectricCar('Nissan', 'Leaf', 2020, 5, 40))
    collected = [v for v in iter_fleet(fleet)]
    assert len(collected) == 2


def test_template_method_report():
    r = SalesReport()
    output = r.generate()
    assert 'Sales Report' in output


def test_memento_restore():
    acct = BankAccount('111', Decimal('100'))
    caretaker = AccountCaretaker()
    caretaker.save(acct)
    acct.deposit(Decimal('50'))
    assert acct.get_balance() == Decimal('150')
    restored = caretaker.restore_last(acct)
    assert restored is True
    assert acct.get_balance() == Decimal('100')
