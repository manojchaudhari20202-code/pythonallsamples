from decimal import Decimal
import pytest

from advanced_oop_principles import (
    EngineFlyweightFactory,
    ConsoleRenderer,
    ReportAbstraction,
    ElectricCar,
    OdometerVisitor,
    parse_expression,
    OrderRepository,
    OrderDAO,
    UnitOfWork,
    retry,
    CircuitBreaker,
    Mediator,
    MediatedDriver,
)


def test_flyweight_shares_specs():
    a = EngineFlyweightFactory.get_spec(200, 'gas')
    b = EngineFlyweightFactory.get_spec(200, 'gas')
    assert a is b


def test_bridge_renderers():
    renderer = ConsoleRenderer()
    report = ReportAbstraction(renderer)
    out = report.produce({'title': 'T', 'content': 'C'})
    assert 'T' in out and 'C' in out


def test_visitor_on_vehicle():
    car = ElectricCar('Tesla', 'S', 2021, 5, 75)
    visitor = OdometerVisitor()
    res = car.accept(visitor)
    assert 'Visited' in res


def test_interpreter_add():
    expr = parse_expression('add 2 3')
    assert expr.interpret() == 5


def test_dao_and_uow():
    repo = OrderRepository()
    dao = OrderDAO(repo)
    uow = UnitOfWork(repo)
    dto = dao.create_order(Decimal('20'))
    uow.register_new_order(dto)
    uow.commit()
    found = dao.find(dto.id)
    assert found is not None


def test_retry_decorator_succeeds_after_failure():
    calls = {'n': 0}

    @retry(times=3, exceptions=(ValueError,))
    def flaky():
        calls['n'] += 1
        if calls['n'] < 2:
            raise ValueError('fail once')
        return 'ok'

    assert flaky() == 'ok'


def test_circuit_breaker_opens():
    cb = CircuitBreaker(max_failures=2)

    def fail():
        raise RuntimeError('boom')

    with pytest.raises(RuntimeError):
        cb.call(fail)
    with pytest.raises(RuntimeError):
        cb.call(fail)
    with pytest.raises(RuntimeError):
        cb.call(fail)


def test_mediator_broadcast():
    mediator = Mediator()
    d1 = MediatedDriver('d1', 'L1', mediator)
    d2 = MediatedDriver('d2', 'L2', mediator)
    # ensure no exceptions when notifying
    mediator.notify('d1', 'event', {'x': 1})
