from decimal import Decimal

from advanced_oop_principles import (
    EventStore,
    Event,
    SagaCoordinator,
    OrderRepository,
    OrderDAO,
    OrderDomain,
    OrderDataMapper,
    CommandBus,
    QueryBus,
    EventBus,
    HealthCheckRegistry,
    ObjectPool,
    DatabaseConnection,
    FeatureToggle,
)


def test_event_store_and_saga():
    store = EventStore()
    saga = SagaCoordinator(store)
    captured = {}

    def handler(e: Event):
        captured[e.type] = e.payload

    saga.register('OrderCreated', handler)
    e = Event('ord1', 'OrderCreated', {'amount': 10})
    saga.handle(e)
    assert 'OrderCreated' in captured


def test_data_mapper_and_dao_uow():
    repo = OrderRepository()
    dao = OrderDAO(repo)
    domain = OrderDomain('id1', Decimal('30'))
    dto = OrderDataMapper.to_dto(domain)
    dao.create_order(dto.amount)
    # create_order returns a new DTO id - ensure repo is not empty
    assert repo._store


def test_cqrs_buses():
    cb = CommandBus()
    qb = QueryBus()

    cb.register('inc', lambda x: x + 1)
    qb.register('echo', lambda x: x)

    assert cb.handle('inc', 1) == 2
    assert qb.handle('echo', 'hi') == 'hi'


def test_event_bus_publish_subscribe():
    bus = EventBus()
    received = {}

    def on_order(e):
        received[e.type] = e.payload

    bus.subscribe('T', on_order)
    bus.publish(Event('a', 'T', {'x': 1}))
    assert 'T' in received


def test_health_checks_and_object_pool_and_feature_toggle():
    registry = HealthCheckRegistry()
    registry.register('ok', lambda: True)
    results = registry.run_all()
    assert results['ok'] is True

    pool = ObjectPool(lambda: DatabaseConnection(), max_size=2)
    c1 = pool.acquire()
    pool.release(c1)
    c2 = pool.acquire()
    assert c2 is not None

    flags = FeatureToggle()
    flags.enable('f1')
    assert flags.is_enabled('f1') is True
