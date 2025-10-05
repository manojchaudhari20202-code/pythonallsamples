import time

from advanced_oop_principles import (
    LRUCache,
    LazyProxy,
    ElectricCar,
    BackpressureQueue,
    double_dispatch_interact,
    DynamicProxy,
    EventAggregator,
    Event,
    throttle,
)


def test_lru_cache_eviction():
    c = LRUCache(capacity=2)
    c.put('a', 1)
    c.put('b', 2)
    c.put('c', 3)
    assert c.get('a') is None


def test_lazy_proxy_creation():
    created = {'n': 0}

    def factory():
        created['n'] += 1
        return ElectricCar('Lazy', 'X', 2020, 4, 50)

    p = LazyProxy(factory)
    # not created until attribute access
    assert created['n'] == 0
    _ = p.get_info()
    assert created['n'] == 1


def test_backpressure_queue():
    bp = BackpressureQueue(maxsize=1)
    assert bp.try_put(1) is True
    # second put should fail due to full
    assert bp.try_put(2) is False


def test_double_dispatch():
    a = ElectricCar('A', 'M', 2021, 5, 60)
    b = ElectricCar('B', 'N', 2022, 5, 70)
    res = double_dispatch_interact(a, b)
    assert 'interacts' in res or 'hum' in res or 'hums' in res


def test_dynamic_proxy_logs():
    car = ElectricCar('D', 'P', 2021, 4, 60)
    proxy = DynamicProxy(car)
    # ensure calling a method through proxy works
    info = proxy.get_info()
    assert '202' in info


def test_event_aggregator_batch_dispatch():
    agg = EventAggregator()
    rec = {}

    def h(e):
        rec[e.type] = e.payload

    agg.subscribe('X', h)
    agg.publish(Event('id', 'X', {'v': 1}))
    agg.dispatch()
    assert 'X' in rec


def test_throttle_decorator_runs():
    @throttle(10)
    def fast():
        return True

    assert fast() is True
