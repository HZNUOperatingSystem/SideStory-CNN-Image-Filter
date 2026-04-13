from collections.abc import Iterable
from typing import Protocol

from torch.utils.data import DataLoader


class _ClosableQueue(Protocol):
    def cancel_join_thread(self) -> None: ...

    def close(self) -> None: ...

    def put(self, item: object) -> None: ...


class _TerminableWorker(Protocol):
    def is_alive(self) -> bool: ...

    def terminate(self) -> None: ...

    def join(self, timeout: float | None = None) -> None: ...

    def kill(self) -> None: ...


def shutdown_loader_workers(loader: DataLoader | None) -> None:
    if loader is None:
        return
    iterator = getattr(loader, '_iterator', None)
    if iterator is None:
        return

    if hasattr(iterator, '_shutdown'):
        iterator._shutdown = True
    _shutdown_pin_memory_thread(iterator)
    index_queues = getattr(iterator, '_index_queues', None)
    if index_queues is not None:
        for queue in index_queues:
            _close_queue(queue)
    _close_queue(getattr(iterator, '_worker_result_queue', None))
    _terminate_workers(getattr(iterator, '_workers', None))
    loader._iterator = None


def _close_queue(queue: _ClosableQueue | None) -> None:
    if queue is None:
        return
    cancel_join_thread = getattr(queue, 'cancel_join_thread', None)
    close = getattr(queue, 'close', None)
    if callable(cancel_join_thread):
        cancel_join_thread()
    if callable(close):
        close()


def _shutdown_pin_memory_thread(iterator: object) -> None:
    done_event = getattr(iterator, '_pin_memory_thread_done_event', None)
    if done_event is not None:
        done_event.set()

    worker_result_queue = getattr(iterator, '_worker_result_queue', None)
    if worker_result_queue is not None:
        _wake_pin_memory_thread(worker_result_queue)

    pin_memory_thread = getattr(iterator, '_pin_memory_thread', None)
    join = getattr(pin_memory_thread, 'join', None)
    if callable(join):
        join(timeout=0.5)


def _wake_pin_memory_thread(queue: _ClosableQueue | None) -> None:
    if queue is None:
        return
    put = getattr(queue, 'put', None)
    if callable(put):
        try:
            put((None, None))
        except ValueError:
            return


def _terminate_workers(
    workers: Iterable[_TerminableWorker] | None,
) -> None:
    if workers is None:
        return
    for worker in workers:
        _terminate_worker(worker)


def _terminate_worker(worker: _TerminableWorker) -> None:
    if worker.is_alive():
        worker.terminate()
        worker.join(timeout=0.5)
    if worker.is_alive():
        worker.kill()
        worker.join(timeout=0.5)
