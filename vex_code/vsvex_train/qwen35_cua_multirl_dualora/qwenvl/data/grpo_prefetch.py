"""Background media IO with cross-step caching and training lookahead."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager


MEDIA_WORKERS = 8


@contextmanager
def prefetch_media(dataset, entries, worker=None):
    """Yield (entry, media), preloading up to eight entries on CPU threads.

    Entries are (dataset_index, payload). Results preserve entry order while
    preparation runs concurrently. An owned pool is joined on context exit; a
    supplied persistent worker is joined by the trainer at the end of training,
    including failure paths.
    """
    executor = None if worker is not None else ThreadPoolExecutor(
        max_workers=MEDIA_WORKERS, thread_name_prefix='grpo-media')
    submit = worker.submit if worker is not None else lambda index: executor.submit(dataset.prepare, index)

    def iterate():
        source = iter(entries)
        pending = deque()
        for entry in source:
            pending.append((entry, submit(entry[0])))
            if len(pending) == MEDIA_WORKERS:
                break
        for upcoming in source:
            current, future = pending.popleft()
            item = future.result()
            pending.append((upcoming, submit(upcoming[0])))
            yield current, item
            del item
        while pending:
            current, future = pending.popleft()
            item = future.result()
            yield current, item
            del item

    iterator = iterate()
    try:
        yield iterator
    finally:
        iterator.close()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)


class MediaWorker:
    """Persistent eight-thread CPU reader: demand first, then next-step plan."""
    def __init__(self, dataset):
        from collections import OrderedDict, deque
        import threading
        self.dataset = dataset
        self._condition = threading.Condition()
        self._cache, self._demands = OrderedDict(), OrderedDict()
        self._queue, self._desired = deque(), set()
        self._inflight = set()
        self._closed = False
        self._threads = [
            threading.Thread(target=self._run, name=f'grpo-media-{i}', daemon=True)
            for i in range(MEDIA_WORKERS)
        ]
        for thread in self._threads:
            thread.start()

    def plan(self, indices):
        """Replace speculative plan, retaining overlapping completed/inflight IO."""
        from collections import deque
        with self._condition:
            if self._closed:
                raise RuntimeError('Media worker is closed')
            indices = list(dict.fromkeys(indices))
            self._desired = set(indices)
            for index in list(self._cache):
                if index not in self._desired:
                    del self._cache[index]
            self._queue = deque(i for i in indices if i not in self._cache and
                                i not in self._demands and i not in self._inflight)
            self._condition.notify_all()

    def submit(self, index):
        from concurrent.futures import Future
        with self._condition:
            if self._closed:
                raise RuntimeError('Media worker is closed')
            if index in self._demands:
                return self._demands[index]
            future = Future()
            if index in self._cache:
                ok, value = self._cache.pop(index)
                (future.set_result if ok else future.set_exception)(value)
            else:
                self._demands[index] = future
            self._condition.notify_all()
            return future

    def _run(self):
        while True:
            with self._condition:
                while True:
                    if self._closed:
                        return
                    for index in self._demands:
                        if index in self._inflight:
                            continue
                        try:
                            self._queue.remove(index)
                        except ValueError:
                            pass
                        break
                    else:
                        while self._queue and (
                            self._queue[0] in self._cache or
                            self._queue[0] not in self._desired or
                            self._queue[0] in self._inflight
                        ):
                            self._queue.popleft()
                        if self._queue:
                            index = self._queue.popleft()
                            break
                        self._condition.wait()
                        continue
                    break
                self._inflight.add(index)
            try:
                value = self.dataset.prepare(index)
                ok = True
            except Exception as error:
                ok, value = False, error
            with self._condition:
                self._inflight.remove(index)
                future = self._demands.pop(index, None)
                if future is not None and not future.cancelled():
                    (future.set_result if ok else future.set_exception)(value)
                elif index in self._desired and not self._closed:
                    self._cache[index] = (ok, value)
                self._condition.notify_all()
            # Do not keep the last delivered tensor alive on an idle thread.
            del value, future

    def close(self):
        with self._condition:
            self._closed = True
            for future in self._demands.values():
                future.cancel()
            self._condition.notify_all()
        for thread in self._threads:
            thread.join()
        self._cache.clear()
        self._demands.clear()
        self._queue.clear()
