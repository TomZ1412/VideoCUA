"""Compatibility guards for vLLM's multimodal transport cache."""
import logging


logger = logging.getLogger('vllm.qwenvl_audio.cache')


def limit_shm_object_size(cache):
    """Keep oversized AV bundles on vLLM's normal uncached transport path.

    vLLM 0.26's ring stores the allocation size as a signed 32-bit integer,
    including its header. Allowing 2 GiB or more raises OverflowError during
    allocation; the sender only handles ValueError/MemoryError as cache misses.
    Limit the object store before it allocates, so its serialized-size check
    (including object metadata and reader flags) takes the native fallback.
    """
    from vllm.multimodal.cache import ShmObjectStoreSenderCache

    if not isinstance(cache, ShmObjectStoreSenderCache):
        return
    store = cache._shm_cache
    ring = store.ring_buffer
    # int2byte uses ID_NBYTES for both fields in vLLM 0.26.
    width = min(ring.ID_NBYTES, ring.SIZE_NBYTES)
    max_size = (1 << (8 * width - 1)) - 1 - ring.MD_SIZE
    if store.max_object_size > max_size:
        logger.warning(
            'Limiting AV SHM object size from %s to %s bytes to fit the '
            'ring header; larger bundles use uncached transport.',
            store.max_object_size, max_size,
        )
        store.max_object_size = max_size
