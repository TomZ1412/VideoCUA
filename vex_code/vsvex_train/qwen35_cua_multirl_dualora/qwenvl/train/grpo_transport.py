"""Bounded CUDA staging for CPU payloads on node-local NCCL groups."""
import pickle

import torch
import torch.distributed as dist


CHUNK_BYTES = 64 * 1024 * 1024


def broadcast_cpu_tensor(tensor, src, group, chunk_bytes=CHUNK_BYTES):
    """Broadcast a contiguous CPU tensor without materializing it all on CUDA."""
    if tensor.device.type != 'cpu' or not tensor.is_contiguous():
        raise ValueError('Expected a contiguous CPU tensor')
    if dist.get_world_size(group) == 1 or tensor.numel() == 0:
        return tensor
    flat = tensor.view(-1)
    count = max(1, chunk_bytes // tensor.element_size())
    staging = torch.empty(min(count, flat.numel()), dtype=tensor.dtype,
                          device=torch.device('cuda', torch.cuda.current_device()))
    root = dist.get_rank() == src
    for start in range(0, flat.numel(), count):
        part = staging[:min(count, flat.numel() - start)]
        if root:
            part.copy_(flat[start:start + part.numel()])
        dist.broadcast(part, src=src, group=group)
        if not root:
            flat[start:start + part.numel()].copy_(part)
    # Calls use the current stream; subsequent CUDA allocator reuse is ordered.
    return tensor


def broadcast_object(obj, src, group):
    """Serialize once at source; broadcast actual bytes, with no padded gather."""
    if dist.get_world_size(group) == 1:
        return obj
    root = dist.get_rank() == src
    payload = bytearray(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)) if root else None
    size = torch.tensor([len(payload) if root else 0], dtype=torch.int64,
                        device=torch.device('cuda', torch.cuda.current_device()))
    dist.broadcast(size, src=src, group=group)
    data = torch.frombuffer(payload, dtype=torch.uint8) if root else torch.empty(size.item(), dtype=torch.uint8)
    broadcast_cpu_tensor(data, src, group)
    return obj if root else pickle.loads(data.numpy().tobytes())


def transfer_object(obj, src, dst, group, chunk_bytes=CHUNK_BYTES):
    """Point-to-point payload transfer; uninvolved ranks allocate nothing.

    The NCCL group must already have participated in a collective (the rollout
    metadata broadcast does this), as required by batch_isend_irecv.
    """
    rank = dist.get_rank()
    if src == dst:
        return obj if rank == dst else None
    if rank not in (src, dst):
        return None
    device = torch.device('cuda', torch.cuda.current_device())
    sending = rank == src
    payload = bytearray(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)) if sending else None
    size = torch.tensor([len(payload) if sending else 0], dtype=torch.int64, device=device)

    def exchange(tensor):
        # Coalesced P2P reuses the group's existing communicator instead of
        # creating a separate communicator for every source/destination pair.
        op = dist.P2POp(dist.isend if sending else dist.irecv, tensor,
                        dst if sending else src, group=group)
        for handle in dist.batch_isend_irecv([op]):
            handle.wait()

    exchange(size)
    data = torch.frombuffer(payload, dtype=torch.uint8) if sending else torch.empty(size.item(), dtype=torch.uint8)
    staging = torch.empty(min(chunk_bytes, data.numel()), dtype=torch.uint8, device=device)
    for start in range(0, data.numel(), chunk_bytes):
        part = staging[:min(chunk_bytes, data.numel() - start)]
        if sending:
            part.copy_(data[start:start + part.numel()])
            exchange(part)
        else:
            exchange(part)
            data[start:start + part.numel()].copy_(part)
    return pickle.loads(data.numpy().tobytes()) if not sending else None


def exchange_rollout_groups(local_groups, metadata, assignments, group, node_start):
    """Only selected groups travel, once, directly from source to destination.

    Metadata and assignments are identical on every node participant. Source
    offsets distinguish repeated occurrences of the same dataset index.
    """
    local_rank = dist.get_rank() - node_start
    received = {}
    for src in sorted({g['source'] for g in metadata}):
        for dst, assigned in enumerate(assignments):
            selected = [g for g in assigned if g['source'] == src]
            if not selected:
                continue
            payload = [local_groups[g['offset']] for g in selected] if local_rank == src else None
            payload = transfer_object(payload, node_start + src, node_start + dst, group)
            if local_rank == dst:
                received.update(((g['source'], g['offset']), value)
                                for g, value in zip(selected, payload, strict=True))
    return [received[g['source'], g['offset']] for g in assignments[local_rank]]
