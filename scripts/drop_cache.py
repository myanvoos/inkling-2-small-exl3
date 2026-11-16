#!/usr/bin/env python3
# The sudoless "drop caches": the serve drops the page cache with it on both nodes before the load when the root
# helper (DROP_HELPER) is not available.
"""drop-cache.py [ROOT...]: the sudoless "drop caches". fadvise(DONTNEED) on every regular file of at least 1 MiB
under each ROOT (default: the models, the inference artifacts, ~/.cache, ~/.nv, /var/crash, /tmp), so the page cache
gives its memory back (GB10: CPU and GPU share it; CUDA does not reclaim the cache the way MemAvailable assumes).
Only clean, unmapped pages go (a running process's mapped files stay); nothing is written. Prints /proc/meminfo's
Cached / Inactive(file) / MemFree / MemAvailable before and after. Threads walk the roots' subdirectories in parallel."""
import os, sys, time
from concurrent.futures import ThreadPoolExecutor

MIN = 1 << 20
G = 2 ** 30


def meminfo():
    m = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        m[k] = int(v.split()[0]) * 1024
    return m


def drop(path):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError:
        return 0
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return 1
    except OSError:
        return 0
    finally:
        os.close(fd)


def walk(top):
    n, stack = 0, [top]
    while stack:
        d = stack.pop()
        try:
            it = os.scandir(d)
        except OSError:
            continue
        with it:
            for e in it:
                try:
                    if e.is_dir(follow_symlinks=False):
                        if e.name not in (".git", "node_modules"):
                            stack.append(e.path)
                    elif e.is_file(follow_symlinks=False) and e.stat(follow_symlinks=False).st_size >= MIN:
                        n += drop(e.path)
                except OSError:
                    pass
    return n


def main():
    home = os.path.expanduser("~")
    roots = sys.argv[1:] or [f"{home}/models", f"{home}/inference", f"{home}/.cache", f"{home}/.nv",
                             "/var/crash", "/tmp"]
    tops, n = [], 0
    for r in roots:
        try:
            with os.scandir(r) as it:
                for e in it:
                    if e.is_dir(follow_symlinks=False):
                        tops.append(e.path)
                    elif e.is_file(follow_symlinks=False) and e.stat(follow_symlinks=False).st_size >= MIN:
                        n += drop(e.path)
        except OSError:
            continue
    a, t = meminfo(), time.time()
    with ThreadPoolExecutor(max_workers=16) as ex:
        n += sum(ex.map(walk, tops))
    b = meminfo()
    print(f"drop-cache: {n} files in {time.time() - t:.1f} s; Cached {a['Cached']/G:.2f} -> {b['Cached']/G:.2f} GiB, "
          f"Inactive(file) {a['Inactive(file)']/G:.2f} -> {b['Inactive(file)']/G:.2f}, MemFree {a['MemFree']/G:.2f} -> "
          f"{b['MemFree']/G:.2f}, MemAvailable {a['MemAvailable']/G:.2f} -> {b['MemAvailable']/G:.2f} GiB")


if __name__ == "__main__":
    main()
