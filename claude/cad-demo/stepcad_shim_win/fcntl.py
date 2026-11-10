"""Заглушка fcntl для Windows: StepCAD берёт блокировку файла кэша, а в Windows модуля fcntl нет.
Кандидат StepCAD работает в одном процессе-писателе, поэтому блокировка не нужна."""
LOCK_SH, LOCK_EX, LOCK_NB, LOCK_UN = 1, 2, 4, 8


def flock(fd, op):  # noqa: ARG001
    return None


def lockf(fd, op, *a):  # noqa: ARG001
    return None
