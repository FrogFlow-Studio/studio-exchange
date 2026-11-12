"""Попытка 0 в отдельном процессе, чтобы на сложной детали её можно было остановить по времени.

python geo_worker.py скан_выровненный.stl sheet.json 1|0 ответ.json [walls|surface|smooth]
"""
import hashlib
import json
import os
import tempfile
import time
from contextlib import contextmanager
import sys
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import trimesh

import geometric



def _memory_guard(share: float = 0.45) -> None:
    """Попытки считаются параллельно. Если одна разрастается (черновик из стенок на огромной
    кривой детали брал больше 13 ГБ), она не должна уронить соседние: выходит сама."""
    try:
        import psutil
    except ImportError:
        return
    import os
    import threading
    import time

    limit = share * psutil.virtual_memory().total
    me = psutil.Process()

    def watch() -> None:
        while True:
            if me.memory_info().rss > limit:
                sys.stderr.write(f"попытка заняла больше {limit / 2**30:.1f} ГБ памяти и остановлена\n")
                sys.stderr.flush()
                os._exit(3)
            time.sleep(1)

    threading.Thread(target=watch, daemon=True).start()



@contextmanager
def _cache_lock(path: Path, timeout: float = 1200):
    """One writer per cache entry; the OS releases the lock if a worker is killed."""
    if os.name == "nt":
        import msvcrt

        def lock(file):
            msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        def lock(file):
            fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    deadline = time.monotonic() + timeout
    with path.open("a+b") as file:
        while True:
            try:
                file.seek(0)
                lock(file)
                break
            except OSError as ex:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("ожидание кэша превысило предел") from ex
                time.sleep(min(0.1, remaining))
        yield


def _scan_digest(scan_path: Path) -> str:
    with scan_path.open("rb") as src:
        return hashlib.file_digest(src, "sha256").hexdigest()


def _run_cache(cache: Path, label: str, read, compute, write):
    def load():
        try:
            with cache.open("rb") as stream:
                return read(stream)
        except (OSError, ValueError, TypeError, KeyError, EOFError, BadZipFile):
            return None

    value = load()
    if value is not None:
        print(f"{label}: из кэша", flush=True)
        return value
    with _cache_lock(cache.with_suffix(".lock")):
        value = load()
        if value is not None:
            print(f"{label}: из кэша после ожидания", flush=True)
            return value
        value = compute()
        temp = None
        try:
            with tempfile.NamedTemporaryFile(dir=cache.parent, suffix=".tmp", delete=False) as stream:
                temp = Path(stream.name)
                write(stream, value)
            temp.replace(cache)
        except OSError:
            pass
        finally:
            if temp is not None:
                try:
                    temp.unlink(missing_ok=True)
                except OSError:
                    pass
        return value


def _clean_surface(scan: trimesh.Trimesh, scan_path: Path, answer: Path, *, digest=None) -> trimesh.Trimesh:
    """Share the unflattened mesh between workers of the same run."""
    import surface

    if os.environ.get("CAD_SURFACE_CACHE", "1") == "0":
        return surface.surface_mesh(scan)
    digest = digest or _scan_digest(scan_path)
    cache = answer.parent / f"surface-clean-{digest}-{int(surface.EDGE_KEEP)}-{surface.SMOOTH}.npz"

    def read(stream):
        with np.load(stream, allow_pickle=False) as data:
            return trimesh.Trimesh(data["vertices"], data["faces"], process=False)

    def write(stream, mesh):
        np.savez(stream, vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces))

    return _run_cache(cache, "очистка поверхности", read, lambda: surface.surface_mesh(scan), write)


def _hole_specs(scan, scan_path: Path, answer: Path, *, digest=None):
    import surface

    if os.environ.get("CAD_HOLE_CACHE", "1") == "0":
        return surface.hole_parameters(scan)
    digest = digest or _scan_digest(scan_path)
    cache = answer.parent / f"hole-specs-v1-{digest}.json"

    def read(stream):
        rows = json.load(stream)
        if not isinstance(rows, list):
            raise ValueError("invalid hole cache")
        specs = []
        for c, d, r, h in rows:
            if len(c) != 3 or len(d) != 3:
                raise ValueError("invalid cylinder vectors")
            specs.append((tuple(map(float, c)), tuple(map(float, d)), float(r), float(h)))
        return specs

    def write(stream, rows):
        stream.write(json.dumps(rows).encode("utf-8"))

    return _run_cache(cache, "поиск отверстий", read, lambda: surface.hole_parameters(scan), write)


def _symmetry_restore(clean, timings, scan=None, cache=None):
    """Этап Д (CAD_SYMMETRY=0 — выключить): сколы кривой детали достраиваются отражением целой стороны (этап Д) до построения
    тела. Возвращает (сетка, строки «достроено», зоны для сравнения со сканом)."""
    if os.environ.get("CAD_SYMMETRY", "1") == "0":
        return clean, [], []
    import symmetry

    t = time.perf_counter()
    try:
        # сколы ищутся по сырому скану: на очищенной поверхности излом сглажен и похож на целую
        # поверхность (крышка: неверная плоскость нашла 20 «сколов»); крупный скан прорежен
        raw = clean if scan is None else scan

        def find():
            src = raw
            if len(src.faces) > 200_000:
                import fast_simplification

                v, f = fast_simplification.simplify(np.asarray(src.vertices, np.float32), np.asarray(src.faces),
                                                    target_reduction=1 - 200_000 / len(src.faces))
                src = trimesh.Trimesh(v, f)
            _, log, planes = symmetry.restore_all(src)
            return {"log": log, "planes": planes}

        if cache is None:
            found = find()
        else:   # 0в и 0к ищут одни и те же сколы: второй берёт найденное первым
            import pickle

            found = _run_cache(cache, "поиск сколов по симметрии", lambda s: pickle.load(s), find,
                               lambda s, v: pickle.dump(v, s))
        log, planes = found["log"], found["planes"]
        out = symmetry.apply_to(clean, log)
        if log and out is clean:
            print("    достройка на очищенной поверхности не удалась: тело по скану как есть", flush=True)
            return clean, [], []
    except Exception as error:  # noqa: BLE001 — достройка не удалась: тело по скану как есть
        print(f"    достройка по симметрии не удалась: {type(error).__name__}", flush=True)
        return clean, [], []
    finally:
        timings["symmetry"] = time.perf_counter() - t
    if not planes:
        print("    плоскости симметрии не найдено: достройки нет", flush=True)
    rows = [symmetry.describe(e) for e in log]
    for r in rows:
        print("    " + r, flush=True)
    return out, rows, [symmetry.zone(e) for e in log]


def _chip_restore(clean, timings, scan, cache=None):
    """Этап Д, способ 2 (CAD_CHIPFILL=1): скол без симметрии находится по шероховатому излому сырого
    скана и закрывается продолжением целой поверхности вокруг него (chipfill.py).
    Возвращает (сетка, строки «достроено», зоны для сравнения со сканом)."""
    if os.environ.get("CAD_CHIPFILL", "0") != "1" or scan is None:
        return clean, [], []
    t = time.perf_counter()
    try:
        import chipfill

        def find():
            out, log, _ = chipfill.restore_all(scan, clean)
            return {"vertices": np.asarray(out.vertices), "faces": np.asarray(out.faces),
                    "log": [{k: v for k, v in e.items() if k != "detail"} for e in log]}

        if cache is None:
            found = find()
        else:
            import pickle

            found = _run_cache(cache, "поиск сколов по излому", lambda s: pickle.load(s), find,
                               lambda s, v: pickle.dump(v, s))
        log = found["log"]
        if not log:
            return clean, [], []
        out = trimesh.Trimesh(found["vertices"], found["faces"], process=False)
    except Exception as error:  # noqa: BLE001 — достройка не удалась: тело по скану как есть
        print(f"    достройка по излому не удалась: {type(error).__name__}", flush=True)
        return clean, [], []
    finally:
        timings["chipfill"] = time.perf_counter() - t
    rows = [chipfill.describe(e) for e in log]
    for r in rows:
        print("    " + r, flush=True)
    return out, rows, [chipfill.zone(e) for e in log]


def _restore(clean, timings, scan, digest, answer):
    """Достройка сколов кривой детали: сначала по симметрии, потом по излому."""
    key = lambda name: None if digest is None else answer.parent / f"{name}-{digest}.pkl"
    before = clean
    clean, rows, zones = _symmetry_restore(clean, timings, scan, key("symmetry-v1"))
    clean, rows2, zones2 = _chip_restore(clean, timings, scan, key("chipfill-v1"))
    if rows or rows2:
        _FILLED[:] = [(before, clean)]
    return clean, rows + rows2, zones + zones2


_FILLED = []   # (поверхность до достройки, после): по ним отсеиваются «отверстия», найденные на изломе


def _not_in_fill(specs):
    """Излом скола бывает похож на стенку отверстия. Если середина оси такого «отверстия» до достройки
    была в воздухе, а после оказалась в достроенном материале, это излом, а не отверстие."""
    if not _FILLED or not specs:
        return specs
    before, after = _FILLED[0]
    mid = np.array([np.asarray(c) + np.asarray(d) * h / 2 for c, d, _, h in specs])
    try:
        was = trimesh.proximity.signed_distance(before, mid)
        now = trimesh.proximity.signed_distance(after, mid)
    except Exception:  # noqa: BLE001
        return specs
    keep = [q for q, a, b in zip(specs, was, now) if not (a < 0 < b)]
    if len(keep) < len(specs):
        print(f"    «отверстий» на изломе достроенного скола отброшено: {len(specs) - len(keep)}", flush=True)
    return keep


def main() -> None:
    from cad_checks import measured

    started = time.perf_counter()
    timings = {}
    _memory_guard()
    scan_path, answer = Path(sys.argv[1]), Path(sys.argv[4])
    scan = trimesh.load(scan_path, force="mesh")
    KIND = sys.argv[5] if len(sys.argv) > 5 else "geometric"
    if KIND in ("walls", "geometric") and len(scan.faces) > 300_000:
        # стенкам и сечениям хватает 300 тыс. треугольников (сдвиг поверхности сотые доли мм), а на
        # миллионе они не успевают за отведённое время
        from holes import _coarse

        scan = _coarse(scan)
    sheet = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    digest = None
    if KIND in ("surface", "smooth") and any(
            os.environ.get(key, "1") != "0" for key in ("CAD_SURFACE_CACHE", "CAD_HOLE_CACHE")):
        digest = _scan_digest(scan_path)

    def hole_finder(mesh):
        return _not_in_fill(_hole_specs(mesh, scan_path, answer, digest=digest))

    if len(sys.argv) > 5 and sys.argv[5] == "walls":
        import walls

        import pickle

        ws = walls.walls(scan)
        code = walls.draft_code(ws)
        # те же стенки отдаст инструмент walls для ИИ, чтобы не считать их второй раз
        Path(sys.argv[4]).with_suffix(".pkl").write_bytes(pickle.dumps(ws))
        meta = {"plan": f"черновик из стенок: найдено {len(ws)} стенок, каждая выдавлена по контуру",
                "restored": [], "assumptions": ["деталь тонкостенная: собрана из пластин"]}
    elif len(sys.argv) > 5 and sys.argv[5] == "surface":
        import cadquery as cq

        import surface

        t = time.perf_counter()
        clean = _clean_surface(scan, scan_path, answer, digest=digest)
        timings["cleanup"] = time.perf_counter() - t
        clean, restored, zones = _restore(clean, timings, scan, digest, answer)
        body, info = surface.surface_body(scan, prepared=clean, timings=timings, hole_finder=hole_finder)
        step = Path(sys.argv[4]).with_suffix(".step")
        with measured(timings, "step_write"):
            cq.exporters.export(body, str(step))
        code = f"import cadquery as cq\nresult = cq.importers.importStep({str(step)!r}).val()\n"
        meta = {"plan": f"тело по поверхности скана: {info['faces']} треугольников, отверстий вырезано точно {info['holes_cut']}",
                "restored": restored, "restored_zones": zones,
                "assumptions": ["форма любая: тело повторяет поверхность скана, плоские участки выпрямлены"]}
    elif len(sys.argv) > 5 and sys.argv[5] == "smooth":
        import cadquery as cq

        import nurbs
        import surface
        from cad_checks import VolumeMemo

        volumes = VolumeMemo()

        # Приёмка гладкого тела меряется по очищенной поверхности, а не по сырому скану: шум там уже снят,
        # и допуск сравнения (3×шум) здесь слишком широк — пропускает крупные грани, срезающие рёбра.
        # Поэтому берётся сам шум, не меньше 0,3 мм (CAD_FIT_TOL=wide — прежнее правило, для сравнения).
        noise = float(sheet.get("diagnostics", {}).get("noise_mm", 0) or 0)
        wide = __import__("os").environ.get("CAD_FIT_TOL") == "wide"
        tol = max(0.3, (3 if wide else 1) * noise)
        t_smooth = time.perf_counter()
        clean = _clean_surface(scan, scan_path, answer, digest=digest)
        t = time.perf_counter()
        timings["cleanup"] = t - t_smooth
        clean, restored, zones = _restore(clean, timings, scan, digest, answer)
        t = time.perf_counter()
        plugged = None
        if os.environ.get("CAD_HOLE_PLUG", "0") == "1":   # отверстия закрыть в сетке, вырезать точно потом
            try:
                found_all = hole_finder(scan)
                clean, done_plug = surface.plug_holes(clean, found_all)
                timings["hole_plugged"] = len(done_plug)
                if done_plug:   # закрытые режем первыми: у них простая грань поперёк цилиндра
                    plugged = done_plug + [q for q in found_all if q not in done_plug]
                    print(f"    отверстий закрыто в сетке перед гладким телом: {len(done_plug)} из {len(found_all)}", flush=True)
            except Exception:  # noqa: BLE001
                plugged = None
        fit_mesh = surface.flatten_planes(clean)
        body, info = nurbs.smooth_body(fit_mesh, scan, tol=tol)
        timings["smooth"] = time.perf_counter() - t
        timings["nurbs_steps"] = info["stages"]
        if body is None:
            raise SystemExit("гладкое тело не собралось")
        fil_note = []
        t = time.perf_counter()
        try:  # скругления между плоскими гранями: точные цилиндры вместо сплайнов; не вышло — тело прежнее
            import fillets

            left = 1000 - (time.perf_counter() - t_smooth)   # у попытки 0к в demo.py предел 1200 с
            on = __import__("os").environ.get("CAD_FILLETS", "1") != "0"   # CAD_FILLETS=0 — без точных скруглений (для сравнения)
            fl = fillets.find_fillets(clean) if on and left > 120 else []
            if fl:
                body, built, _ = fillets.apply(
                    body, fl, budget=min(240.0, left - 90), mesh=clean, volumes=volumes, timings=timings,
                    limit_time=True)
                fil_note = [fillets.summary(fl) + (f"; точными цилиндрами построено кусков: {built}" if built else "")]
        except Exception:  # noqa: BLE001
            fil_note = []
        timings["fillets"] = time.perf_counter() - t
        t = time.perf_counter()
        analytic = None
        specs = None
        if os.environ.get("CAD_ANALYTIC_BORES", "0") == "1":
            with measured(timings, "holes_analytic"):
                try:
                    from analytic_bores import try_analytic_bores

                    specs = hole_finder(scan)
                    body, analytic = try_analytic_bores(body, scan, specs, tol=0.3)
                except Exception as error:  # an inconclusive trial keeps the existing route
                    analytic = {"status": "rollback", "reason": f"{type(error).__name__}: {error}"}
        local_bores = None
        if analytic and analytic["status"] == "accepted":
            cut = analytic["bores"]
        elif plugged is None and os.environ.get("CAD_BORE_STITCH", "0") == "1":
            with measured(timings, "holes_stitch"):
                try:
                    from bore_stitch import repair_bores

                    if specs is None:
                        specs = hole_finder(scan)
                    body, local_bores = repair_bores(body, scan, specs, clean=fit_mesh, noise_mm=noise)
                except Exception as error:
                    local_bores = {"accepted": 0, "reason": f"{type(error).__name__}: {error}"}
            cut = local_bores["accepted"]
        else:
            cut = 0
        if not cut:
            found = specs if specs is not None else plugged
            finder = hole_finder if found is None else lambda _: found
            body, cut = surface.cut_holes(body, scan, volumes=volumes, timings=timings, hole_finder=finder,
                                           limit_time=True)
        timings["holes"] = time.perf_counter() - t
        step = Path(sys.argv[4]).with_suffix(".step")
        with measured(timings, "step_write"):
            cq.exporters.export(body, str(step))
        code = f"import cadquery as cq\nresult = cq.importers.importStep({str(step)!r}).val()\n"
        kinds = __import__("collections").Counter(f.geomType() for f in body.Faces())
        exact = ", ".join(f"{name} {kinds[t]}" for t, name in (("PLANE", "плоскостей"), ("CYLINDER", "цилиндров"),
                                                              ("CONE", "конусов"), ("SPHERE", "сфер")) if kinds[t])
        meta = {"plan": f"гладкое тело: {sum(kinds.values())} граней по скану (точных: {exact or 'нет'}; остальные B-сплайн), {round(info['near'] * 100, 1)}% поверхности "
                        f"в пределах 0,2 мм, отверстий построено точно {cut}",
                "restored": restored, "restored_zones": zones,
                "assumptions": ["форма любая: гладкие грани повторяют поверхность скана"] + fil_note}
        if analytic:
            meta["analytic_bores"] = analytic
        if local_bores is not None:
            meta["bore_stitch"] = local_bores
    else:
        code, meta = geometric.build(scan, sheet, restore=sys.argv[3] == "1")
    timings["total"] = time.perf_counter() - started
    answer.write_text(json.dumps({"code": code, "meta": meta, "timings": timings}, ensure_ascii=False),
                      encoding="utf-8")


if __name__ == "__main__":
    main()
