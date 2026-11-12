"""Local CAD workspace. Uses the existing geometric route and its STEP artifacts."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
from urllib.parse import unquote, urlsplit
from uuid import uuid4
import webbrowser
import zipfile

HERE = Path(__file__).resolve().parent
DEMO = HERE.parent                       # окно лежит в папке демо (claude/cad-demo/workspace)
AUDIT = HERE.parents[2] / "chatgpt-1/cad-intelligence/cad-acceptance-2026-11-12"   # проверка STEP Севера
sys.path.insert(0, str(AUDIT))
import geometry_acceptance as audit
import numpy as np
import trimesh

SAMPLE = DEMO / "samples/flange_scan.stl"
TRUTH = DEMO / "samples/flange_truth.step"
UNITS = {"mm": 1., "cm": 10., "m": 1000., "in": 25.4}
ASSETS = {"scan": "scan.stl", "model": "model.stl", "step": "model.step",
          "deviations": "deviations.bin", "model_deviation": "model-deviation.bin",
          "unobserved": "unobserved.bin", "macro": "model_sw.bas", "report": "inspection.json"}
STATIC = {"": "index.html", "workspace.css": "workspace.css",
          "workspace.js": "workspace.js", "viewport.js": "viewport.js"}
VENDOR = {"three.module.min.js", "OrbitControls.js", "STLLoader.js"}
MAX_UPLOAD = 300 * 1024**2
LIMIT = 2100
STAGES = (
    (re.compile(r"\b1/4\b"), "Анализ скана"),
    (re.compile(r"\b2/4\b"), "Реконструкция"),
    (re.compile(r"\b3/4\b"), "Построение модели"),
    (re.compile(r"\b4/4\b"), "Проверка результата"),
)


def load_mesh(path, *, process=True):
    mesh = trimesh.load(path, force="mesh", process=process)
    if not len(mesh.faces) or not np.isfinite(mesh.vertices).all() or mesh.area <= 0:
        raise ValueError("В файле нет пригодной треугольной сетки.")
    return mesh


def json_read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def mesh_info(mesh):
    return {"triangles": len(mesh.faces), "extents": mesh.extents.tolist()}


def engine_revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=DEMO,
                                       text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        return None


AI_BUDGET_USD = 0.3   # предел расхода на один расчёт с ИИ: прогон останавливается, если оценка выше


def ai_key():
    """Ключ ИИ задаётся в окружении перед запуском окна (set ANTHROPIC_API_KEY=...)."""
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"))


def run_identity(name, revision, regions, ai=False):
    mode = "с ИИ" if ai else "без ИИ"
    return {"name": f"{name} · {revision[:7] if revision else 'версия неизвестна'} · {mode}",
            "engine_revision": revision, "cad_regions": regions, "ai": ai}


def update_progress(job, line):
    message = re.sub(r"^\d\d:\d\d:\d\d\s+", "", line).strip()
    for pattern, stage in STAGES:
        if pattern.match(message):
            job.update(stage=stage, stage_detail=message[3:].strip().rstrip("…"), heartbeat=None)
            attempt = re.match(r"2/4\s+попытка\s+(\w+):", message)
            if attempt:
                job["attempt"] = attempt[1]
            elif not message.startswith(("3/4 строю модель", "4/4 сравниваю со сканом")):
                job["attempt"] = None
            return
    heartbeat = re.match(r"…\s*идёт\s+(\d+)\s+мин(?:\s+из предела\s+(\d+)\s+мин)?", message)
    if heartbeat:
        job["heartbeat"] = {"minutes": int(heartbeat[1]), "limit_minutes": int(heartbeat[2]) if heartbeat[2] else None}
    attempt = job.get("attempt")
    if attempt and message.startswith("в допуске ") and job.get("stage") == "Проверка результата":
        job["compared"] = list(dict.fromkeys([*job.get("compared", []), attempt]))
    if attempt and message.startswith("обязательная проверка не пройдена:"):
        job["rejected"] = list(dict.fromkeys([*job.get("rejected", []), attempt]))
    winner = re.match(r"Выбран путь\s+(\w+)\b", message)
    if winner:
        job["winner"] = winner[1]


def display_distances(scan, points, ceiling):
    """Exact distance to scan triangles up to the top of the color scale."""
    tree = scan.triangles_tree
    triangles = scan.triangles
    result = np.full(len(points), ceiling, dtype="<f4")
    for start in range(0, len(points), 512):
        query = points[start:start + 512]
        low, high = query - ceiling, query + ceiling
        try:
            indices, counts = tree.intersection_v(low, high)
            owners = np.repeat(np.arange(len(query)), counts.astype(np.intp))
        except Exception:
            # Older Rtree builds may lack the vectorized query.
            candidates = [list(tree.intersection(bounds)) for bounds in np.column_stack((low, high))]
            indices = np.concatenate(candidates).astype(np.intp)
            owners = np.repeat(np.arange(len(query)), list(map(len, candidates)))
        if len(indices):
            p = query[owners]
            near = trimesh.triangles.closest_point(triangles[indices], p)
            np.minimum.at(result[start:start + len(query)], owners, np.linalg.norm(near - p, axis=1))
    return result


def display_model(directory, model, edge=2.0):
    """Сетка модели для просмотра: длинные треугольники делятся (не длиннее ~2 мм или сотой габарита), иначе
    карта отклонений на крупных гранях идёт полосами — цвет берётся по центру треугольника. На STEP
    это не влияет, файл модели для скачивания — STEP."""
    edge = max(edge, float(model.extents.max()) / 100)   # большая деталь: шаг сотая габарита
    try:
        v, f = trimesh.remesh.subdivide_to_size(model.vertices, model.faces, max_edge=edge, max_iter=8)
        if len(f) <= 2_000_000:
            model = trimesh.Trimesh(v, f, process=False)
    except Exception:  # noqa: BLE001 — не вышло делить: показываем как есть
        pass
    model.export(directory / "model.stl")
    return model


def prepare_inspection(directory, scan, model, step, example=False, transform=None, identity=None):
    tol = audit.Tolerance(samples=4000)
    result = audit.evaluate(step, TRUTH if example else None, transform=transform, tol=tol)
    scan_fit = audit.distances(scan, model, tol)
    p, _ = trimesh.sample.sample_surface(scan, tol.samples, seed=tol.seed)
    distance = np.concatenate([trimesh.proximity.closest_point(model, p[i:i + 96])[1]
                               for i in range(0, len(p), 96)])
    np.asarray(np.c_[p, distance], dtype="<f4").tofile(directory / ASSETS["deviations"])
    # STL triangle order is also the viewer's order. Values above the legend
    # ceiling use the same color, so no search outside that radius is needed.
    display_distances(scan, model.triangles_center, 2 * tol.cover_mm).tofile(
        directory / ASSETS["model_deviation"])
    (directory / ASSETS["unobserved"]).unlink(missing_ok=True)
    issues = []
    dimensions = {
        "offset_mm": ("Смещение плоскости", "plane_mm", "мм"),
        "sampled_p95_mm": ("Отклонение плоскости", "plane_mm", "мм"),
        "angle_deg": ("Угол плоскости", "plane_angle_deg", "°"),
        "radius_mm": ("Отклонение радиуса", "radius_mm", "мм"),
        "axis_offset_mm": ("Смещение оси", "axis_offset_mm", "мм"),
        "axis_angle_deg": ("Угол оси", "axis_angle_deg", "°"),
        "end_mm": ("Отклонение осевой границы", "end_mm", "мм"),
        "semi_angle_deg": ("Полуугол конуса", "cone_angle_deg", "°"),
        "radius_at_station_mm": ("Радиус в опорном сечении", "radius_mm", "мм"),
        "center_mm": ("Смещение центра", "axis_offset_mm", "мм"),
        "minor_radius_mm": ("Малый радиус тора", "radius_mm", "мм"),
    }
    roles = {"bore": "отверстие", "plane": "плоскость", "cylinder": "цилиндр",
             "cone": "конус", "sphere": "сфера", "torus": "тор",
             "cylindrical_fillet": "цилиндрическое скругление",
             "partial_inner_cylinder": "частичная стенка отверстия"}
    for row in result["features"] or ():
        status = row["status"]
        if status == "ok":
            continue
        feature = row["model"] or row["reference"]
        name = roles.get(feature["role"], "геометрический элемент")
        issue = {"label": {"extra": "Дополнительная грань", "missing": "Пропущенный элемент",
                           "parameter_error": "Параметры вне допуска"}[status] + " — " + name,
                 "source": "Эталонный STEP", "point": feature["center"], "severity": "warning"}
        key = next((k for k in row.get("failed", ()) if k in dimensions), None)
        if key:
            label, limit, unit = dimensions[key]
            issue.update(label=label, value=row["errors"][key], limit=getattr(tol, limit), unit=unit)
        issues.append(issue)
    top = result["model_topology"]
    if not all(top[k] for k in ("valid", "closed", "connected_solid")):
        issues.append({"label": "Проверка тела требует внимания", "severity": "warning",
                       "source": "Повторное чтение STEP"})
    audit.write_json(directory / ASSETS["report"],
                     {**result, "scan_surface": scan_fit, "run": identity})
    return {"topology": top, "scan_surface": scan_fit, "reference_surface": result["surface"],
            "reference_outcomes": result["outcomes"], "tolerance_mm": tol.cover_mm,
            "issues": issues, "has_reference": example}


class Workspace:
    def __init__(self, data):
        self.data = data
        data.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.jobs = {}
        self.active = None
        self.process = None
        self.load_jobs()
        self.example()

    def path(self, id):
        return self.data / id

    def save(self, job):
        directory = self.path(job["id"])
        staging = directory / "state.tmp"
        audit.write_json(staging, job)
        staging.replace(directory / "state.json")

    def load_jobs(self):
        for path in self.data.glob("*/state.json"):
            if not re.fullmatch(r"[0-9a-f]{12}", path.parent.name):
                continue
            try:
                job = json_read(path)
                if job["id"] != path.parent.name:
                    continue
                if job["status"] == "running":
                    job.update(status="error", error="Предыдущий расчёт прервался при закрытии приложения.")
                    self.save(job)
                self.jobs[job["id"]] = job
            except (OSError, ValueError, KeyError):
                continue

    def example(self):
        directory = self.path("example")
        directory.mkdir(exist_ok=True)
        with zipfile.ZipFile(HERE / "example.zip") as archive:
            for filename in ("model.stl", "model.step", "example.json", ASSETS["macro"]):
                (directory / filename).write_bytes(archive.read(filename))
        example = json_read(directory / "example.json")
        scan = load_mesh(SAMPLE)
        scan.apply_transform(example["transform"])
        scan.export(directory / "scan.stl")
        model = display_model(directory, load_mesh(directory / "model.stl", process=False))
        identity = run_identity("Тестовый фланец", example["provenance"]["commit"], False)
        check = prepare_inspection(directory, scan, model, directory / "model.step",
                                   example=True, transform=example["transform"], identity=identity)
        self.jobs["example"] = {"id": "example", "name": "Тестовый фланец", "filename": SAMPLE.name,
                                "status": "ready", "example": True, "sample": True, "units": "mm",
                                "created": 0, "stage": "Результат сохранён", "seconds": None,
                                "scan": mesh_info(scan), "check": check,
                                "provenance": example["provenance"], "identity": identity,
                                "compared": ["0", "0ж"], "winner": "0ж", "log": []}

    def summary(self):
        with self.lock:
            return [{"id": j["id"], "name": j["name"], "status": j["status"], "created": j["created"],
                     "identity": j.get("identity")}
                    for j in sorted(self.jobs.values(), key=lambda j: j["created"], reverse=True)]

    def snapshot(self, id):
        with self.lock:
            job = self.jobs.get(id)
            if job is None:
                return None
            result = dict(job, log=list(job["log"]))
            if job["status"] == "running":
                result["seconds"] = time.time() - job["started"]
            result["assets"] = {name: f"/api/jobs/{id}/asset/{name}" for name, file in ASSETS.items()
                                if (self.path(id) / file).is_file()
                                and (name == "scan" or job["status"] == "ready")}
            return result

    def create(self, original, units, stream=None, count=0):
        id = uuid4().hex[:12]
        directory = self.path(id)
        directory.mkdir()
        suffix = Path(original).suffix.lower()
        source = directory / f"input{suffix}"
        try:
            if stream is None:
                shutil.copyfile(SAMPLE, source)
            else:
                with source.open("wb") as output:
                    while count:
                        block = stream.read(min(count, 1024**2))
                        if not block:
                            raise ValueError("Загрузка файла прервалась.")
                        output.write(block)
                        count -= len(block)
            try:
                scan = load_mesh(source)
            except Exception as exc:
                raise ValueError("Не удалось прочитать сетку. Проверьте формат и целостность файла.") from exc
            info = mesh_info(scan)
            scan.export(directory / "scan.stl")
        except Exception:
            source.unlink(missing_ok=True)
            (directory / "scan.stl").unlink(missing_ok=True)
            directory.rmdir()
            raise
        job = {"id": id, "name": "Тестовый фланец" if stream is None else Path(original).stem,
               "filename": original, "sample": stream is None, "example": False,
               "status": "draft", "stage": "Скан загружен", "created": time.time(),
               "units": units, "seconds": None, "scan": info, "check": None, "log": []}
        with self.lock:
            self.save(job)
            self.jobs[id] = job
        return id

    def start(self, id, units, regions=False, notes="", ai=False):
        with self.lock:
            if self.active:
                raise ValueError("Уже идёт расчёт. Дождитесь его завершения.")
            old = self.jobs[id]
            if old["status"] != "draft":
                if old["sample"]:
                    id = self.create(SAMPLE.name, "mm")
                else:
                    original = self.path(id) / ("input" + Path(old["filename"]).suffix.lower())
                    with original.open("rb") as stream:
                        id = self.create(old["filename"], units, stream, original.stat().st_size)
            job = self.jobs[id]
            job.update(status="running", units="mm" if job["sample"] else units, started=time.time(),
                       stage="Запуск расчёта", log=[], compared=[], rejected=[],
                       notes=notes, identity=run_identity(job["name"], engine_revision(), regions, ai))
            self.save(job)
            self.active = id
            threading.Thread(target=self.work, args=(job,), daemon=True).start()
            return id

    def work(self, job):
        id = job["id"]
        directory = self.path(id)
        source = directory / ("input" + Path(job["filename"]).suffix.lower())
        out = directory / "engine"
        ai = bool(job["identity"].get("ai")) and ai_key()
        # без ИИ ключи в расчёт не передаются вовсе: так нельзя случайно потратить деньги
        env = {k: v for k, v in os.environ.items()
               if ai or k.upper() not in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")}
        env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
                   CAD_REGIONS="1" if job["identity"]["cad_regions"] else "0")
        cmd = [sys.executable, "-u", str(DEMO / "demo.py"), str(source), "--mode", "auto" if ai else "geometric",
               "--units", job["units"], "--out", str(out), "--no-open"]
        if job.get("notes"):
            cmd += ["--notes", job["notes"]]
        if ai:
            cmd += ["--budget-usd", str(AI_BUDGET_USD)]
        process = None
        timer = None
        try:
            process = subprocess.Popen(cmd, cwd=DEMO, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            with self.lock:
                self.process = process
            timer = threading.Timer(LIMIT, stop_tree, args=(process,))
            timer.daemon = True
            timer.start()
            with (directory / "engine.log").open("w", encoding="utf-8") as log:
                for line in process.stdout:
                    line = line.rstrip()
                    log.write(line + "\n")
                    with self.lock:
                        job["log"].append(line)
                        del job["log"][:-160]
                        update_progress(job, line)
            if process.wait() != 0:
                raise RuntimeError("Расчёт завершился с ошибкой. Подробности в журнале.")
            roots = list(out.glob("*/model.step"))
            if len(roots) != 1:
                raise RuntimeError("Конвейер не выдал итоговый STEP.")
            run = roots[0].parent
            with self.lock:
                job.update(stage="Независимая проверка STEP", stage_detail=None, heartbeat=None)
            for name, filename in (("scan_aligned.stl", "scan.stl"), ("model.stl", "model.stl"),
                                   ("model.step", "model.step")):
                temporary = directory / (filename + ".tmp")
                shutil.copyfile(run / name, temporary)
                temporary.replace(directory / filename)
            macro = run / ASSETS["macro"]
            if macro.is_file():
                shutil.copyfile(macro, directory / ASSETS["macro"])
            scan = load_mesh(directory / "scan.stl")
            model = display_model(directory, load_mesh(directory / "model.stl", process=False))
            transform = json_read(run / "sheet.json")["transform_from_scan_4x4"]
            try:
                check = prepare_inspection(directory, scan, model, directory / "model.step",
                                           example=job["sample"], transform=transform, identity=job["identity"])
                inspection_error = None
            except Exception as exc:
                check = None
                inspection_error = "STEP записан. Проверка прервалась — смотри журнал."
                with self.lock:
                    job["log"].append(f"Проверка STEP: {type(exc).__name__}: {exc}")
            outcome = dict(status="ready", check=check, stage="Расчёт завершён",
                           inspection_error=inspection_error, aligned=True, scan=mesh_info(scan))
        except Exception as exc:
            outcome = dict(status="error", error=str(exc), stage="Расчёт остановлен")
        finally:
            if timer:
                timer.cancel()
            if process and process.poll() is None:
                stop_tree(process)
            with self.lock:
                job.update(outcome, seconds=time.time() - job["started"], stage_detail=None, heartbeat=None)
                self.active = None
                self.process = None
                self.save(job)

    def close(self):
        with self.lock:
            process = self.process
        if process is not None:
            stop_tree(process)


def stop_tree(process):
    if process.poll() is not None:
        return
    import psutil
    try:
        children = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    for child in reversed(children):
        try:
            child.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    try:
        process.kill()
    except ProcessLookupError:
        pass


class Handler(BaseHTTPRequestHandler):
    def send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path, name=None):
        if not path.is_file():
            return self.send_json({"error": "Файл ещё недоступен."}, 404)
        with path.open("rb") as stream:
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(path)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(os.fstat(stream.fileno()).st_size))
            self.send_header("Cache-Control", "no-store")
            if name:
                self.send_header("Content-Disposition", f'attachment; filename="{name}"')
            self.end_headers()
            shutil.copyfileobj(stream, self.wfile)

    def allowed(self):
        host = self.headers.get("Host")
        expected = f"127.0.0.1:{self.server.server_port}"
        return host == expected and self.headers.get("Origin") in (None, f"http://{expected}")

    def do_GET(self):
        if not self.allowed():
            return self.send_json({"error": "Откройте локальный адрес приложения."}, 403)
        parts = urlsplit(self.path).path.strip("/").split("/")
        if len(parts) == 1 and parts[0] in STATIC:
            return self.send_file(HERE / STATIC[parts[0]])
        if len(parts) == 2 and parts[0] == "vendor" and parts[1] in VENDOR:
            return self.send_file(DEMO / "vendor" / parts[1])
        if parts == ["favicon.ico"]:
            self.send_response(204)
            self.end_headers()
            return
        if parts == ["api", "jobs"]:
            return self.send_json(self.server.workspace.summary())
        if parts == ["api", "config"]:
            return self.send_json({"ai": ai_key(), "ai_budget_usd": AI_BUDGET_USD})
        if parts[:2] == ["api", "jobs"] and len(parts) in (3, 5):
            id = parts[2]
            job = self.server.workspace.snapshot(id)
            if job and len(parts) == 3:
                return self.send_json(job)
            if job and parts[3] == "asset" and parts[4] in job["assets"]:
                key = parts[4]
                downloads = {"step": f"{'flange' if job['sample'] else 'reconstruction'}-{id}.step",
                             "report": f"inspection-{id}.json", "macro": f"solidworks-{id}.bas"}
                name = downloads.get(key)
                return self.send_file(self.server.workspace.path(id) / ASSETS[key], name)
        return self.send_json({"error": "Адрес не найден."}, 404)

    def do_POST(self):
        if not self.allowed() or self.headers.get("X-CAD-Workspace") != "1":
            return self.send_json({"error": "Недопустимый источник запроса."}, 403)
        units = self.headers.get("X-Units", "mm")
        if units not in UNITS:
            return self.send_json({"error": "Выберите единицы скана."}, 400)
        try:
            if self.path == "/api/jobs":
                count = int(self.headers.get("Content-Length", "0"))
                name = Path(unquote(self.headers.get("X-Filename", ""))).name
                if not 0 < count <= MAX_UPLOAD or Path(name).suffix.lower() not in (".stl", ".obj", ".ply"):
                    return self.send_json({"error": "Выберите STL, OBJ или PLY до 300 МБ."}, 400)
                id = self.server.workspace.create(name, units, self.rfile, count)
            else:
                match = re.fullmatch(r"/api/jobs/(example|[a-f0-9]{12})/run", self.path)
                if not match:
                    return self.send_json({"error": "Адрес не найден."}, 404)
                regions = self.headers.get("X-CAD-Regions", "0")
                if regions not in ("0", "1"):
                    return self.send_json({"error": "Недопустимый режим разметки."}, 400)
                notes = unquote(self.headers.get("X-Notes", ""))[:2000]
                ai = self.headers.get("X-AI", "0") == "1"
                if ai and not ai_key():
                    return self.send_json({"error": "Ключ ИИ не задан: расчёт с ИИ недоступен."}, 400)
                id = self.server.workspace.start(match[1], units, regions == "1", notes, ai)
            return self.send_json(self.server.workspace.snapshot(id), 201)
        except KeyError:
            return self.send_json({"error": "Расчёт не найден."}, 404)
        except (ValueError, OSError, RuntimeError) as exc:
            return self.send_json({"error": str(exc)}, 400)


def main():
    default = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local/share")) / "FrogFlow" / "CAD Workspace"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--data", type=Path, default=default)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        server.workspace = Workspace(args.data.resolve())
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"CAD Intelligence — {url}", flush=True)
        if not args.no_open:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.workspace.close()


if __name__ == "__main__":
    main()
