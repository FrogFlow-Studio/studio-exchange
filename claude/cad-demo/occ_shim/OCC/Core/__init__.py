import importlib
import importlib.abc
import importlib.util
import sys
import types

# pythonocc: модуль-помощник в нижнем регистре -> класс OCP с _s-методами
_LOWER = {"topods": "TopoDS", "brepgprop": "BRepGProp", "breplib": "BRepLib", "breptools": "BRepTools",
          "precision": "Precision", "topexp": "TopExp", "brepbndlib": "BRepBndLib", "bndlib": "BndLib",
          "brepbuilderapi": "BRepBuilderAPI", "shapeanalysis": "ShapeAnalysis"}


class _Static:
    """Обёртка класса OCP: Cls.Method -> Cls.Method_s, если обычного нет; вызов — конструктор."""

    def __init__(self, cls):
        object.__setattr__(self, "_cls", cls)

    def __getattr__(self, name):
        c = object.__getattribute__(self, "_cls")
        try:
            return getattr(c, name)
        except AttributeError:
            return getattr(c, name + "_s")

    def __call__(self, *a, **k):
        return object.__getattribute__(self, "_cls")(*a, **k)

    def __instancecheck__(self, obj):
        return isinstance(obj, object.__getattribute__(self, "_cls"))


class _Module(types.ModuleType):
    def __init__(self, name, ocp):
        super().__init__(name)
        self._ocp = ocp

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        ocp = self._ocp
        if name in _LOWER and hasattr(ocp, _LOWER[name]):
            return _Static(getattr(ocp, _LOWER[name]))
        obj = getattr(ocp, name)
        if isinstance(obj, type) and any(n.endswith("_s") for n in dir(obj)):
            return _Static(obj)
        return obj


class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith("OCC.Core."):
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        return _Module(spec.name, importlib.import_module("OCP." + spec.name.split(".", 2)[2]))

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, _Finder())
