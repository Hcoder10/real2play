"""Real-arm control. Importing this package never connects to hardware."""

__all__ = ["RealRobot", "ReadOnlyBus", "MockRobot", "GravityComp", "GravitySag", "Guard", "Session", "Operator"]


def __getattr__(name):
    from importlib import import_module
    modules = {"RealRobot": "io", "ReadOnlyBus": "io", "MockRobot": "io",
               "GravityComp": "gravity", "GravitySag": "gravity", "Guard": "guard",
               "Session": "runner", "Operator": "operator"}
    if name not in modules:
        raise AttributeError(name)
    return getattr(import_module(f"r2p.robot.{modules[name]}"), name)
