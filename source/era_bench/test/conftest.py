"""Shared headless Isaac Sim bootstrap for the ERA Bench test suite."""

from isaacsim import SimulationApp


simulation_app = SimulationApp({"headless": True})
