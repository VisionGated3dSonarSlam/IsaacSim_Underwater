"""Runtime control of underwater visibility (turbidity) for the OceanSim UW camera.

Why this exists
---------------
OceanSim renders underwater appearance with a Jaffe-McGlamery style model, evaluated per
frame inside a Warp kernel:

    uw_rgb = raw_rgb * exp(-depth * atten_coeff)
             + backscatter_value * 255 * (1 - exp(-depth * backscatter_coeff))

`UW_Camera.initialize()` stores the three coefficients as plain `wp.vec3f` attributes and
`_process_underwater_frame()` passes them to the kernel on every launch. Nothing caches or
compiles them in between, so **reassigning those attributes changes the very next rendered
frame** -- no simulator restart required.

That makes a single continuous run with changing water clarity possible, which is the
experiment that actually justifies a visibility gate: a fixed always-on or always-off
policy cannot win in both halves of one run, but a gate can.

This module is deliberately standalone and additive. If it is never enabled, the
simulator behaves exactly as before.

Usage from the simulator script:

    controller = TurbidityController.from_config(uw_camera, config.get("turbidity"))
    ...
    controller.update(sim_time_s)   # once per frame, cheap

Control at runtime:

    ros2 topic pub --once /sim/turbidity std_msgs/msg/Float32 "{data: 0.7}"
"""

from __future__ import annotations

import bisect
from typing import Any, Optional, Sequence

import numpy as np

# Turbidity is expressed as a single scalar in [0, 1] rather than nine raw coefficients:
# 0 = clearest water the scene supports, 1 = the murkiest. One dimension keeps experiments
# describable ("turbidity 0.6") and gives the visibility gate a scalar ground truth to be
# validated against. The endpoints below are interpolated to recover the nine coefficients.
TURBIDITY_MIN = 0.0
TURBIDITY_MAX = 1.0

# Defaults chosen so 0.0 reproduces OceanSim's own clear-water defaults, and 1.0 is murky
# enough that a camera is genuinely unusable while sonar is unaffected.
DEFAULT_CLEAR = {
    "backscatter_value": [0.0, 0.31, 0.24],
    "backscatter_coeff": [0.05, 0.05, 0.20],
    "atten_coeff": [0.05, 0.05, 0.05],
}
DEFAULT_TURBID = {
    "backscatter_value": [0.10, 0.36, 0.30],
    "backscatter_coeff": [0.60, 0.60, 0.80],
    "atten_coeff": [0.80, 0.75, 0.70],
}


def _as_vec3(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (3,):
        raise ValueError(f"turbidity.{name} must have exactly 3 values, got {array.shape[0]}")
    return array


class TurbidityProfile:
    """Maps a scalar turbidity in [0, 1] onto the nine OceanSim coefficients.

    Interpolation is linear per channel between a clear and a turbid endpoint. Linear is
    the right choice here despite attenuation being physically exponential in *depth*: the
    coefficients themselves are what the model exponentiates, so a linear sweep of the
    coefficient already produces exponential change in the image.
    """

    def __init__(self, clear: dict | None = None, turbid: dict | None = None) -> None:
        clear = clear or DEFAULT_CLEAR
        turbid = turbid or DEFAULT_TURBID

        self.clear_backscatter_value = _as_vec3(
            clear.get("backscatter_value", DEFAULT_CLEAR["backscatter_value"]), "clear.backscatter_value")
        self.clear_backscatter_coeff = _as_vec3(
            clear.get("backscatter_coeff", DEFAULT_CLEAR["backscatter_coeff"]), "clear.backscatter_coeff")
        self.clear_atten_coeff = _as_vec3(
            clear.get("atten_coeff", DEFAULT_CLEAR["atten_coeff"]), "clear.atten_coeff")

        self.turbid_backscatter_value = _as_vec3(
            turbid.get("backscatter_value", DEFAULT_TURBID["backscatter_value"]), "turbid.backscatter_value")
        self.turbid_backscatter_coeff = _as_vec3(
            turbid.get("backscatter_coeff", DEFAULT_TURBID["backscatter_coeff"]), "turbid.backscatter_coeff")
        self.turbid_atten_coeff = _as_vec3(
            turbid.get("atten_coeff", DEFAULT_TURBID["atten_coeff"]), "turbid.atten_coeff")

    def coefficients(self, turbidity: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (backscatter_value, atten_coeff, backscatter_coeff) for a turbidity."""
        t = float(np.clip(turbidity, TURBIDITY_MIN, TURBIDITY_MAX))
        backscatter_value = (1.0 - t) * self.clear_backscatter_value + t * self.turbid_backscatter_value
        backscatter_coeff = (1.0 - t) * self.clear_backscatter_coeff + t * self.turbid_backscatter_coeff
        atten_coeff = (1.0 - t) * self.clear_atten_coeff + t * self.turbid_atten_coeff
        return backscatter_value, atten_coeff, backscatter_coeff


class TurbiditySchedule:
    """A time-indexed turbidity profile: [(t0, v0), (t1, v1), ...].

    Values are held before the first keyframe and after the last. Between keyframes the
    behaviour depends on `interpolate`:
      * True  -- ramp linearly, modelling drifting into a silty region
      * False -- step at each keyframe, giving an unambiguous before/after boundary

    A schedule makes a run reproducible: every arm of a comparison sees an identical
    visibility timeline, so differences come from the estimator rather than from when the
    water happened to cloud over.
    """

    def __init__(self, keyframes: Sequence[Sequence[float]], interpolate: bool = True) -> None:
        parsed = []
        for entry in keyframes:
            if len(entry) != 2:
                raise ValueError("each turbidity keyframe must be [time_s, turbidity]")
            parsed.append((float(entry[0]), float(np.clip(entry[1], TURBIDITY_MIN, TURBIDITY_MAX))))

        parsed.sort(key=lambda item: item[0])
        if not parsed:
            raise ValueError("turbidity schedule must contain at least one keyframe")

        self._times = [item[0] for item in parsed]
        self._values = [item[1] for item in parsed]
        self.interpolate = bool(interpolate)

    @property
    def end_time(self) -> float:
        return self._times[-1]

    def value_at(self, elapsed_s: float) -> float:
        if elapsed_s <= self._times[0]:
            return self._values[0]
        if elapsed_s >= self._times[-1]:
            return self._values[-1]

        index = bisect.bisect_right(self._times, elapsed_s) - 1
        if not self.interpolate:
            return self._values[index]

        t0, t1 = self._times[index], self._times[index + 1]
        v0, v1 = self._values[index], self._values[index + 1]
        span = t1 - t0
        if span <= 0.0:
            return v1
        ratio = (elapsed_s - t0) / span
        return v0 + ratio * (v1 - v0)


class TurbidityController:
    """Applies turbidity to a live UW_Camera, from a schedule and/or a ROS 2 topic.

    Publishes the current value on `turbidity_state_topic` so recorded runs carry a
    ground-truth visibility signal. That signal is what lets the visibility gate be
    validated -- without it, a gate that fires at the right moments cannot be
    distinguished from one that fires by luck.
    """

    def __init__(
        self,
        uw_camera: Any,
        profile: Optional[TurbidityProfile] = None,
        schedule: Optional[TurbiditySchedule] = None,
        initial_turbidity: float = 0.0,
        command_topic: str = "/sim/turbidity",
        state_topic: str = "/sim/turbidity_state",
        enable_ros: bool = True,
        publish_period_s: float = 1.0,
        verbose: bool = True,
    ) -> None:
        self._camera = uw_camera
        self._profile = profile or TurbidityProfile()
        self._schedule = schedule
        self._verbose = verbose
        self._publish_period_s = max(0.0, float(publish_period_s))

        self._turbidity = float(np.clip(initial_turbidity, TURBIDITY_MIN, TURBIDITY_MAX))
        self._applied_turbidity: Optional[float] = None
        self._start_time: Optional[float] = None
        self._last_publish_time = -1.0
        self._override: Optional[float] = None

        self._node = None
        self._publisher = None
        self._executor = None
        self._owns_ros_context = False
        if enable_ros:
            self._setup_ros(command_topic, state_topic)

        # Apply immediately so frame zero already reflects the requested visibility.
        self._apply(self._turbidity, force=True)

    # -- construction ------------------------------------------------------------

    @classmethod
    def from_config(cls, uw_camera: Any, config: Optional[dict]) -> Optional["TurbidityController"]:
        """Build from the `turbidity` block of sim_params.json.

        Returns None when the block is absent or disabled, so the simulator runs exactly
        as it did before this module existed.
        """
        if uw_camera is None or not isinstance(config, dict) or not config.get("enabled", False):
            return None

        profile = TurbidityProfile(config.get("clear"), config.get("turbid"))

        schedule = None
        schedule_cfg = config.get("schedule")
        if isinstance(schedule_cfg, dict) and schedule_cfg.get("enabled", False):
            keyframes = schedule_cfg.get("keyframes", [])
            if keyframes:
                schedule = TurbiditySchedule(
                    keyframes, interpolate=schedule_cfg.get("interpolate", True))

        return cls(
            uw_camera,
            profile=profile,
            schedule=schedule,
            initial_turbidity=float(config.get("initial_turbidity", 0.0)),
            command_topic=str(config.get("command_topic", "/sim/turbidity")),
            state_topic=str(config.get("state_topic", "/sim/turbidity_state")),
            # Defaults to False: see _setup_ros() for why a second rclpy context inside
            # the simulator process is risky. Schedules need no ROS at all.
            enable_ros=bool(config.get("ros_control", False)),
            publish_period_s=float(config.get("publish_period_s", 1.0)),
            verbose=bool(config.get("verbose", True)),
        )

    # -- ROS ---------------------------------------------------------------------

    def _setup_ros(self, command_topic: str, state_topic: str) -> None:
        """Bring up an rclpy node for live control.

        Off by default (`ros_control: false`). The simulator publishes everything else
        through OmniGraph `isaacsim.ros2.bridge` nodes and never imports rclpy itself, so
        calling rclpy.init() here creates a *second* ROS context inside the same process.
        That is at best redundant and at worst destabilises the bridge, so live control is
        opt-in and a schedule is the recommended way to drive a run.
        """
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from std_msgs.msg import Float32

            # Only initialise a context if nothing else already did. Tearing down or
            # re-initialising a context the bridge owns would take the publishers with it.
            self._owns_ros_context = not rclpy.ok()
            if self._owns_ros_context:
                rclpy.init(args=None)

            self._node = rclpy.create_node("turbidity_controller")
            self._node.create_subscription(Float32, command_topic, self._on_command, 10)
            self._publisher = self._node.create_publisher(Float32, state_topic, 10)

            # A dedicated executor spun manually from update(): the simulator owns the
            # main loop, so there is no thread available to block in rclpy.spin().
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self._node)

            if self._verbose:
                print(f"[turbidity] listening on {command_topic}, publishing {state_topic}")
        except Exception as exc:  # pragma: no cover - depends on the sim environment
            # ROS control is a convenience; a schedule-driven run must not die without it.
            print(f"[turbidity] ROS control unavailable ({exc}); schedule-only mode")
            self._node = None
            self._publisher = None
            self._executor = None

    def _on_command(self, msg: Any) -> None:
        value = float(np.clip(msg.data, TURBIDITY_MIN, TURBIDITY_MAX))
        # A manual command overrides the schedule for the rest of the run: mixing the two
        # silently would make a recorded visibility timeline impossible to interpret.
        self._override = value
        if self._verbose and self._schedule is not None:
            print(f"[turbidity] manual command {value:.3f} overrides the schedule")

    # -- per-frame ---------------------------------------------------------------

    def update(self, sim_time_s: Optional[float] = None) -> float:
        """Advance the controller. Call once per rendered frame; cheap when idle."""
        if self._executor is not None:
            self._executor.spin_once(timeout_sec=0.0)

        if sim_time_s is not None:
            if self._start_time is None:
                self._start_time = sim_time_s
            elapsed = sim_time_s - self._start_time
        else:
            elapsed = 0.0

        if self._override is not None:
            target = self._override
        elif self._schedule is not None:
            target = self._schedule.value_at(elapsed)
        else:
            target = self._turbidity

        self._apply(target)
        self._publish(elapsed)
        return self._turbidity

    def _apply(self, turbidity: float, force: bool = False) -> None:
        value = float(np.clip(turbidity, TURBIDITY_MIN, TURBIDITY_MAX))
        self._turbidity = value

        # Skip the Warp allocations when nothing changed -- update() runs every frame and
        # a static-turbidity run should cost nothing.
        if not force and self._applied_turbidity is not None:
            if abs(value - self._applied_turbidity) < 1e-6:
                return

        backscatter_value, atten_coeff, backscatter_coeff = self._profile.coefficients(value)

        try:
            import warp as wp

            # These three attributes are read by the Warp kernel on every launch, so the
            # assignment takes effect on the next rendered frame.
            self._camera._backscatter_value = wp.vec3f(*backscatter_value.tolist())
            self._camera._atten_coeff = wp.vec3f(*atten_coeff.tolist())
            self._camera._backscatter_coeff = wp.vec3f(*backscatter_coeff.tolist())
        except Exception as exc:  # pragma: no cover - depends on the sim environment
            print(f"[turbidity] failed to apply turbidity {value:.3f}: {exc}")
            return

        self._applied_turbidity = value
        if self._verbose:
            print(f"[turbidity] {value:.3f} "
                  f"atten={np.round(atten_coeff, 3).tolist()} "
                  f"backscatter={np.round(backscatter_coeff, 3).tolist()}")

    def _publish(self, elapsed_s: float) -> None:
        if self._publisher is None:
            return
        if self._publish_period_s > 0.0 and self._last_publish_time >= 0.0:
            if (elapsed_s - self._last_publish_time) < self._publish_period_s:
                return

        try:
            from std_msgs.msg import Float32

            message = Float32()
            message.data = float(self._turbidity)
            self._publisher.publish(message)
            self._last_publish_time = elapsed_s
        except Exception:  # pragma: no cover - depends on the sim environment
            pass

    @property
    def turbidity(self) -> float:
        return self._turbidity

    def close(self) -> None:
        try:
            if self._executor is not None:
                self._executor.remove_node(self._node)
                self._executor = None
            if self._node is not None:
                self._node.destroy_node()
                self._node = None
            self._publisher = None

            # Never shut down a context we did not create -- the ROS 2 bridge's
            # publishers live in it and would go down with it.
            if self._owns_ros_context:
                import rclpy

                if rclpy.ok():
                    rclpy.shutdown()
                self._owns_ros_context = False
        except Exception as exc:  # pragma: no cover - teardown must not mask a real error
            print(f"[turbidity] error during shutdown: {exc}")
