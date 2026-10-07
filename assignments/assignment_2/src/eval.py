"""Evaluate Snake neural-network controllers.

No ARIEL source files are modified.
The supplied A2_template_2026.py is imported and reused where appropriate.
"""

from __future__ import annotations

# cSpell:ignore qpos qvel

import sys
from pathlib import Path

import mujoco as mj
import numpy as np
import numpy.typing as npt

from ariel.body_phenotypes.robogen_lite.prebuilt_robots.john_set import snake
from ariel.utils.runners import simple_runner

from config import HIDDEN_SIZE, RHYTHM_FREQUENCY, SIM_DURATION, SPAWN_POS, TARGET_POS


# ---------------------------------------------------------------------------
# Import supplied Assignment 2 template
# ---------------------------------------------------------------------------

ASSIGNMENT_DIR = Path(__file__).resolve().parents[1]

if str(ASSIGNMENT_DIR) not in sys.path:
    sys.path.insert(0, str(ASSIGNMENT_DIR))

import A2_template_2026 as template  # noqa: E402


FloatArray = npt.NDArray[np.float64]


class Evaluator:
    """
    Compile one Snake world and evaluate many controllers.
    """

    def __init__(self) -> None:
        # Use the SimpleFlatWorld construction supplied by Assignment 2.
        self.world = template.build_world()

        # Use Snake instead of the template's Gecko.
        self.robot = snake()

        self.world.spawn(self.robot.spec, position=list(SPAWN_POS), correct_collision_with_floor=True)

        self.model = self.world.spec.compile()
        self.data = mj.MjData(self.model)

        mj.mj_resetData(self.model, self.data)
        mj.mj_forward(self.model, self.data)

        self.output_size = int(self.model.nu)
        self.input_size = len(self._controller_inputs(self.data))

        self.w1_shape = (self.input_size, HIDDEN_SIZE)
        self.w2_shape = (HIDDEN_SIZE, self.output_size)

        self.num_weights = int(
            np.prod(self.w1_shape)
            + np.prod(self.w2_shape)
        )

        # Make sure our target matches the Assignment 2 template.
        if not np.allclose(
            np.asarray(TARGET_POS),
            np.asarray(template.TARGET_POSITION),
        ):
            raise ValueError(
                "TARGET_POS in config.py differs from "
                "TARGET_POSITION in A2_template_2026.py."
            )

    # -----------------------------------------------------------------------
    # NN inputs
    # -----------------------------------------------------------------------

    def _controller_inputs(self, data: mj.MjData) -> FloatArray:
        """
        Construct the fixed neural-network input vector.
        """

        qpos = np.asarray(data.qpos, dtype=np.float64)
        qvel = np.asarray(data.qvel, dtype=np.float64)

        position = template.get_core_position(data)
        target = np.asarray(TARGET_POS, dtype=np.float64)

        # Direction from current position to target.
        target_delta = (target[:2] - position[:2]) / 2.0

        # Periodic timing information for rhythmic locomotion.
        phase = 2.0 * np.pi * RHYTHM_FREQUENCY * data.time
        rhythm = np.asarray([np.sin(phase), np.cos(phase)],dtype=np.float64)

        # Limit extreme velocity values.
        qvel_scaled = np.tanh(qvel / 5.0)

        return np.concatenate([qpos, qvel_scaled, target_delta, rhythm])

    # -----------------------------------------------------------------------
    # Genotype -> network weights
    # -----------------------------------------------------------------------

    def unpack_weights(self, genotype: FloatArray) -> tuple[FloatArray, FloatArray]:
        """
        Convert flat genotype to the two NN weight matrices.
        """

        if len(genotype) != self.num_weights:
            raise ValueError(
                f"Expected genotype length {self.num_weights}, "
                f"received {len(genotype)}."
            )

        split = int(np.prod(self.w1_shape))

        w1 = genotype[:split].reshape(self.w1_shape)
        w2 = genotype[split:].reshape(self.w2_shape)

        return w1, w2

    def controller(
        self,
        data: mj.MjData,
        genotype: FloatArray,
    ) -> FloatArray:
        """Feed-forward neural-network controller."""

        w1, w2 = self.unpack_weights(genotype)

        inputs = self._controller_inputs(data)

        hidden = np.tanh(inputs @ w1)
        outputs = np.tanh(hidden @ w2)

        return outputs * (np.pi / 2.0)

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------

    def evaluate(
        self,
        genotype: FloatArray,
        spawn_position: tuple[float, float, float] = SPAWN_POS,
    ) -> float:
        """Evaluate one genotype. Lower fitness is better."""

        genotype = np.asarray(genotype, dtype=np.float64)

        if genotype.shape != (self.num_weights,):
            raise ValueError(
                f"Genotype must have shape ({self.num_weights},), "
                f"received {genotype.shape}."
            )

        mj.set_mjcb_control(None)

        # Reset physics to the model's initial state.
        mj.mj_resetData(self.model, self.data)

        # Root free joint stores world x/y/z in qpos[0:3].
        self.data.qpos[0:3] = np.asarray(spawn_position,dtype=np.float64)
        mj.mj_forward(self.model, self.data)
        initial_position = template.get_core_position(self.data)

        def control_callback(model: mj.MjModel, data: mj.MjData) -> None:
            del model
            actions = self.controller(data, genotype)

            if not np.all(np.isfinite(actions)):
                raise FloatingPointError(
                    "Controller produced NaN or infinite actuator commands."
                )

            data.ctrl[:] = actions

        try:
            mj.set_mjcb_control(control_callback)

            simple_runner(
                self.model,
                self.data,
                duration=SIM_DURATION,
            )

        finally:
            mj.set_mjcb_control(None)

        final_position = template.get_core_position(self.data)

        return template.fitness_function(initial_position, final_position)

    # -----------------------------------------------------------------------
    # Generalisation testing
    # -----------------------------------------------------------------------

    @staticmethod
    def test_positions(n_positions: int,radius: float) -> list[tuple[float, float, float]]:
        """
        Generate equally spaced spawn positions around the target.
        """

        target_x, target_y, _ = TARGET_POS

        positions: list[tuple[float, float, float]] = []

        for i in range(n_positions):
            theta = 2.0 * np.pi * i / n_positions

            x = target_x + radius * np.cos(theta)
            y = target_y + radius * np.sin(theta)
            positions.append((float(x), float(y), SPAWN_POS[2]))

        return positions

    def evaluate_generalisation(self, genotype: FloatArray, n_positions: int, radius: float) -> list[float]:
        """
        Evaluate a trained controller from all test positions.
        """
        return [self.evaluate(genotype, position) for position in self.test_positions(n_positions, radius)]


def main() -> None:
    evaluator = Evaluator()

    print("Snake controller")
    print("=" * 50)
    print(f"qpos dimension  : {len(evaluator.data.qpos)}")
    print(f"qvel dimension  : {len(evaluator.data.qvel)}")
    print(f"NN inputs       : {evaluator.input_size}")
    print(f"Hidden neurons  : {HIDDEN_SIZE}")
    print(f"NN outputs      : {evaluator.output_size}")
    print(f"Total weights   : {evaluator.num_weights}")


if __name__ == "__main__":
    main()
