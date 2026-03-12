"""
Custom robosuite environment with 6 colored blocks on a table.

Colors are fixed; sizes (small/large) and positions are randomized each reset.
Uses hard_reset=True so the model is rebuilt every reset, allowing size re-randomization.
"""

from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler

from taska.config import (
    BLOCK_COLORS,
    PLACEMENT_X_RANGE,
    PLACEMENT_Y_RANGE,
    PLACEMENT_Z_OFFSET,
    SIZE_CATEGORIES,
    SIZE_MAP,
    TABLE_FULL_SIZE,
    TABLE_OFFSET,
)


class BlockManipulationEnv(ManipulationEnv):
    """
    Tabletop environment with 6 colored blocks for the Code-as-Policies project.

    Blocks have fixed colors (red, blue, green, yellow, purple, orange).
    Each reset randomizes:
      - Block sizes: each block independently assigned "small" or "large"
      - Block positions: scattered on the table via UniformRandomSampler
    """

    def __init__(
        self,
        robots="Panda",
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        table_full_size=TABLE_FULL_SIZE,
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=False,
        use_object_obs=True,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=False,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=10000,
        ignore_done=True,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
    ):
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array(TABLE_OFFSET)

        self.use_object_obs = use_object_obs
        self.placement_initializer = placement_initializer

        # Populated during _load_model(); stores per-block metadata and BoxObject refs.
        self.block_objects = []
        self.block_meta = []

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

    def reward(self, action=None):
        return 0.0

    def _load_model(self):
        super()._load_model()

        # Position the robot base for the table
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](
            self.table_full_size[0]
        )
        self.robots[0].robot_model.set_base_xpos(xpos)

        # Create the table arena
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        # Create 6 blocks with randomized sizes
        self.block_objects = []
        self.block_meta = []

        for i, block_def in enumerate(BLOCK_COLORS):
            size_cat = self.rng.choice(SIZE_CATEGORIES)
            half_extents = SIZE_MAP[size_cat]

            block = BoxObject(
                name=block_def["name"],
                size=half_extents,
                rgba=block_def["rgba"],
            )

            self.block_objects.append(block)
            self.block_meta.append(
                {
                    "id": i,
                    "name": block_def["name"],
                    "color": block_def["color"],
                    "size": size_cat,
                    "rgba": block_def["rgba"],
                    "half_extents": list(half_extents),
                }
            )

        # Placement sampler for randomized positions
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.block_objects)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="BlockSampler",
                mujoco_objects=self.block_objects,
                x_range=PLACEMENT_X_RANGE,
                y_range=PLACEMENT_Y_RANGE,
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=PLACEMENT_Z_OFFSET,
                rng=self.rng,
            )

        # Assemble the full MuJoCo model
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.block_objects,
        )

    def _setup_references(self):
        super()._setup_references()

        # Map block integer IDs to MuJoCo body IDs
        self.block_body_ids = {}
        for meta, obj in zip(self.block_meta, self.block_objects):
            self.block_body_ids[meta["id"]] = self.sim.model.body_name2id(
                obj.root_body
            )

    def _setup_observables(self):
        observables = super()._setup_observables()

        if self.use_object_obs:
            modality = "object"

            for meta in self.block_meta:
                bid = meta["id"]
                bname = meta["name"]

                def _make_pos_sensor(block_id, block_name):
                    @sensor(modality=modality)
                    def block_pos(obs_cache):
                        return np.array(
                            self.sim.data.body_xpos[self.block_body_ids[block_id]]
                        )

                    block_pos.__name__ = f"{block_name}_pos"
                    return block_pos

                s = _make_pos_sensor(bid, bname)
                observables[s.__name__] = Observable(
                    name=s.__name__,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        super()._reset_internal()

        if not self.deterministic_reset:
            object_placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                )

    def _check_success(self):
        return False
