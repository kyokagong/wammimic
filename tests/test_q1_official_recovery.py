import mujoco
import pytest

from scripts.prepare_q1_official_recovery import lowest_collision


def test_collision_floor_respects_rotation_and_ignores_visuals():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom type="box" size=".1 .2 .3" pos="0 0 .5" euler="0 90 0"/>
      <geom type="sphere" size=".05" pos="0 0 .6"/>
      <geom type="sphere" size=".1" pos="0 0 -5" contype="0" conaffinity="0"/>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert lowest_collision(model, data) == pytest.approx(.4)
