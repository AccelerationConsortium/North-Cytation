from copy import copy
import math
import logging
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
    cast,
)  # pylint: disable=unused-import

import north.n9_kinematics as n9


class Location:
    params = ['name', 'a0', 'a1', 'a2', 'a3', 'x', 'y', 'z']
    param_labels = ['Name', 'Gripper', 'Elbow', 'Shoulder', 'Z Axis', 'X', 'Y', 'Z']

    MM_PREC_STR = "%.1f"

    elbow_offset = 21250
    shoulder_offset = 33667
    z_axis_max_cnts = 26200
    z_axis_offset = 26 # test grippers - TODO: MAKE THIS GENERIC, TOOL IK ETC.

    ELBOW_COUNTS_PER_REV = 51000
    SHOULDER_COUNTS_PER_REV = 101000
    Z_AXIS_COUNTS_PER_MM = 100
# 0.01mm/cnt -- 1000cnts /rev 10mm pitch screw
    JOINT_SPACE = 0
    TASK_SPACE = 1
    GRID = 2

    ik_names = [
        "Shoulder center",
        "Shoulder out"
    ]

    SHOULDER_CENTER = 0
    SHOULDER_OUT = 1

    # dialog context mode enum
    DEFINE= 1
    CAPTURE = 2
    EDIT = 3

    #tool dir enum
    POS_X = 0
    POS_Y = 1
    NEG_X = 2
    NEG_Y = 3

    #parallels the tool dir enum
    tool_dir_names = [
        "+X",
        "+Y",
        "-X",
        "-Y",
        "Other"
    ]

    #tool dir lookup
    tool_directions = {
        POS_X : 0,
        POS_Y : math.pi/2,
        NEG_X : math.pi,
        NEG_Y : -math.pi/2
    }

    #tool name enum
    GRIPPER = 0
    PIPETTE_TIP = 1
    BERNOULLI = 2
    VACUUM = 3

    #parallels the tool name enum
    tool_names = [
        "Gripper",
        "Pipette Tip",
        "Bernoulli Gripper",
        "Vacuum Gripper"
    ]

    tool_keys = [
        "gripper",
        "pipette",
        "bernoulli",
        "vacuum"
    ]

    tool_offsets = {
        GRIPPER : [0, 0, 0],
        PIPETTE_TIP : [0, 0, -28], #pipette tip is 44mm offset from gripper in x, but is handled differently as it is not held
        BERNOULLI : [70, 0, 10],
        VACUUM : [70, 0, 8]
    }

    grasped_tools = [BERNOULLI, VACUUM]

    def __init__(self, name='new_loc', mode=None, input=[0, 0, 0, 10000],
                 tool:int = None, ik_soln:int = None, reference:bool = False, linked_parent_name:str=None):
        self._counts = [0, 0, 0, 0]
        self._mms = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'th': 0.0}
        self.is_valid = False
        self.name: str = name
        #self.tool_orientation=None

        self._ik_soln = ik_soln if ik_soln is not None else self.SHOULDER_CENTER
        self.tool = tool if tool is not None else self.GRIPPER

        if mode == self.JOINT_SPACE or mode is None:
            self.update_counts(input)
        elif mode == self.TASK_SPACE:
            if len(input) == 3:  # if no theta (gripper angle) given:
                input.append(0)
            self.update_mms(input)

        self.is_reference=reference

        self.linked_parent_name = linked_parent_name
        self.linked_parent = None
        self.linked_child_names = set()

    def __str__(self):
        return self.name
    #########################
    # Public getter methods #
    def get_name(self) -> str:
        return self.name

    def get_linked_parent_name_str(self) -> str:
        return self.linked_parent_name if self.linked_parent_name is not None else ''

    def get_counts(self) -> List[int]:
        return self._counts

    def get_counts_str(self) -> List[str]:
        return [str(self._counts[0]),
                str(self._counts[1]),
                str(self._counts[2]),
                str(self._counts[3])]

    def get_mm(self) -> List[float]:
        return [self._mms['x'], self._mms['y'], self._mms['z'], math.degrees(self._mms['th'])]

    def get_mm_str(self) -> List[str]:
        return [self.MM_PREC_STR % coord for coord in self.get_mm()]

    def get_values(self) -> List[any]:
        return self._counts + self.get_mm()

    def get_values_str(self) -> List[str]:
        return self.get_counts_str() + self.get_mm_str()

    def get_py_str(self) -> str:
        return self.name + " =  " + str(self._counts) + "\n"

    # def get_save_str(self):
    #     return " ".join(["L",
    #                      str(self.tool),
    #                      self.name] + \
    #                     [str(e) for e in self.counts] + \
    #                     [str(self.tool_direction),
    #                      str(self.ik_soln)]) + \
    #                 "\n"

    def get_tool_str(self) -> str:
        if self.tool == self.GRIPPER:
            return "(G)"
        elif self.tool == self.PIPETTE_TIP:
            return "(P)"
        elif self.tool == self.BERNOULLI:
            return "(B)"
        elif self.tool == self.VACUUM:
            return "(V)"
        else:
            return ""

    def get_ik_soln(self) -> int:
        return self._ik_soln

    def get_dict(self) -> Dict[str, Any]:
        return {'type': 'Location',
                'tool': self.tool,
                'counts': self._counts,
                'ik_soln': self._ik_soln,
                'reference': self.is_reference,
                'linked_parent_name': self.linked_parent_name
                }

    ################################
    # Public setter/update methods #
    def update_name(self, new_name :str):
        self.name = new_name

    def update_count(self, index :int, value :int):
        self._counts[index] = value
        self._recalculate_mms()
        self.is_valid = True

    def update_counts(self, values):
        if len(values) != len(self._counts):
            raise ValueError("values array must have size 4")
        self._counts = values
        self._recalculate_mms()
        self.is_valid = True

    def update_mm(self, name :str, value :float):
        if name not in self._mms:
            raise ValueError("wrong name ("+name+") when updating mm value")
        # theta is passed in degrees
        if name == 'th': #convert to rad
            value = math.radians(value)
        self._mms[name] = value
        self._recalculate_counts()

    def update_mms(self, values):
        if len(values) != len(self._mms):
            raise ValueError("values array must have size 4")
        self._mms['x'] = values[0]
        self._mms['y'] = values[1]
        self._mms['z'] = values[2]
        self._mms['th'] = math.radians(values[3])
        self._recalculate_counts()

    def update_ik_soln(self, ik_soln):
        self._ik_soln = ik_soln
        self._recalculate_counts()

    ###################
    # Private methods #
    def _recalculate_counts(self):
        self._counts = self._ik()

    def _recalculate_mms(self):
        x, y, th = n9.fk(self._counts[0],
                     self._counts[1],
                     self._counts[2],
                     self.tool_offsets[self.tool][0],
                     self.tool == self.PIPETTE_TIP)
        z = (n9.Z_AXIS_MAX_COUNTS - self._counts[3]) / n9.Z_AXIS_COUNTS_PER_MM + n9.Z_AXIS_OFFSET
        z -= self.tool_offsets[self.tool][2]
        self._mms = {'x': x, 'y': y, 'z': z, 'th': th}

    def _ik(self) -> List[int]:
        # TODO: should come from a common place as north_c9 library
        # TODO: HOW TO DEAL WITH CONFIGURATIONS?
        # TODO: pos/joint limits, exception
        # TODO: support perpendicular and z-axis tool offset, use DH-coordinate nomenclature
        x = self._mms['x']; y = self._mms['y']; z = self._mms['z']
        theta = self._mms['th']
        try:
            gripper_theta, elbow_theta, shoulder_theta = n9.ik(x, y,
                                                                  tool_length=self.tool_offsets[self.tool][0],
                                                                  tool_orientation= theta, #self.tool_directions[self.tool_direction],
                                                                  pipette_tip_offset=(self.tool == self.PIPETTE_TIP),
                                                                  shoulder_preference=self._ik_soln)

            gripper_cts = n9.rad_to_counts(n9.GRIPPER, gripper_theta) if theta is not None else 0
            elbow_cts = n9.rad_to_counts(n9.ELBOW, elbow_theta)
            shoulder_cts = n9.rad_to_counts(n9.SHOULDER, shoulder_theta)
            z_axis_cts = n9.mm_to_counts(n9.Z_AXIS, z + self.tool_offsets[self.tool][2])

            if elbow_cts < 0 or elbow_cts > n9.ELBOW_MAX_COUNTS \
                or shoulder_cts < 0 or shoulder_cts > n9.SHOULDER_MAX_COUNTS \
                or z_axis_cts < 0 or z_axis_cts > n9.Z_AXIS_MAX_COUNTS:
                raise ValueError

            self.is_valid = True
        except ValueError as ve:
            gripper_cts = -1
            elbow_cts = -1
            shoulder_cts = -1
            z_axis_cts = -1
            self.is_valid = False
            logging.error(f'{self.__class__.__name__}._ik(): ValueError - {str(ve)}')

        except ZeroDivisionError as zde:
            gripper_cts = -1
            elbow_cts = -1
            shoulder_cts = -1
            z_axis_cts = -1
            self.is_valid = False
            logging.error(f'{self.__class__.__name__}._ik(): ZeroDivisionError - {str(zde)}')
        return [gripper_cts, elbow_cts, shoulder_cts, z_axis_cts]

    def unlink_parent(self):
        self.linked_parent_name = None
        self.linked_parent = None

    def link_parent(self, parent_loc):
        if parent_loc is None:
            self.linked_parent_name = None
            self.linked_parent = None
            return
        self.linked_parent_name = parent_loc.name
        self.linked_parent = copy(parent_loc)
        parent_loc.linked_child_names.add(self.name)

    def on_parent_update(self, new_parent):
        prev_parent_mm = self.linked_parent.get_mm()
        new_parent_mm = new_parent.get_mm()
        this_mm = self.get_mm()
        for i in range(3):
            this_mm[i] += new_parent_mm[i] - prev_parent_mm[i]
        self.update_mms(this_mm)
        self.link_parent(new_parent)

class Grid(Location):
    def __init__(self,
                 name='new_grid',
                 origin=[200, 100, 50, 0],
                 n=[1, 1, 1],
                 pitch=[1, 1, 1],
                 tool=None,
                 ik_soln=None,
                 reference=False,
                 rot_angle=0,
                 linked_parent_name=None):

        super().__init__(name, Location.TASK_SPACE, origin, tool, ik_soln, reference, linked_parent_name)
        self.n = n
        self.pitch = pitch
        self.children = []
        self.rot_angle = rot_angle

        self.make_grid()

    def make_grid(self):
        self.children = []

        origin = self.get_mm()
        pos = [0, 0, 0, 0]
        for i in range(self.n[0]):
            for j in range(self.n[1]):
                for k in range(self.n[2]):
                    pos[0] = i*self.pitch[0]
                    pos[1] = j*self.pitch[1]
                    pos[2] = k*self.pitch[2]

                    cos_r = math.cos(self.rot_angle)
                    sin_r = math.sin(self.rot_angle)

                    rot_x = pos[0]*cos_r - pos[1]*sin_r
                    rot_y = pos[0]*sin_r + pos[1]*cos_r

                    pos[0] = rot_x + origin[0]
                    pos[1] = rot_y + origin[1]
                    pos[2] += origin[2]
                    pos[3] = origin[3]

                    self.children.append(Location(
                        name=self.name + '[' + str(len(self.children)) + ']',
                        mode=Location.TASK_SPACE,
                        input=pos,
                        tool=self.tool,
                        ik_soln=self._ik_soln,
                        reference=self.is_reference
                    ))

    def add_child(self, loc):
        self.children.append(loc)

    def update_child (self, child_index :int, new_child):
        new_child.is_reference = self.is_reference
        self.children[child_index] = new_child

    def remove_child(self, child_index):
        self.children.pop(child_index)

    def get_values_str(self):
        return []

    def get_py_str(self) -> str:
        return self.name + " =  " + str([child._counts for child in self.children]) + "\n"

    # def get_save_str(self):
    #     return " ".join(["G",
    #                     str(self.tool),
    #                     self.name] + \
    #                     [str(e) for e in self.get_mm()] + \
    #                     [str(e) for e in self.n] + \
    #                     [str(e) for e in self.pitch] + \
    #                     [str(self.tool_direction),
    #                     str(self.ik_soln)]) + \
    #            "\n" + \
    #            "".join([" " + child.get_save_str() for child in self.children])


    def get_dict(self) -> Dict[str, Any]:
        return {'type': 'Grid',
                'tool': self.tool,
                'ik_soln': self._ik_soln,
                'reference': self.is_reference,
                'origin' : self.get_mm(),
                'n': self.n,
                'pitch': self.pitch,
                'rotation': self.rot_angle,
                'children': {child.name: child.get_dict() for child in self.children},
                'linked_parent_name': self.linked_parent_name
                }