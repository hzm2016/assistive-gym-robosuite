from envs.robosuite.robosuite .models.arenas import Arena
from envs.robosuite.robosuite .utils.mjcf_utils import xml_path_completion


class EmptyArena(Arena):
    """Empty workspace."""

    def __init__(self):
        super().__init__(xml_path_completion("arenas/empty_arena.xml"))
