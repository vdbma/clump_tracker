from dataclasses import dataclass


@dataclass
class Clump:
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    mass: float
    ncells: float
    area: float
    max_density: float

    @property
    def coords(self):
        return self.x, self.y, self.z
