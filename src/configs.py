from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AlignConfig:
    shape_predictor_path: str = field(default="shape_predictor_68_face_landmarks.dat")
    transform_size: int = field(default=1024)
    padding: bool = field(default=False)
    output_size: Optional[int] = field(default=None)


@dataclass
class TaskConfig:
    dataroot: str = field(default="")
    align: bool = field(default=False)
    crop: bool = field(default=False)
    cut: bool = field(default=False)
    scgan_labels: bool = field(default=False)

    align_config: AlignConfig = field(default_factory=AlignConfig)

