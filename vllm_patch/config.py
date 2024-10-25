from dataclasses import dataclass, fields
from typing import Any, Dict, Optional

import yaml


@dataclass
class SpecConfig:

    keep_strategy: Optional[str]
    keep_kwargs: Optional[Dict[str, Any]] = None
    gradient_checkpointing: bool = True
    algo: str = "backprop"

    @classmethod
    def from_path(cls, config_path: Optional[str] = None):
        if config_path is None:
            return cls()

        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Get the fields of the dataclass
        field_names = {f.name for f in fields(cls)}
        
        # Check for unused fields in the YAML
        unused_fields = set(data.keys()) - field_names
        if unused_fields:
            raise ValueError(f"Unused fields in YAML: {unused_fields}.")
        
        # Filter out keys that are not fields of the dataclass
        used_data = {k: v for k, v in data.items() if k in field_names}
        
        # Return an instance of the dataclass populated with the filtered data
        return cls(**used_data)
    
    def __post_init__(self):
        assert self.keep_strategy in ["adaptive", "percentage"]
        assert self.algo in ["backprop"]

        if self.keep_strategy is None:
            self.keep_strategy = "percentage"
            self.keep_kwargs["percentage"] = 0.5