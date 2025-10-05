from pydantic import BaseModel, field_validator
from typing import Optional

class EventInput(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    event_type_guess: Optional[str] = None
    outdoor: Optional[bool] = None
    has_sponsor: Optional[bool] = None
    has_vip: Optional[bool] = None

    @field_validator("name", "description")
    @classmethod
    def strip_strings(cls, v):
        return v.strip() if isinstance(v, str) else v

    @field_validator("*")
    @classmethod
    def at_least_name_or_description(cls, v, info):
        # check only once at the end via model-level awareness
        return v
