from typing import List, Optional
from pydantic import BaseModel, StrictStr, StrictInt, Field
from fastapi import HTTPException
from ics_components.common import CoderRegistry as CoderRegistryInterface
from ics_models import Coder, Task as TaskBase, TaskUpdate as TaskUpdateBase
from ics.isaac_sas import fetch_stored_models, delete_model


class TaskInstructions(BaseModel):
    @staticmethod
    def description() -> str:
        return "TODO"


class Task(TaskBase):
    instructions: Optional[TaskInstructions] = None

class TaskUpdate(TaskUpdateBase):
    instructions: Optional[TaskInstructions] = None

class CoderRegistry(CoderRegistryInterface):
    def list_coders(self) -> list[Coder]:
        return []

    def delete_coder(self, coder_id: str) -> None:
        pass