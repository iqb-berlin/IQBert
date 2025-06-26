import os
from typing import Optional

from fastapi import HTTPException
from pydantic import BaseModel, Field
from redis import StrictRedis

from ics_components.common import CoderRegistry as CoderRegistryInterface
from ics_models import Coder, Task as TaskBase, TaskUpdate as TaskUpdateBase

redis_host = os.getenv('REDIS_HOST') or 'localhost'
redis_store = StrictRedis(host=redis_host, port=6379, db=0, decode_responses=True)

class IcService(BaseModel):
    @staticmethod
    def name() -> str:
        return "iqbert"

class TaskInstructions(BaseModel):
    @staticmethod
    def description() -> str:
        return ("Training parameter: TODO")

class Task(TaskBase):
    instructions: Optional[TaskInstructions] = None

class TaskUpdate(TaskUpdateBase):
    instructions: Optional[TaskInstructions] = None

class CoderRegistry(CoderRegistryInterface):
    def list_coders(self) -> list[Coder]:
        models = []
        for subdir in os.listdir('./data'):
            if subdir.startswith("."):
                continue
            models.append(subdir)
        return [Coder(id=model_id, label=model_id) for model_id in models]

    def delete_coder(self, coder_id: str) -> None:
        try:
            os.remove(f"./data/{coder_id}")
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Coder id {coder_id} not found")
        except PermissionError:
            raise HTTPException(status_code=403, detail=f"Could not delete file of {coder_id}")
