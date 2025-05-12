import json, re
import ics.isaac_sas as isaac_sas
from typing import List
from pydantic import StrictInt, BaseModel

from ics_components.common.models import TrainingResult
from lib.feature_extraction.data import ShortAnswerInstance
from ics.models import LanguageDataRequest
from ics.implementation import TaskInstructions
from ics_models import Response



def code(model_id: str, input_data: List[Response]) -> List[Response]:
    return []

def train(task_label: str, instructions: TaskInstructions, input_data: List[Response]) -> TrainingResult:
    pass
