# these are the endpoints provided in the original issac-sas implementation
from fastapi import APIRouter

import ics.isaac_sas as isaac_sas
from ics.models import LanguageDataRequest, PredictFromLanguageDataResponse, ModelIdResponse

router = APIRouter()

