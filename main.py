from typing import Annotated, List, Tuple
from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum
from fastapi.testclient import TestClient

from tests.utils.read_session_data import read_session_data
from inference import make_prediction


BATCH_DURATION = 3000  # ms
EXPECTED_FREQUENCY = 33  # Hz
SENSORS_NUMBER = 3
BATCH_SIZE = ((BATCH_DURATION / 1000) * EXPECTED_FREQUENCY) * SENSORS_NUMBER


class Activity(str, Enum):
    cycling = "cycling"


class Sensor(int, Enum):
    accelerometer = 0
    gyroscope = 1
    magnetometer = 2


class Batch(BaseModel):
    sequence_number: int
    sensors_readings: Annotated[List[Tuple[int,
                                           Sensor, float, float, float]], BATCH_SIZE]


class Response(BaseModel):
    result: bool


app = FastAPI()


@app.post("/predict")
async def predict(body: Batch, activity: Activity) -> Response:
    # result = make_prediction()
    return {
        "result": True
    }


client = TestClient(app)

TEST_FILE_NAME = "Denys-1678895240206-sensor-readings.csv"


def test_send_one_sensors_readings_batch():
    response = client.post("/predict",
                           json={
                               "sequence_number": 0,
                               "sensors_readings": [[1679345782003, 2, 1.93127, 16237.3, 12356.3], [1679345782003, 1, 1.93127, 16237.3, 12356.3]]
                           })
    assert (2 + 2 == 4)
