from pydantic import BaseModel

class FlightInput(BaseModel):
    weather: str
    dep_hour: int
    route: str
    aircraft_type: str
