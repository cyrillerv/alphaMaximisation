from dataclasses import dataclass

@dataclass(frozen=True) # frozen=True empêche la modification accidentelle des valeurs
class Config:
    UNIT = "months"
    WINDOW = 21  # in months 
    STEP = 1     # in months   
    START_DATE = "2010-01-01"
    MAX_VOL = 0.30
    INITIAL_CAPITAL = 1_000_000
    BENCHMARK = "^FCHI"