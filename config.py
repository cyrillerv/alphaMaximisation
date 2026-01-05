from dataclasses import dataclass
import pandas as pd 

@dataclass(frozen=True) # frozen=True empêche la modification accidentelle des valeurs
class Config:
    UNIT = "months"
    WINDOW = 21  # in months 
    STEP = 1     # in months   
    START_DATE = "2024-01-01"
    MAX_VOL = 0.15
    INITIAL_CAPITAL = 1_000_000
    BENCHMARK = "^GSPC"
    END_DATE_STRAT = pd.Timestamp('2024-12-29') # checker suffisamment stock price pour ça