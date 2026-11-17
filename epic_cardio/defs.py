WELL_IDS = [['C1', 'C2', 'C3', 'C4'], ['B1', 'B2', 'B3', 'B4'], ['A1', 'A2', 'A3', 'A4']]  # Tükrözés korrigálva
WIDTH = 80
WELL_NAMES = sorted(item for row in WELL_IDS for item in row)

class RangeType:
    MEASUREMENT_PHASE = 0
    INDIVIDUAL_POINT = 1

class MeasurementType:
    TYPE_NORMAL = 0
    TYPE_HIGH_FREQ = 1

def flatten(t):
    return [item for sublist in t for item in sublist]
