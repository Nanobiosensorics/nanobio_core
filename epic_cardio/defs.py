WELL_IDS = [['C1', 'C2', 'C3', 'C4'], ['B1', 'B2', 'B3', 'B4'], ['A1', 'A2', 'A3', 'A4']] # Tükrözés korrigálva
WIDTH = 80

def flatten(t):
    return [item for sublist in t for item in sublist]

WELL_NAMES = sorted(flatten(WELL_IDS))