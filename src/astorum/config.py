from pathlib import Path

# Database files live alongside this module under src/astorum/databases/
DATABASE_DIR = Path(__file__).parent / "databases"

DETECTOR_EFFICIENCY = DATABASE_DIR / "interpolated_SDD_efficiency.txt"
PERIODIC_TABLE_INFO = DATABASE_DIR / "periodic_table_symbols.json"
PERIODIC_TABLE_NUMBERS = DATABASE_DIR / "periodic_table_number.json"
SIEGBAHN_TO_IUPAC = DATABASE_DIR / "siegbahn_to_iupac.json"
MASS_ABSORPTION_COEFFICIENTS = DATABASE_DIR / "interpolated_mass_absorption_coefficients.json"
XRAY_200KeV = DATABASE_DIR / "200keV_xrays_transformed.json"
XRAY_300KeV = DATABASE_DIR / "300keV_xrays_transformed.json"