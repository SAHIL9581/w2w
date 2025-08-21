# utils/config.py

LOG_CONFIG = [
    {'mnemonic': 'GR', 'range': (0, 150)}, {'mnemonic': 'SGR', 'range': (0, 300)},
    {'mnemonic': 'RSHA', 'range': (0.2, 2000), 'log_scale': True}, {'mnemonic': 'RMED', 'range': (0.2, 2000), 'log_scale': True},
    {'mnemonic': 'RDEP', 'range': (0.2, 2000), 'log_scale': True}, {'mnemonic': 'RXO', 'range': (0.2, 2000), 'log_scale': True},
    {'mnemonic': 'RMIC', 'range': (0.2, 2000), 'log_scale': True}, {'mnemonic': 'SP', 'range': (-150, 150)},
    {'mnemonic': 'DTC', 'range': (40, 200)}, {'mnemonic': 'DTS', 'range': (80, 300)},
    {'mnemonic': 'RHOB', 'range': (1.95, 2.95)}, {'mnemonic': 'DRHO', 'range': (-0.1, 0.1)},
    {'mnemonic': 'NPHI', 'range': (0, 0.6)}, {'mnemonic': 'PEF', 'range': (0, 10)},
    {'mnemonic': 'CALI', 'range': (6, 17)}, {'mnemonic': 'BS', 'range': (6, 17)},
    {'mnemonic': 'DCAL', 'range': (-1, 1)}, {'mnemonic': 'ROP', 'range': (0, 1000)},
    {'mnemonic': 'ROPA', 'range': (0, 1000)}, {'mnemonic': 'MUDWEIGHT', 'range': (8, 22)},
]

LOG_PLOT_COLORS = ['red', 'black', 'blue']

# --- UPDATED MAPPING BASED ON CLIENT REQUEST ---
# This dictionary now defines the specific aliases for the new dataset.
# The "smart mapping" will use this as a fallback if a direct match isn't found.
CLIENT_MAPPING = {
    # Specific mappings requested by the client
    "RDEP": "RES",
    "RMED": "SN18",
    "RSHA": "IND",
    "SP": "SP",

    # All other standard curves set to None as requested,
    # so they will only plot if a direct match is found (e.g., GR -> GR).
    "GR": "None",
    "SGR": "None",
    "RXO": "None",
    "RMIC": "None",
    "DTC": "None",
    "DTS": "None",
    "RHOB": "None",
    "DRHO": "None",
    "NPHI": "None",
    "PEF": "None",
    "CALI": "None",
    "BS": "None",
    "DCAL": "None",
    "ROP": "None",
    "ROPA": "None",
    "MUDWEIGHT": "None",
}