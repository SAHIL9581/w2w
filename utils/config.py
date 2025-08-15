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

CLIENT_MAPPING = {
    "CALI": "None", "RSHA": "None", "RMED": "None", "RDEP": "RES", "RHOB": "None",
    "GR": "None", "SGR": "SN18", "NPHI": "None", "PEF": "None", "DTC": "None",
    "SP": "SP", "BS": "None", "ROP": "None", "DTS": "None", "DCAL": "None",
    "DRHO": "None", "MUDWEIGHT": "None", "RMIC": "None", "ROPA": "None", "RXO": "IND"
}
