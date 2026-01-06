# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 20:26:14 2026

@author: doaao
"""
import sys
import subprocess
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xlsxwriter'])
import pandas as pd
import numpy as np
import random

# ==========================================
# CONFIGURATION
# ==========================================
FILENAME = "hospital_data.xlsx"
NUM_SCENARIOS = 15   
DAYS = range(1, 8)   
SHIFTS = ['Day', 'Evening', 'Night']
ICU_TYPES = ['Adult', 'Pediatric', 'Neonatal']
SPECIALTIES = ['Anes', 'IntMed', 'GenSurg', 'Peds', 'PedSurg', 'Micro']


NUM_CANDIDATES_PER_SPEC = 15  
NUM_CANDIDATE_NURSES = 80    


REAL_SALARIES = {
    'Anes': 73525,
    'IntMed': 49225,
    'GenSurg': 55491,
    'Peds': 50394,
    'PedSurg': 31275,
    'Micro': 11950
}


staff_data = []

# Physicians
for spec in SPECIALTIES:
    for i in range(1, NUM_CANDIDATES_PER_SPEC + 1):
        # Look up salary from your dictionary, default to 2000 if missing
        base_sal = REAL_SALARIES.get(spec, 2000)
        
        staff_data.append({
            'ID': f'{spec}_{i:02d}',
            'Role': 'Physician',
            'Specialty': spec,
            'WeeklySalary': base_sal, 
            'MaxOvertime': 15,
            'MaxNightShifts': 3
        })

# Nurses
for i in range(1, NUM_CANDIDATE_NURSES + 1):
    staff_data.append({
        'ID': f'Nurse_{i:02d}',
        'Role': 'Nurse',
        'Specialty': 'General',
        'WeeklySalary': 10550,
        'MaxOvertime': 15,
        'MaxNightShifts': 4
    })

df_staff = pd.DataFrame(staff_data)

# ==========================================
# 2. GENERATE SCENARIOS
# ==========================================
with pd.ExcelWriter(FILENAME, engine='xlsxwriter') as writer:
    df_staff.to_excel(writer, sheet_name='Staff', index=False)
    
    for s_num in range(1, NUM_SCENARIOS + 1):
        scenario_data = []
        for icu in ICU_TYPES:
            for day in DAYS:
                for shift in SHIFTS:
                    # Demand Logic
                    if icu == 'Adult': base = 12
                    elif icu == 'Pediatric': base = 8
                    else: base = 5
                    
                    if shift == 'Evening': base *= 0.8
                    if shift == 'Night': base *= 0.6
                    
                    noise = random.randint(-2, 3)
                    patient_demand = max(1, int(base + noise))
                    
                    # Ratios
                    if icu == 'Adult': r = 1/5
                    elif icu == 'Pediatric': r = 1/3
                    else: r = 1/6
                    
                    n_req = np.ceil(patient_demand * r)
                    
                    scenario_data.append({
                        'ICU': icu,
                        'Day': day,
                        'Shift': shift,
                        'Patient_Demand': patient_demand,
                        'Physician_Req': 1,
                        'Nurse_Req': n_req
                    })
        
        df_scen = pd.DataFrame(scenario_data)
        df_scen.to_excel(writer, sheet_name=f'Scenario_{s_num}', index=False)

print(f"✅ Success: '{FILENAME}' created.")
