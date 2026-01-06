# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 00:45:57 2026

@author: doaao
"""
import pandas as pd
import numpy as np
import random
import math
import gurobipy as gp
from gurobipy import GRB

# =============================================================================
# 1. CONFIGURATION & CONSTANTS
# =============================================================================
FILENAME = "hospital_data.xlsx"

# --- Sets & Mappings ---
ICU_TYPES = ['Adult', 'Pediatric', 'Neonatal']
SHIFTS = ['Day', 'Evening', 'Night']
SPECIALTIES = ['Anes', 'IntMed', 'GenSurg', 'Peds', 'PedSurg', 'Micro']
WEEKEND_DAYS = [6, 7] # Saturday, Sunday

# --- Costs ---
C_BED = {'Adult': 1000, 'Pediatric': 1200, 'Neonatal': 1500}
C_SALARY_NURSE = 10550
C_SALARY_PHY = {
    'Anes': 73525,
    'IntMed': 49225,
    'GenSurg': 55491,
    'Peds': 50394,
    'PedSurg': 31275,
    'Micro': 11950
}

C_DIVERT = 5000       # Penalty for Blocked Patients
C_OT_PHY = 100        
C_OT_NURSE = 50       
C_SHORTAGE = 100000   # Penalty for missing staff

# Fairness Weights
ALPHA = {'Night_Phy': 50, 'Night_Nurse': 30, 'Week_Phy': 40, 'Week_Nurse': 20}

# --- Constraints ---
MIN_BEDS = {'Adult': 4, 'Pediatric': 4, 'Neonatal': 4}
MIN_PHY_PER_SPEC = {s: 1 for s in SPECIALTIES} # Min 1 hired per spec
RATIOS = {'Adult': 1/5, 'Pediatric': 1/3, 'Neonatal': 1/6}

# Operational Limits
OT_MAX = 15           # Max overtime hours/week
NS_MAX_PHY = 3        # Max night shifts/week
NS_MAX_NURSE = 4
BIG_M = 100 

# --- GOVERNMENT REGULATION MAP ---
REQ_SPECS_PER_ICU = {
    'Adult': ['Anes', 'IntMed', 'GenSurg'],
    'Pediatric': ['Peds', 'Anes', 'PedSurg', 'Micro'],
    'Neonatal': ['Peds']
}

try:
    df_staff_raw = pd.read_excel(FILENAME, sheet_name='Staff')
    
    xls = pd.ExcelFile(FILENAME)
    scenarios = [pd.read_excel(FILENAME, sheet_name=s) for s in xls.sheet_names if s.startswith('Scenario')]
    big_df = pd.concat(scenarios)
    
    avg_df = big_df.groupby(['ICU', 'Day', 'Shift'])[['Patient_Demand', 'Physician_Req', 'Nurse_Req']].mean().reset_index()
    avg_df['Patient_Demand'] = np.ceil(avg_df['Patient_Demand']).astype(int)
    avg_df['Nurse_Req'] = np.ceil(avg_df['Nurse_Req']).astype(int)
    
    DAYS = sorted(avg_df['Day'].unique())
    DEMAND_LOOKUP = avg_df.set_index(['ICU', 'Day', 'Shift'])['Patient_Demand'].to_dict()
    NREQ_LOOKUP = avg_df.set_index(['ICU', 'Day', 'Shift'])['Nurse_Req'].to_dict()
    
    PHYSICIANS = df_staff_raw[df_staff_raw['Role']=='Physician']['ID'].tolist()
    NURSES = df_staff_raw[df_staff_raw['Role']=='Nurse']['ID'].tolist()
    PHY_SPEC_MAP = dict(zip(df_staff_raw['ID'], df_staff_raw['Specialty']))
    SALARY_MAP = dict(zip(df_staff_raw['ID'], df_staff_raw['WeeklySalary']))
    
    HEURISTIC_DEMAND = avg_df.groupby('ICU')['Patient_Demand'].max().to_dict()
    print("Data loaded successfully.")

except Exception as e:
    print(f"CRITICAL ERROR: Could not load data ({e}). Run Part 1 first.")
    exit()

# =============================================================================
# 3. STAGE 1: TABU SEARCH (Bed Capacity)
# =============================================================================
class TabuSearchBeds:
    def __init__(self, demand_profile, max_beds=30, tenure=5, iterations=100):
        self.demand = demand_profile
        self.max_beds = max_beds
        self.tabu_tenure = tenure
        self.max_iter = iterations
        self.tabu_list = {} 

    def _calculate_cost(self, beds):
        capital_cost = sum(beds[i] * C_BED[i] for i in ICU_TYPES)
        quality_penalty = 0
        approx_op_cost = 0
        
        for i in ICU_TYPES:
            d = self.demand.get(i, 0)
            serviced = min(d, beds[i])
            blocked = max(0, d - beds[i])
            quality_penalty += blocked * C_DIVERT
            n_req = math.ceil(serviced * RATIOS[i])
            approx_op_cost += n_req * C_SALARY_NURSE * 3 
            
        approx_op_cost += 10000 * 3 
        return capital_cost + quality_penalty + approx_op_cost

    def _get_neighbors(self, current_beds):
        neighbors = []
        for i in ICU_TYPES:
            if current_beds[i] < self.max_beds:
                new_b = current_beds.copy()
                new_b[i] += 1
                neighbors.append((new_b, f"{i}+1"))
            if current_beds[i] > MIN_BEDS[i]:
                new_b = current_beds.copy()
                new_b[i] -= 1
                neighbors.append((new_b, f"{i}-1"))
        return neighbors

    def solve(self):
        current = {i: MIN_BEDS[i] + 2 for i in ICU_TYPES}
        best = current.copy()
        best_cost = self._calculate_cost(best)
        
        for it in range(self.max_iter):
            neighbors = self._get_neighbors(current)
            best_neighbor = None
            best_neighbor_cost = float('inf')
            best_move_sig = None

            for sol, sig in neighbors:
                is_tabu = (sig in self.tabu_list and self.tabu_list[sig] > it)
                cost = self._calculate_cost(sol)
                if (not is_tabu) or (cost < best_cost):
                    if cost < best_neighbor_cost:
                        best_neighbor = sol
                        best_neighbor_cost = cost
                        best_move_sig = sig
            
            if best_neighbor:
                current = best_neighbor
                reverse_sig = best_move_sig.replace('+','t').replace('-','+').replace('t','-')
                self.tabu_list[reverse_sig] = it + self.tabu_tenure
                if best_neighbor_cost < best_cost:
                    best = best_neighbor
                    best_cost = best_neighbor_cost
        return best

# =============================================================================
# 4. STAGE 2: ALNS (Staff Hiring)
# =============================================================================
class ALNSStaffing:
    def __init__(self, bed_config, demand_profile, iterations=500):
        self.beds = bed_config
        self.demand = demand_profile
        self.iter = iterations
        self.temp = 10000 
        self.cooling = 0.99
    
    def _calc_staff_cost(self, p_counts, n_count):
        sal_cost = (n_count * C_SALARY_NURSE) + sum(p_counts[s] * C_SALARY_PHY[s] for s in SPECIALTIES)
        shortage_pen = 0
        DOCS_NEEDED_FOR_247 = 4.2 
        
        for i in ICU_TYPES:
            if self.beds[i] > 0:
                required_specs = REQ_SPECS_PER_ICU.get(i, [])
                for spec in required_specs:
                    if p_counts[spec] < DOCS_NEEDED_FOR_247:
                        shortage_pen += (DOCS_NEEDED_FOR_247 - p_counts[spec]) * 50000

        total_nurse_shifts = n_count * 5
        shifts_req_n = 0
        for i in ICU_TYPES:
            d = min(self.demand[i], self.beds[i]) 
            shifts_req_n += math.ceil(d * RATIOS[i]) * 21 
            
        if total_nurse_shifts < shifts_req_n:
            shortage_pen += (shifts_req_n - total_nurse_shifts) * 5000
        return sal_cost + shortage_pen

    def destroy(self, current_p, current_n):
        new_p = current_p.copy()
        new_n = current_n
        if random.random() < 0.5:
            s = random.choice(SPECIALTIES)
            if new_p[s] > MIN_PHY_PER_SPEC[s]: new_p[s] -= 1
        else:
            if new_n > 10: new_n -= 1
        return new_p, new_n

    def repair(self, partial_p, partial_n):
        new_p = partial_p.copy()
        new_n = partial_n
        if random.random() < 0.5:
            s = random.choice(SPECIALTIES)
            new_p[s] += 1
        else:
            new_n += 1
        return new_p, new_n

    def solve(self):
        curr_p = {s: 5 for s in SPECIALTIES}
        curr_n = 30
        curr_cost = self._calc_staff_cost(curr_p, curr_n)
        best_p, best_n, best_cost = curr_p.copy(), curr_n, curr_cost
        
        for i in range(self.iter):
            dest_p, dest_n = self.destroy(curr_p, curr_n)
            cand_p, cand_n = self.repair(dest_p, dest_n)
            cand_cost = self._calc_staff_cost(cand_p, cand_n)
            delta = cand_cost - curr_cost
            
            accept = False
            if delta < 0: accept = True
            elif random.random() < math.exp(-delta / self.temp): accept = True
            
            if accept:
                curr_p, curr_n, curr_cost = cand_p, cand_n, cand_cost
                if curr_cost < best_cost:
                    best_p, best_n, best_cost = curr_p.copy(), curr_n, curr_cost
            self.temp *= self.cooling
        return best_p, best_n

# =============================================================================
# 5. EXECUTION & STAGE 3 (GUROBI)
# =============================================================================
if __name__ == "__main__":
    print("\n--- Running Stage 1: Tabu Search ---")
    tabu = TabuSearchBeds(HEURISTIC_DEMAND)
    final_beds = tabu.solve()
    print(f"   Optimal Beds: {final_beds}")

    print("\n--- Running Stage 2: ALNS ---")
    alns = ALNSStaffing(final_beds, HEURISTIC_DEMAND)
    final_p_counts, final_n_count = alns.solve()
    print(f"   Optimal Hires: {final_n_count} Nurses")
    print(f"   Physicians: {final_p_counts}")

    print("\n--- Running Stage 3: Gurobi Schedule Generation ---")
    model = gp.Model("Final_Schedule_Generation")

    # VARIABLES
    is_hired = model.addVars(PHYSICIANS + NURSES, vtype=GRB.BINARY, name="Hired")
    x = model.addVars(PHYSICIANS, DAYS, SHIFTS, ICU_TYPES, vtype=GRB.BINARY, name="x_phy")
    y = model.addVars(NURSES, DAYS, SHIFTS, ICU_TYPES, vtype=GRB.BINARY, name="y_nurse")
    U_phy = model.addVars(ICU_TYPES, SPECIALTIES, DAYS, SHIFTS, vtype=GRB.CONTINUOUS, name="U_phy")
    U_nurse = model.addVars(ICU_TYPES, DAYS, SHIFTS, vtype=GRB.CONTINUOUS, name="U_nurse")
    O_phy = model.addVars(PHYSICIANS, vtype=GRB.CONTINUOUS, name="O_phy")
    O_nurse = model.addVars(NURSES, vtype=GRB.CONTINUOUS, name="O_nurse")
    Z_blocked = model.addVars(ICU_TYPES, DAYS, SHIFTS, vtype=GRB.CONTINUOUS, name="BlockedPatients")

    Imbalance_Night_Phy = model.addVar(vtype=GRB.CONTINUOUS)
    Imbalance_Night_Nurse = model.addVar(vtype=GRB.CONTINUOUS)
    Imbalance_Week_Phy = model.addVar(vtype=GRB.CONTINUOUS)
    Imbalance_Week_Nurse = model.addVar(vtype=GRB.CONTINUOUS)
    MaxN_P, MinN_P = model.addVar(), model.addVar()
    MaxN_N, MinN_N = model.addVar(), model.addVar()
    MaxW_P, MinW_P = model.addVar(), model.addVar()
    MaxW_N, MinW_N = model.addVar(), model.addVar()

    # OBJECTIVE
    cost_cap = (sum(C_BED[i] * final_beds[i] for i in ICU_TYPES) + 
                gp.quicksum(SALARY_MAP[p] * is_hired[p] for p in PHYSICIANS + NURSES))
    cost_op = (gp.quicksum(C_OT_PHY * O_phy[p] for p in PHYSICIANS) +
               gp.quicksum(C_OT_NURSE * O_nurse[n] for n in NURSES) +
               C_SHORTAGE * (U_phy.sum() + U_nurse.sum()))
    cost_qual = C_DIVERT * Z_blocked.sum()
    cost_fair = (ALPHA['Night_Phy'] * Imbalance_Night_Phy + ALPHA['Night_Nurse'] * Imbalance_Night_Nurse +
                 ALPHA['Week_Phy'] * Imbalance_Week_Phy + ALPHA['Week_Nurse'] * Imbalance_Week_Nurse)
    model.setObjective(cost_cap + cost_op + cost_qual + cost_fair, GRB.MINIMIZE)

    # CONSTRAINTS
    # 1. Fix Hires
    model.addConstr(gp.quicksum(is_hired[n] for n in NURSES) == final_n_count)
    for spec in SPECIALTIES:
        relevant_docs = [p for p in PHYSICIANS if PHY_SPEC_MAP[p] == spec]
        target = min(final_p_counts[spec], len(relevant_docs))
        model.addConstr(gp.quicksum(is_hired[p] for p in relevant_docs) == target)

    # 2. Blocked Patients
    for i in ICU_TYPES:
        for d in DAYS:
            for s in SHIFTS:
                model.addConstr(Z_blocked[i,d,s] >= DEMAND_LOOKUP[(i,d,s)] - final_beds[i])

    # 3. Regulated Coverage
    for i in ICU_TYPES:
        for d in DAYS:
            for s in SHIFTS:
                req_specs = REQ_SPECS_PER_ICU.get(i, [])
                for spec in SPECIALTIES:
                    relevant_docs = [p for p in PHYSICIANS if PHY_SPEC_MAP[p] == spec]
                    if spec in req_specs:
                        model.addConstr(gp.quicksum(x[p,d,s,i] for p in relevant_docs) + U_phy[i,spec,d,s] >= 1)
                model.addConstr(y.sum('*',d,s,i) + U_nurse[i,d,s] >= NREQ_LOOKUP[(i,d,s)])

    # 4. Regulations
    for p in PHYSICIANS: model.addConstr(x.sum(p,'*','*','*') <= 100 * is_hired[p])
    for n in NURSES: model.addConstr(y.sum(n,'*','*','*') <= 100 * is_hired[n])
    for d in DAYS:
        for s in SHIFTS:
            for p in PHYSICIANS: model.addConstr(x.sum(p,d,s,'*') <= 1)
            for n in NURSES: model.addConstr(y.sum(n,d,s,'*') <= 1)
    for d in DAYS[:-1]:
        for p in PHYSICIANS: model.addConstr(x.sum(p,d,'Night','*') + x.sum(p,d+1,'Day','*') <= 1)
        for n in NURSES: model.addConstr(y.sum(n,d,'Night','*') + y.sum(n,d+1,'Day','*') <= 1)
    for p in PHYSICIANS:
        model.addConstr(8 * x.sum(p,'*','*','*') <= 40 + O_phy[p])
        model.addConstr(O_phy[p] <= OT_MAX)
        model.addConstr(x.sum(p,'*','Night','*') <= NS_MAX_PHY)
    for n in NURSES:
        model.addConstr(8 * y.sum(n,'*','*','*') <= 40 + O_nurse[n])
        model.addConstr(O_nurse[n] <= OT_MAX)
        model.addConstr(y.sum(n,'*','Night','*') <= NS_MAX_NURSE)

    # 5. Fairness
    Night_P = {p: x.sum(p,'*','Night','*') for p in PHYSICIANS}
    Night_N = {n: y.sum(n,'*','Night','*') for n in NURSES}
    Week_P = {p: gp.quicksum(x[p,d,s,i] for d in WEEKEND_DAYS for s in SHIFTS for i in ICU_TYPES) for p in PHYSICIANS}
    Week_N = {n: gp.quicksum(y[n,d,s,i] for d in WEEKEND_DAYS for s in SHIFTS for i in ICU_TYPES) for n in NURSES}

    for p in PHYSICIANS:
        model.addConstr(MaxN_P >= Night_P[p])
        model.addConstr(MinN_P <= Night_P[p] + BIG_M * (1 - is_hired[p]))
        model.addConstr(MaxW_P >= Week_P[p])
        model.addConstr(MinW_P <= Week_P[p] + BIG_M * (1 - is_hired[p]))
    for n in NURSES:
        model.addConstr(MaxN_N >= Night_N[n])
        model.addConstr(MinN_N <= Night_N[n] + BIG_M * (1 - is_hired[n]))
        model.addConstr(MaxW_N >= Week_N[n])
        model.addConstr(MinW_N <= Week_N[n] + BIG_M * (1 - is_hired[n]))

    model.addConstr(Imbalance_Night_Phy >= MaxN_P - MinN_P)
    model.addConstr(Imbalance_Week_Phy >= MaxW_P - MinW_P)
    model.addConstr(Imbalance_Night_Nurse >= MaxN_N - MinN_N)
    model.addConstr(Imbalance_Week_Nurse >= MaxW_N - MinW_N)

    # SOLVE
    print("\n--- Solving Final Schedule ---")
    model.optimize()

    if model.status == GRB.OPTIMAL:
        print(f"\n✅ SUCCESS! Total Cost: {model.objVal:,.2f} ₺")
        print(f"   Beds Used: {final_beds}")
        
        # Check Shortages
        total_short = sum(v.X for v in U_phy.values()) + sum(v.X for v in U_nurse.values())
        if total_short > 0:
            print(f"   ⚠️  Warning: Schedule relies on {total_short:.1f} shortage incidents.")
        else:
            print("   ✅ Full Coverage Achieved (No Shortages).")
            
    elif model.status == GRB.INFEASIBLE:
        print("\n❌ INFEASIBLE. Analyzing Gaps...")
        model.feasRelaxS(0, True, False, True)
        model.optimize()
        v_hours = 0
        for c in model.getConstrs():
            if c.Violation > 1e-4:
                if "x_phy" in c.ConstrName or "U_phy" in c.ConstrName: v_hours += c.Violation * 8
        fte_needed = int(np.ceil(v_hours / 40))
        print(f"   Gap Analysis: You need approx {fte_needed} more Physicians (FTE).")