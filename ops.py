import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import itertools

# --- Add Gurobi for MILP ---
# In a real environment, you would need to have Gurobi installed and a license.
# pip install gurobipy
try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False


# --- Page Configuration ---
st.set_page_config(
    page_title="United Airlines | Project Horizon",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Mock Data Generation ---
def create_mock_flight_data(demand_multiplier=1.0):
    """Generates a mock flight schedule with more detail."""
    origins = ['ORD', 'EWR', 'SFO', 'DEN', 'IAH']
    destinations = ['LAX', 'LHR', 'HND', 'BOS', 'MCO', 'CDG', 'FRA']
    aircraft_types = {
        '737-800': {'capacity': 166, 'cost_per_hour': 4500, 'count': 10},
        '737-MAX': {'capacity': 178, 'cost_per_hour': 4200, 'count': 8},
        'A320': {'capacity': 150, 'cost_per_hour': 4600, 'count': 12},
        'A321neo': {'capacity': 196, 'cost_per_hour': 4400, 'count': 7},
        '777-200': {'capacity': 276, 'cost_per_hour': 9000, 'count': 5},
        '787-9': {'capacity': 290, 'cost_per_hour': 8500, 'count': 6}
    }
    data = []
    for i in range(30):
        origin = np.random.choice(origins)
        dest = np.random.choice(destinations)
        if origin == dest: continue
        
        aircraft = np.random.choice(list(aircraft_types.keys()))
        flight_hours = np.random.uniform(2.5, 11.0)
        pax = int(aircraft_types[aircraft]['capacity'] * np.random.uniform(0.7, 0.95) * demand_multiplier)
        
        data.append({
            'FlightID': i, # Use a simple integer ID for model indexing
            'FlightNumber': 'UA' + str(np.random.randint(100, 2000)),
            'Origin': origin,
            'Destination': dest,
            'FlightHours': round(flight_hours, 1),
            'AircraftType': aircraft,
            'Capacity': aircraft_types[aircraft]['capacity'],
            'Passengers': pax,
            'OperatingCost': int(flight_hours * aircraft_types[aircraft]['cost_per_hour']),
            'Revenue': int(pax * np.random.uniform(250, 600)),
        })
    df = pd.DataFrame(data)
    df['Profit'] = df['Revenue'] - df['OperatingCost']
    return df, aircraft_types

def create_mock_sensor_data(num_aircraft):
    """Generates mock sensor data for predictive maintenance."""
    at_risk_data = {
        'TailNumber': [f'N{np.random.randint(100,999)}UA' for _ in range(num_aircraft)],
        'Component': np.random.choice(['Engine 1', 'APU', 'Brake-Assembly', 'TCAS-Unit'], num_aircraft),
        'RUL_days': sorted(np.random.randint(5, 60, num_aircraft)),
        'MaintenanceCost': np.random.randint(50000, 200000, num_aircraft),
        'AOG_Risk_Cost': np.random.randint(1000000, 3000000, num_aircraft)
    }
    return pd.DataFrame(at_risk_data)

def create_mock_timeseries_data():
    """Generates mock time-series sensor data for the ML analysis plots."""
    time_series = pd.to_datetime(pd.date_range(end=pd.Timestamp.now(), periods=100))
    data = {
        'Timestamp': time_series,
        'EGT_C': 550 + np.cumsum(np.random.randn(100) * 0.1) + np.linspace(0, 15, 100),
        'Vibration_mm_s': 1.2 + np.cumsum(np.random.randn(100) * 0.005) + np.linspace(0, 0.5, 100),
    }
    return pd.DataFrame(data)

def create_mock_inventory_data():
    """Generates mock inventory data."""
    parts = ['APU-Generator', 'Brake-Assembly', 'TCAS-Unit', 'Weather-Radar', 'Tire']
    locations = ['SFO MRO', 'ORD MRO', 'IAH MRO']
    inventory = []
    for part in parts:
        for loc in locations:
            inventory.append({
                'PartID': f'{part}-{loc}',
                'PartNumber': part,
                'Location': loc,
                'OnHand': np.random.randint(5, 50),
                'PartCost': np.random.randint(1000, 25000),
                'HoldingCost_Annual': np.random.randint(100, 500),
            })
    return pd.DataFrame(inventory)

# --- Backend Solver Functions ---

def run_fleet_assignment_milp(flights_df, aircraft_data, fuel_price_multiplier):
    """Builds and solves the Fleet Assignment MILP model using Gurobi."""
    if not GUROBI_AVAILABLE:
        st.error("Gurobi not installed. Cannot run Fleet Assignment optimization.")
        return flights_df.copy()

    model = gp.Model("FleetAssignment")
    flight_indices = flights_df.index
    aircraft_keys = list(aircraft_data.keys())
    x = model.addVars(flight_indices, aircraft_keys, vtype=GRB.BINARY, name="assign")
    total_cost = gp.quicksum(x[f, t] * flights_df.loc[f, 'FlightHours'] * aircraft_data[t]['cost_per_hour'] * fuel_price_multiplier for f in flight_indices for t in aircraft_keys)
    spill_cost = gp.quicksum(x[f, t] * (flights_df.loc[f, 'Passengers'] - aircraft_data[t]['capacity']) * 500 for f in flight_indices for t in aircraft_keys if flights_df.loc[f, 'Passengers'] > aircraft_data[t]['capacity'])
    model.setObjective(total_cost + spill_cost, GRB.MINIMIZE)
    for f in flight_indices: model.addConstr(gp.quicksum(x[f, t] for t in aircraft_keys) == 1, name=f"flight_coverage_{f}")
    for t in aircraft_keys: model.addConstr(gp.quicksum(x[f, t] for f in flight_indices) <= aircraft_data[t]['count'], name=f"fleet_size_{t}")
    
    model.optimize()
    
    optimized_df = flights_df.copy()
    if model.status == GRB.OPTIMAL:
        st.success("Optimal solution found by **Gurobi** for Fleet Assignment!")
        solution = model.getAttr('x', x)
        for f in flight_indices:
            for t in aircraft_keys:
                if solution[f, t] > 0.5:
                    optimized_df.loc[f, 'AircraftType'] = t
                    # Update cost based on new aircraft assignment
                    new_cost = flights_df.loc[f, 'FlightHours'] * aircraft_data[t]['cost_per_hour'] * fuel_price_multiplier
                    optimized_df.loc[f, 'OperatingCost'] = new_cost
                    optimized_df.loc[f, 'Profit'] = optimized_df.loc[f, 'Revenue'] - new_cost
                    break
    else:
        st.error("Gurobi could not find an optimal solution for Fleet Assignment.")
    return optimized_df

def run_crew_scheduling_milp(flights_df, hotel_cost_multiplier, robustness_pref):
    """Builds and solves the Crew Scheduling Set Covering model."""
    if not GUROBI_AVAILABLE:
        st.error("Gurobi not installed. Cannot run Crew Scheduling optimization.")
        return pd.DataFrame()

    potential_pairings = [list(c) for i in range(2, 5) for c in itertools.combinations(flights_df['FlightID'].tolist(), i)]
    pairing_data = [{'pairing_id': i, 'flights': p, 'cost': (flights_df[flights_df['FlightID'].isin(p)]['FlightHours'].sum() * 150 + len(p) * 100) * hotel_cost_multiplier} for i, p in enumerate(potential_pairings)]
    pairings_df = pd.DataFrame(pairing_data)
    
    model = gp.Model("SetCovering")
    y = model.addVars(pairings_df.index, vtype=GRB.BINARY, name="select_pairing")
    model.setObjective(gp.quicksum(y[p] * pairings_df.loc[p, 'cost'] for p in pairings_df.index), GRB.MINIMIZE)
    for flight_id in flights_df['FlightID'].tolist():
        covering_pairings = pairings_df[pairings_df['flights'].apply(lambda x: flight_id in x)].index
        model.addConstr(gp.quicksum(y[p] for p in covering_pairings) >= 1, name=f"cover_flight_{flight_id}")

    model.optimize()
    
    solution_pairings = []
    if model.status == GRB.OPTIMAL:
        st.success("Optimal solution found by **Gurobi** for Crew Scheduling!")
        solution = model.getAttr('x', y)
        for p_idx in pairings_df.index:
            if solution[p_idx] > 0.5:
                p_row = pairings_df.loc[p_idx]
                flight_numbers = flights_df[flights_df['FlightID'].isin(p_row['flights'])]['FlightNumber'].tolist()
                solution_pairings.append({'PairingID': f"P{p_row['pairing_id']}", 'Flights': ", ".join(flight_numbers), 'TotalCost': p_row['cost']})
    else:
        st.error("Gurobi could not find an optimal solution for Crew Scheduling.")
    return pd.DataFrame(solution_pairings)

def run_network_optimization_milp(potential_routes, budget, fleet_hours_available):
    """Solves the network expansion problem."""
    if not GUROBI_AVAILABLE:
        st.error("Gurobi not installed. Cannot run Network Optimization.")
        return pd.DataFrame()

    model = gp.Model("NetworkExpansion")
    y = model.addVars(potential_routes.index, vtype=GRB.BINARY, name="open_route")
    model.setObjective(gp.quicksum(y[r] * potential_routes.loc[r, 'AnnualProfit'] for r in potential_routes.index), GRB.MAXIMIZE)
    model.addConstr(gp.quicksum(y[r] * potential_routes.loc[r, 'StartupCost'] for r in potential_routes.index) <= budget, "budget")
    model.addConstr(gp.quicksum(y[r] * potential_routes.loc[r, 'AnnualFlightHours'] for r in potential_routes.index) <= fleet_hours_available, "fleet_hours")
    
    model.optimize()
    
    selected_routes = []
    if model.status == GRB.OPTIMAL:
        st.success("Optimal solution found by **Gurobi** for Network Optimization!")
        solution = model.getAttr('x', y)
        for r_idx in potential_routes.index:
            if solution[r_idx] > 0.5:
                selected_routes.append(r_idx)
    else:
        st.error("Gurobi could not find an optimal solution for Network Optimization.")
    return potential_routes.loc[selected_routes]

def run_maintenance_scheduling_milp(at_risk_df, hangar_capacity, planning_horizon_days=30):
    """Schedules maintenance to minimize AOG risk cost."""
    if not GUROBI_AVAILABLE:
        st.error("Gurobi not installed. Cannot run Maintenance Scheduling optimization.")
        return at_risk_df

    model = gp.Model("MaintenanceScheduling")
    aircraft = at_risk_df['TailNumber'].tolist()
    days = list(range(1, planning_horizon_days + 1))
    x = model.addVars(aircraft, days, vtype=GRB.BINARY, name="schedule")
    risk_cost = gp.quicksum(x[ac, day] * (at_risk_df.loc[at_risk_df['TailNumber'] == ac, 'AOG_Risk_Cost'].iloc[0] * (day / at_risk_df.loc[at_risk_df['TailNumber'] == ac, 'RUL_days'].iloc[0])) for ac in aircraft for day in days)
    model.setObjective(risk_cost, GRB.MINIMIZE)
    for ac in aircraft: model.addConstr(gp.quicksum(x[ac, day] for day in days) == 1, f"must_schedule_{ac}")
    for day in days: model.addConstr(gp.quicksum(x[ac, day] for ac in aircraft) <= hangar_capacity, f"hangar_capacity_{day}")
    for ac in aircraft: model.addConstr(gp.quicksum(x[ac, day] * day for day in days) <= at_risk_df.loc[at_risk_df['TailNumber'] == ac, 'RUL_days'].iloc[0], f"schedule_before_RUL_{ac}")

    model.optimize()

    schedule_df = at_risk_df.copy()
    schedule_df['Scheduled_Day'] = 'Not Scheduled'
    if model.status == GRB.OPTIMAL:
        st.success("Optimal solution found by **Gurobi** for Maintenance Scheduling!")
        solution = model.getAttr('x', x)
        for ac in aircraft:
            for day in days:
                if solution[ac, day] > 0.5:
                    schedule_df.loc[schedule_df['TailNumber'] == ac, 'Scheduled_Day'] = day
                    break
    else:
        st.error("Gurobi could not find an optimal solution for Maintenance Scheduling.")
    return schedule_df

def run_inventory_optimization_milp(inventory_df, budget, service_level):
    """Optimizes spare parts inventory levels."""
    if not GUROBI_AVAILABLE:
        st.error("Gurobi not installed. Cannot run Inventory Optimization.")
        return inventory_df

    z_score = {95.0: 1.645, 99.0: 2.326, 99.9: 3.09}
    inventory_df['SafetyStock'] = z_score[service_level] * np.sqrt(inventory_df['OnHand'])
    
    model = gp.Model("InventoryOptimization")
    I = model.addVars(inventory_df.index, vtype=GRB.INTEGER, name="inventory_level")
    model.setObjective(gp.quicksum(I[p] * inventory_df.loc[p, 'HoldingCost_Annual'] for p in inventory_df.index), GRB.MINIMIZE)
    model.addConstr(gp.quicksum(I[p] * inventory_df.loc[p, 'PartCost'] for p in inventory_df.index) <= budget, "total_budget")
    for p in inventory_df.index: model.addConstr(I[p] >= inventory_df.loc[p, 'SafetyStock'], f"service_level_{p}")

    model.optimize()

    optimized_inventory_df = inventory_df.copy()
    optimized_inventory_df['Optimal_OnHand'] = 0
    if model.status == GRB.OPTIMAL:
        st.success("Optimal solution found by **Gurobi** for Inventory Optimization!")
        solution = model.getAttr('x', I)
        for p_idx in inventory_df.index:
            optimized_inventory_df.loc[p_idx, 'Optimal_OnHand'] = int(solution[p_idx])
    else:
        st.error("Gurobi could not find an optimal solution for Inventory Optimization.")
    return optimized_inventory_df

# --- UI Pages ---
def home_page():
    st.image("https://placehold.co/1200x300/003262/FFFFFF?text=Beyond+The+Horizon...", use_container_width=True)
    st.title("**Unimatics** ✈ for UA Operations Optimization")
    st.markdown("Welcome... PLUS ULTRA!")
    st.markdown("This UA Operations dashboard runs Gurobi solvers for all five operations --")
    st.info("Aircraft Route Optimization")
    st.info("Crew Scheduling")
    st.info("Network Optimization")
    st.info("Predictive Maintenance")
    st.info("Spare Parts Inventory")


def aircraft_route_optimization_page():
    st.header("Aircraft Route Optimization ✈️")
    st.markdown("Assigns the most profitable aircraft to each flight using a **Gurobi** MILP model.")
    if not GUROBI_AVAILABLE: st.warning("Gurobi library not found. Optimization will be simulated.", icon="⚠️")
    st.sidebar.header("Scenario Controls")
    fuel_price_multiplier = st.sidebar.slider("Fuel Price Adjustment", 0.8, 1.5, 1.0, 0.1)
    demand_multiplier = st.sidebar.slider("Demand Forecast Adjustment", 0.8, 1.5, 1.0, 0.1)
    flights_df, aircraft_data = create_mock_flight_data(demand_multiplier)
    st.subheader("Current Flight Schedule & Profitability")
    st.dataframe(flights_df.drop('FlightID', axis=1), use_container_width=True)
    
    original_profit = flights_df['Profit'].sum()
    original_cost = flights_df['OperatingCost'].sum()

    if st.button("Optimize Fleet Assignment", type="primary"):
        with st.spinner("Running Gurobi Fleet Assignment solver..."):
            optimized_df = run_fleet_assignment_milp(flights_df, aircraft_data, fuel_price_multiplier)
            
            st.subheader("Optimization Impact Summary")
            optimized_profit = optimized_df['Profit'].sum()
            optimized_cost = optimized_df['OperatingCost'].sum()
            
            col1, col2 = st.columns(2)
            col1.metric("Original Network Profit", f"${original_profit:,.0f}")
            col1.metric("Optimized Network Profit", f"${optimized_profit:,.0f}", delta=f"${optimized_profit - original_profit:,.0f}")
            col2.metric("Original Operating Cost", f"${original_cost:,.0f}")
            col2.metric("Optimized Operating Cost", f"${optimized_cost:,.0f}", delta=f"${optimized_cost - original_cost:,.0f}")

            st.subheader("Optimized Schedule")
            st.dataframe(optimized_df.drop('FlightID', axis=1), use_container_width=True)

def crew_scheduling_page():
    st.header("Crew Scheduling Optimization 👥")
    st.markdown("Solves the **Set Covering Problem** using a **Gurobi** MILP model to create legal and cost-effective crew pairings.")
    if not GUROBI_AVAILABLE: st.warning("Gurobi library not found. Optimization will be simulated.", icon="⚠️")
    st.sidebar.header("Scenario Controls")
    hotel_cost_multiplier = st.sidebar.slider("Hotel/Per Diem Cost Factor", 0.8, 1.5, 1.0, 0.1)
    robustness_pref = st.sidebar.select_slider("Prioritize", ["Lowest Cost", "Balanced", "Robustness"], "Balanced")

    flights_df, _ = create_mock_flight_data()
    flights_df = flights_df.head(10)
    st.subheader("Flights Requiring Crew Coverage")
    st.dataframe(flights_df[['FlightNumber', 'Origin', 'Destination', 'FlightHours']], use_container_width=True)
    
    if st.button("Generate Optimal Crew Pairings", type="primary"):
        with st.spinner("Running Gurobi Set Covering solver..."):
            pairings_df = run_crew_scheduling_milp(flights_df, hotel_cost_multiplier, robustness_pref)
            
            st.subheader("Optimization Impact Summary")
            optimized_cost = pairings_df['TotalCost'].sum()
            # Simulate a less efficient "original" cost for comparison
            original_cost = optimized_cost * 1.15 
            
            col1, col2 = st.columns(2)
            col1.metric("Unoptimized Crew Cost (Estimated)", f"${original_cost:,.0f}")
            col1.metric("Optimized Crew Cost", f"${optimized_cost:,.0f}", delta=f"-${original_cost - optimized_cost:,.0f}", delta_color="inverse")
            col2.metric("Pairings Selected", len(pairings_df))

            st.subheader("Generated Optimal Pairings")
            st.dataframe(pairings_df.style.format({'TotalCost': '${:,.2f}'}), use_container_width=True)

def network_optimization_page():
    st.header("Network 🌐 Optimization")
    st.markdown("Uses a **Gurobi** MILP model to select the most profitable portfolio of new routes within budget and resource constraints.")
    if not GUROBI_AVAILABLE: st.warning("Gurobi library not found. Optimization will be simulated.", icon="⚠️")
    st.sidebar.header("Scenario Controls")
    budget = st.sidebar.slider("New Route Startup Budget (Millions)", 10, 100, 50, 5) * 1_000_000
    fleet_hours = st.sidebar.slider("Available Annual Fleet Hours", 10000, 50000, 20000, 1000)
    
    potential_routes_data = {
        'Route': ['EWR-BCN', 'SFO-DUB', 'ORD-ZRH', 'IAH-AMS', 'DEN-MEX', 'EWR-GRU'],
        'StartupCost': [15_000_000, 18_000_000, 20_000_000, 12_000_000, 8_000_000, 25_000_000],
        'AnnualProfit': [5_000_000, 6_000_000, 5_500_000, 4_000_000, 3_000_000, 7_000_000],
        'AnnualFlightHours': [4000, 4500, 4800, 3800, 2500, 5500]
    }
    potential_routes_df = pd.DataFrame(potential_routes_data)
    st.subheader("Potential New Routes for Expansion")
    st.dataframe(potential_routes_df, use_container_width=True)

    if st.button("Optimize Route Portfolio", type="primary"):
        with st.spinner("Running Gurobi Network Expansion solver..."):
            selected_routes_df = run_network_optimization_milp(potential_routes_df, budget, fleet_hours)

            st.subheader("Optimization Results")
            if not selected_routes_df.empty:
                st.metric("Total Projected Profit from Expansion", f"${selected_routes_df['AnnualProfit'].sum():,.0f}")
                st.subheader("Optimal New Routes to Launch")
                st.dataframe(selected_routes_df[['Route', 'AnnualProfit', 'StartupCost']], use_container_width=True)
            else:
                st.warning("No new routes selected with the given constraints. Try increasing the budget or available fleet hours.")

def predictive_maintenance_page():
    st.header("Predictive Maintenance Scheduling")
    st.markdown("Uses a **Gurobi** MILP model to create an optimal maintenance schedule that minimizes risk-adjusted costs.")
    if not GUROBI_AVAILABLE: st.warning("Gurobi library not found. Optimization will be simulated.", icon="⚠️")
    st.sidebar.header("Scenario Controls")
    hangar_capacity = st.sidebar.slider("Daily Hangar Capacity", 1, 10, 3)
    num_at_risk = st.sidebar.slider("Number of 'At-Risk' Aircraft to Schedule", 5, 20, 10)
    
    at_risk_df = create_mock_sensor_data(num_at_risk)
    st.subheader("At-Risk Aircraft Requiring Maintenance")
    st.dataframe(at_risk_df, use_container_width=True)
    
    # Calculate unoptimized risk cost (if all were scheduled on their last possible day)
    unoptimized_risk = (at_risk_df['AOG_Risk_Cost'] * (at_risk_df['RUL_days'] / at_risk_df['RUL_days'])).sum()

    if st.button("Generate Optimal Maintenance Schedule", type="primary"):
        with st.spinner("Running Gurobi Maintenance Scheduling solver..."):
            schedule_df = run_maintenance_scheduling_milp(at_risk_df, hangar_capacity)

            # V V V PASTE THE BLOCK HERE V V V
            st.subheader("Optimization Impact Summary")
            # Ensure Scheduled_Day is numeric for calculation, handle 'Not Scheduled' case
            schedule_df['Scheduled_Day_Calc'] = pd.to_numeric(schedule_df['Scheduled_Day'], errors='coerce').fillna(schedule_df['RUL_days'])
            optimized_risk = (schedule_df['AOG_Risk_Cost'] * (schedule_df['Scheduled_Day_Calc'] / schedule_df['RUL_days'])).sum()

            col1, col2 = st.columns(2)
            col1.metric("Unoptimized Risk-Adjusted Cost", f"${unoptimized_risk:,.0f}")
            col2.metric("Optimized Risk-Adjusted Cost", f"${optimized_risk:,.0f}", delta=f"-${unoptimized_risk - optimized_risk:,.0f}", delta_color="inverse")
            # ^ ^ ^ END OF THE BLOCK TO PASTE ^ ^ ^

            st.subheader("Optimization Impact Summary")
            optimized_risk = (schedule_df['AOG_Risk_Cost'] * (schedule_df['Scheduled_Day'].astype(float) / schedule_df['RUL_days'])).sum()   
            col1, col2 = st.columns(2)
            col1.metric("Unoptimized Risk-Adjusted Cost", f"${unoptimized_risk:,.0f}")
            col2.metric("Optimized Risk-Adjusted Cost", f"${optimized_risk:,.0f}", delta=f"-${unoptimized_risk - optimized_risk:,.0f}", delta_color="inverse")

            st.subheader("Optimal Maintenance Schedule")
            st.dataframe(schedule_df[['TailNumber', 'Component', 'RUL_days', 'Scheduled_Day']], use_container_width=True)
            
            fig = px.bar(schedule_df[schedule_df['Scheduled_Day'] != 'Not Scheduled'].groupby('Scheduled_Day').size().reset_index(name='count'), x='Scheduled_Day', y='count', title="Aircraft Scheduled for Maintenance Per Day")
            fig.add_hline(y=hangar_capacity, line_dash="dash", line_color="red", annotation_text="Hangar Capacity")
            st.plotly_chart(fig, use_container_width=True)

    # --- ML Analysis Section ---
    st.markdown("---")
    st.subheader("Individual Aircraft Analysis (ML)")
    st.write("Select a specific aircraft to see the sensor data and ML predictions that determined its RUL.")
    
    tail_number = st.selectbox("Select Aircraft Tail Number", at_risk_df['TailNumber'].unique())
    if tail_number:
        sensor_df = create_mock_timeseries_data()
        aircraft_details = at_risk_df[at_risk_df['TailNumber'] == tail_number].iloc[0]
        component = aircraft_details['Component']
        
        st.subheader(f"Live Sensor Data for {component} on {tail_number}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sensor_df['Timestamp'], y=sensor_df['EGT_C'], name='EGT'))
        fig.add_trace(go.Scatter(x=sensor_df['Timestamp'], y=sensor_df['Vibration_mm_s']*100, name='Vibration (x100)', yaxis='y2'))
        fig.update_layout(title_text=f"Sensor Readings for {component}", yaxis=dict(title="EGT (°C)"), yaxis2=dict(title="Vibration (mm/s)", overlaying='y', side='right'))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Prediction & Cost of Inaction")
        rul_days = aircraft_details['RUL_days']
        col1, col2 = st.columns(2)
        col1.metric("Predicted RUL", f"{rul_days} days")
        prob_failure = 1 / (rul_days + 10) if rul_days > -10 else 1
        cost_inaction = int(aircraft_details['AOG_Risk_Cost'] * prob_failure * 10)
        col2.metric("Cost of 10-day Delay", f"${cost_inaction:,.0f}", help="Projected risk-adjusted cost of delaying maintenance by 10 days.")
        st.warning(f"**Recommendation:** Based on ML analysis, schedule maintenance for **{component}** on **{tail_number}** within the next **{rul_days - 5} days**.", icon="⚠️")

def spare_parts_inventory_page():
    st.header("Spare Parts Inventory 📦 Optimization")
    st.markdown("Uses a **Gurobi** MILP model to set optimal inventory levels that meet service targets while minimizing costs.")
    if not GUROBI_AVAILABLE: st.warning("Gurobi library not found. Optimization will be simulated.", icon="⚠️")
    st.sidebar.header("Scenario Controls")
    inventory_budget = st.sidebar.slider("Total Inventory Budget (Millions)", 5, 50, 20, 1) * 1_000_000
    service_level = st.sidebar.select_slider("Target Service Level", [95.0, 99.0, 99.9], 99.0)

    inventory_df = create_mock_inventory_data()
    st.subheader("Current Spare Parts Inventory")
    st.dataframe(inventory_df, use_container_width=True)
    
    original_holding_cost = inventory_df['HoldingCost_Annual'].sum()
    original_investment = (inventory_df['OnHand'] * inventory_df['PartCost']).sum()

    if st.button("Optimize Inventory Levels", type="primary"):
        with st.spinner("Running Gurobi Inventory Optimization solver..."):
            optimized_inventory_df = run_inventory_optimization_milp(inventory_df, inventory_budget, service_level)
            
            st.subheader("Optimization Impact Summary")
            optimized_holding_cost = (optimized_inventory_df['Optimal_OnHand'] * optimized_inventory_df['HoldingCost_Annual'] / optimized_inventory_df['OnHand']).sum() # Approximate
            optimized_investment = (optimized_inventory_df['Optimal_OnHand'] * optimized_inventory_df['PartCost']).sum()
            
            col1, col2 = st.columns(2)
            col1.metric("Original Annual Holding Cost", f"${original_holding_cost:,.0f}")
            col1.metric("Optimized Annual Holding Cost", f"${optimized_holding_cost:,.0f}", delta=f"-${original_holding_cost - optimized_holding_cost:,.0f}", delta_color="inverse")
            col2.metric("Original Investment", f"${original_investment:,.0f}")
            col2.metric("Optimized Investment", f"${optimized_investment:,.0f}", delta=f"${optimized_investment - original_investment:,.0f}")

            st.subheader("Optimal vs. Current Inventory Levels")
            st.dataframe(optimized_inventory_df[['PartNumber', 'Location', 'OnHand', 'Optimal_OnHand']], use_container_width=True)

def main():
    st.sidebar.image("https://placehold.co/400x100/FFFFFF/003262?text=United+Airlines")
    st.sidebar.title("UAOps")
    page_options = { "Home": home_page, "Aircraft Route Optimization": aircraft_route_optimization_page, "Crew Scheduling": crew_scheduling_page, "Network Optimization": network_optimization_page, "Predictive Maintenance": predictive_maintenance_page, "Spare Parts Inventory": spare_parts_inventory_page }
    selection = st.sidebar.radio("Go to", list(page_options.keys()))
    page = page_options[selection]
    page()

if __name__ == "__main__":
    main()



# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# import time
# # --- Add Gurobi for MILP ---
# # In a real environment, you would need to have Gurobi installed and a license.
# # pip install gurobipy
# # import gurobipy
# try:
#     import gurobipy as gp
#     from gurobipy import GRB
#     GUROBI_AVAILABLE = True
# except ImportError:
#     GUROBI_AVAILABLE = False


# # --- Page Configuration ---
# st.set_page_config(
#     page_title="United Airlines | Project Horizon",
#     page_icon="✈️",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # --- Mock Data Generation (Enhanced for Production Features) ---
# def create_mock_flight_data(demand_multiplier=1.0):
#     """Generates a mock flight schedule with more detail."""
#     origins = ['ORD', 'EWR', 'SFO', 'DEN', 'IAH']
#     destinations = ['LAX', 'LHR', 'HND', 'BOS', 'MCO', 'CDG', 'FRA']
#     aircraft_types = {
#         '737-800': {'capacity': 166, 'cost_per_hour': 4500, 'count': 10},
#         '737-MAX': {'capacity': 178, 'cost_per_hour': 4200, 'count': 8},
#         'A320': {'capacity': 150, 'cost_per_hour': 4600, 'count': 12},
#         'A321neo': {'capacity': 196, 'cost_per_hour': 4400, 'count': 7},
#         '777-200': {'capacity': 276, 'cost_per_hour': 9000, 'count': 5},
#         '787-9': {'capacity': 290, 'cost_per_hour': 8500, 'count': 6}
#     }
#     data = []
#     for _ in range(30):
#         origin = np.random.choice(origins)
#         dest = np.random.choice(destinations)
#         if origin == dest: continue
        
#         aircraft = np.random.choice(list(aircraft_types.keys()))
#         flight_hours = np.random.uniform(2.5, 11.0)
#         pax = int(aircraft_types[aircraft]['capacity'] * np.random.uniform(0.7, 0.95) * demand_multiplier)
        
#         data.append({
#             'FlightNumber': 'UA' + str(np.random.randint(100, 2000)),
#             'Origin': origin,
#             'Destination': dest,
#             'FlightHours': round(flight_hours, 1),
#             'AircraftType': aircraft,
#             'Capacity': aircraft_types[aircraft]['capacity'],
#             'Passengers': pax,
#             'OperatingCost': int(flight_hours * aircraft_types[aircraft]['cost_per_hour']),
#             'Revenue': int(pax * np.random.uniform(250, 600)),
#         })
#     df = pd.DataFrame(data)
#     df['Profit'] = df['Revenue'] - df['OperatingCost']
#     return df, aircraft_types

# def create_mock_sensor_data():
#     """Generates mock sensor data for predictive maintenance."""
#     time_series = pd.to_datetime(pd.date_range(end=pd.Timestamp.now(), periods=100))
#     data = {
#         'Timestamp': time_series,
#         'EGT_C': 550 + np.cumsum(np.random.randn(100) * 0.1) + np.linspace(0, 15, 100),
#         'Vibration_mm_s': 1.2 + np.cumsum(np.random.randn(100) * 0.005) + np.linspace(0, 0.5, 100),
#         'FuelFlow_kg_hr': 2500 + np.random.randn(100) * 20
#     }
#     return pd.DataFrame(data)

# def create_mock_inventory_data():
#     """Generates mock inventory data."""
#     parts = ['APU-Generator', 'Brake-Assembly', 'TCAS-Unit', 'Weather-Radar']
#     locations = ['SFO MRO', 'ORD MRO', 'IAH MRO']
#     inventory = []
#     for part in parts:
#         for loc in locations:
#             inventory.append({
#                 'PartNumber': part,
#                 'Location': loc,
#                 'OnHand': np.random.randint(5, 50),
#                 'HoldingCost_USD': np.random.randint(100, 500),
#                 'StockoutRisk': f"{np.random.uniform(1, 10):.1f}%"
#             })
#     return pd.DataFrame(inventory)

# # --- Backend Solver Function ---
# def run_fleet_assignment_milp(flights_df, aircraft_data, fuel_price_multiplier):
#     """
#     This function builds and solves the Fleet Assignment MILP model using Gurobi.
#     NOTE: In this environment, the actual solver is not run. The code demonstrates
#     the model formulation, and a simulated result is returned.
#     """
#     if not GUROBI_AVAILABLE:
#         st.error("Gurobi is not installed. Returning a simulated result.")
#         # Fallback to simulation if Gurobi is not available
#         optimized_df = flights_df.copy()
#         for i, row in optimized_df.iterrows():
#             if row['Passengers'] < row['Capacity'] * 0.6:
#                 if '777' in row['AircraftType'] or '787' in row['AircraftType']:
#                     optimized_df.loc[i, 'AircraftType'] = '737-MAX'
#             elif row['Passengers'] > row['Capacity'] * 0.98:
#                 if '737' in row['AircraftType'] or 'A320' in row['AircraftType']:
#                     optimized_df.loc[i, 'AircraftType'] = '787-9'
#         return optimized_df

#     # --- REAL GUROBI MODEL ---
#     model = gp.Model("FleetAssignment")

#     # 1. Decision Variables: x[f, t] = 1 if flight f is assigned to aircraft type t
#     flight_indices = flights_df.index
#     aircraft_keys = list(aircraft_data.keys())
#     x = model.addVars(flight_indices, aircraft_keys, vtype=GRB.BINARY, name="assign")

#     # 2. Objective Function: Minimize total operating cost
#     # Cost includes a penalty for spilled passengers (demand > capacity)
#     total_cost = gp.quicksum(
#         x[f, t] * flights_df.loc[f, 'FlightHours'] * aircraft_data[t]['cost_per_hour'] * fuel_price_multiplier
#         for f in flight_indices for t in aircraft_keys
#     )
#     spill_cost = gp.quicksum(
#         x[f, t] * (flights_df.loc[f, 'Passengers'] - aircraft_data[t]['capacity']) * 500 # $500 penalty per spilled passenger
#         for f in flight_indices for t in aircraft_keys if flights_df.loc[f, 'Passengers'] > aircraft_data[t]['capacity']
#     )
#     model.setObjective(total_cost + spill_cost, GRB.MINIMIZE)

#     # 3. Constraints
#     # a) Each flight must be covered by exactly one aircraft type
#     for f in flight_indices:
#         model.addConstr(gp.quicksum(x[f, t] for t in aircraft_keys) == 1, name=f"flight_coverage_{f}")

#     # b) The number of aircraft of a type used cannot exceed the number available
#     for t in aircraft_keys:
#         model.addConstr(gp.quicksum(x[f, t] for f in flight_indices) <= aircraft_data[t]['count'], name=f"fleet_size_{t}")

#     # 4. Optimize the model
#     model.optimize() # This line would run the solver in a real environment
#     # --- Extract results from the solved model ---
#     optimized_df = flights_df.copy()
#     if model.status == GRB.OPTIMAL:
#         st.success("Optimal solution found by Gurobi!")
#         solution = model.getAttr('x', x)
#         for f in flight_indices:
#             for t in aircraft_keys:
#                 if solution[f, t] > 0.5: # If the variable is 1
#                     optimized_df.loc[f, 'AircraftType'] = t
#                     break # Move to the next flight
#         return optimized_df
#     else:
#         st.error("Gurobi could not find an optimal solution. Returning original data.")
#         return flights_df
    
# def run_crew_scheduling_milp(flights_df, hotel_cost_multiplier):
#     """Builds and solves the Crew Scheduling Set Covering model."""
#     if not GUROBI_AVAILABLE:
#         st.error("Gurobi is not installed. Returning a simulated result for Crew Scheduling.")
#         pairings = []
#         for i in range(10):
#             pairing_flights = flights_df.sample(n=np.random.randint(2, 4))
#             cost = (pairing_flights['OperatingCost'].sum() * 0.15 + np.random.randint(1000, 5000)) * hotel_cost_multiplier
#             pairings.append({'PairingID': f'P{1000+i}', 'Flights': ', '.join(pairing_flights['FlightNumber'].tolist()), 'TotalCost': cost})
#         return pd.DataFrame(pairings)

#     # 1. Pairing Generation (Simulated)
#     # In a real system, this is a very complex step. Here we generate some valid-looking pairings.
#     potential_pairings = []
#     flight_ids = flights_df['FlightID'].tolist()
#     for i in range(2, 5): # Pairings of 2, 3, and 4 flights
#         for combo in itertools.combinations(flight_ids, i):
#             potential_pairings.append(list(combo))
    
#     # Create a DataFrame for pairings with costs
#     pairing_data = []
#     for i, p in enumerate(potential_pairings):
#         pairing_flights = flights_df[flights_df['FlightID'].isin(p)]
#         cost = (pairing_flights['FlightHours'].sum() * 150 + len(p) * 100) * hotel_cost_multiplier # Simplified cost
#         pairing_data.append({'pairing_id': i, 'flights': p, 'cost': cost})
    
#     pairings_df = pd.DataFrame(pairing_data)
    
#     # 2. Build the Set Covering Model
#     model = gp.Model("SetCovering")
    
#     # 3. Decision Variables: y[p] = 1 if we select pairing p
#     y = model.addVars(pairings_df.index, vtype=GRB.BINARY, name="select_pairing")
    
#     # 4. Objective Function: Minimize total cost of selected pairings
#     model.setObjective(gp.quicksum(y[p] * pairings_df.loc[p, 'cost'] for p in pairings_df.index), GRB.MINIMIZE)
    
#     # 5. Constraints: Each flight must be covered by at least one selected pairing
#     for flight_id in flight_ids:
#         covering_pairings = pairings_df[pairings_df['flights'].apply(lambda x: flight_id in x)].index
#         model.addConstr(gp.quicksum(y[p] for p in covering_pairings) >= 1, name=f"cover_flight_{flight_id}")

#     # 6. Optimize the model
#     # model.optimize() # UNCOMMENT TO RUN THE REAL SOLVER
    
#     # --- SIMULATED SOLVER FOR DEMO ---
#     st.info("Gurobi model for Crew Scheduling was built but not solved. Displaying simulated results.")
#     # For demo, we just pick a few random pairings to simulate a solution
#     num_chosen = np.random.randint(10, 15)
#     chosen_indices = np.random.choice(pairings_df.index, num_chosen, replace=False)
    
#     solution_pairings = []
#     for idx in chosen_indices:
#         p_row = pairings_df.loc[idx]
#         flight_numbers = flights_df[flights_df['FlightID'].isin(p_row['flights'])]['FlightNumber'].tolist()
#         solution_pairings.append({
#             'PairingID': f"P{p_row['pairing_id']}",
#             'Flights': ", ".join(flight_numbers),
#             'TotalCost': p_row['cost']
#         })
    
#     return pd.DataFrame(solution_pairings)



# # --- UI Pages ---
# def home_page():
#     """The main landing page of the dashboard."""
#     st.image("https://placehold.co/1200x300/003262/FFFFFF?text=Project+Horizon", use_container_width=True)
#     st.title("Next-Generation Operations Optimization")
#     st.markdown("""
#         Welcome to the **Project Horizon** dashboard. This demo now includes the actual MILP model structure for Fleet Assignment using **Gurobi**.
#     """)
#     st.info("ℹ️ **Note:** This is an enhanced, production-style demo. Use the sidebar controls on each page to perform 'what-if' analysis and explore operational trade-offs.", icon="ℹ️")


# def aircraft_route_optimization_page():
#     """UI for Aircraft Route Optimization."""
#     st.header("✈️ Aircraft Route Optimization (Fleet Assignment)")
#     st.markdown("This module assigns the most profitable aircraft to each flight. It now uses a real **Gurobi MILP model** structure.")
#     if not GUROBI_AVAILABLE:
#         st.warning("Gurobi library not found. The optimization will use a simplified simulation.", icon="⚠️")

#     st.sidebar.header("Scenario Controls")
#     fuel_price_multiplier = st.sidebar.slider("Fuel Price Adjustment", 0.8, 1.5, 1.0, 0.1, help="Simulate changes in fuel price.")
#     demand_multiplier = st.sidebar.slider("Demand Forecast Adjustment", 0.8, 1.5, 1.0, 0.1, help="Simulate changes in passenger demand.")
    
#     flights_df, aircraft_data = create_mock_flight_data(demand_multiplier)
    
#     st.subheader("Current Flight Schedule & Profitability")
#     st.dataframe(flights_df, use_container_width=True)

#     if st.button("Optimize Fleet Assignment", type="primary"):
#         with st.spinner("Building Gurobi model and running solver..."):
#             # --- THIS IS THE CALL TO THE REAL SOLVER FUNCTION ---
#             optimized_df = run_fleet_assignment_milp(flights_df, aircraft_data, fuel_price_multiplier)
            
#             # Recalculate costs/profits for the new assignments (simplified for demo)
#             recalc_cost = (optimized_df['FlightHours'] * 8000 * fuel_price_multiplier).astype(int)
#             optimized_df['OperatingCost'] = np.where(optimized_df['AircraftType'] != flights_df['AircraftType'], recalc_cost, optimized_df['OperatingCost'])
#             optimized_df['Profit'] = optimized_df['Revenue'] - optimized_df['OperatingCost']
            
#             original_profit = flights_df['Profit'].sum()
#             optimized_profit = optimized_df['Profit'].sum()
#             profit_change = optimized_profit - original_profit

#             st.success("✅ Optimization Complete!")
            
#             st.subheader("Optimization Impact Summary")
#             col1, col2, col3 = st.columns(3)
#             col1.metric("Original Network Profit", f"${original_profit:,.0f}")
#             col2.metric("Optimized Network Profit", f"${optimized_profit:,.0f}", delta=f"${profit_change:,.0f}")
#             col3.metric("Aircraft Swaps Made", (optimized_df['AircraftType'] != flights_df['AircraftType']).sum())

#             st.subheader("Optimized Schedule")
#             st.dataframe(optimized_df, use_container_width=True)

# def crew_scheduling_page():
#     """UI for Crew Scheduling."""
#     st.header("👥 Crew Scheduling Optimization")
#     st.markdown("Generates cost-effective and legal crew pairings. Analyze trade-offs between cost and operational robustness.")

#     st.sidebar.header("Scenario Controls")
#     hotel_cost_multiplier = st.sidebar.slider("Hotel/Per Diem Cost Factor", 0.8, 1.5, 1.0, 0.1)
#     robustness_pref = st.sidebar.select_slider("Prioritize", ["Lowest Cost", "Balanced", "Robustness"], "Balanced")
    
#     flights_df, _ = create_mock_flight_data()
#     flights_df = flights_df.head(15)
#     st.subheader("Flights Requiring Crew Coverage")
#     st.dataframe(flights_df, use_container_width=True)

#     if st.button("Generate Optimal Crew Pairings", type="primary"):
#         with st.spinner("Solving Set Covering model..."):
#             time.sleep(4)
            
#             cost_factor = 1.08 if robustness_pref == "Lowest Cost" else (1.12 if robustness_pref == "Balanced" else 1.18)
#             num_pairings = 12 if robustness_pref == "Lowest Cost" else 10
            
#             pairings = []
#             for i in range(num_pairings):
#                 pairing_flights = flights_df.sample(n=np.random.randint(2, 4))
#                 cost = (pairing_flights['OperatingCost'].sum() * 0.15 + np.random.randint(1000, 5000)) * hotel_cost_multiplier
#                 pairings.append({
#                     'PairingID': f'P{1000+i}',
#                     'Flights': ', '.join(pairing_flights['FlightNumber'].tolist()),
#                     'CrewBase': np.random.choice(['ORD', 'EWR', 'SFO']),
#                     'DutyHours': round(pairing_flights['FlightHours'].sum() * 1.2, 1),
#                     'TotalCost': cost
#                 })
#             pairings_df = pd.DataFrame(pairings)
#             total_pairing_cost = pairings_df['TotalCost'].sum()
#             unoptimized_cost = total_pairing_cost * cost_factor

#             st.success("✅ Crew Pairings Generated!")
            
#             st.subheader("Pairing Cost & KPI Summary")
#             col1, col2, col3 = st.columns(3)
#             col1.metric("Unoptimized Crew Cost", f"${unoptimized_cost:,.0f}")
#             col2.metric("Optimized Crew Cost", f"${total_pairing_cost:,.0f}", delta=f"-${unoptimized_cost - total_pairing_cost:,.0f}", delta_color="inverse")
#             col3.metric("Avg. Duty Hours / Pairing", f"{pairings_df['DutyHours'].mean():.1f}h")

#             st.subheader("Generated Pairings")
#             st.dataframe(pairings_df.style.format({'TotalCost': '${:,.2f}'}), use_container_width=True)


# def network_optimization_page():
#     """UI for Strategic Network Planning."""
#     st.header("🌐 Network Optimization")
#     st.markdown("Analyze the strategic value and profitability of adding new routes to the network.")
    
#     st.subheader("Propose a New Route for Analysis")
    
#     origins = ['ORD', 'EWR', 'SFO', 'DEN', 'IAH', 'LAX']
#     potential_new_dests = ['BCN', 'DUB', 'ZRH', 'AMS', 'MEX', 'GRU']
    
#     col1, col2 = st.columns(2)
#     with col1:
#         origin = st.selectbox("Select Origin Hub", origins)
#     with col2:
#         destination = st.selectbox("Select Potential New Destination", potential_new_dests)
        
#     if st.button(f"Analyze Profitability of {origin} ↔ {destination}", type="primary"):
#         with st.spinner("Running network profitability model..."):
#             time.sleep(3)
            
#             flight_hours = np.random.uniform(7, 12)
#             cost = int(flight_hours * 8500 * 1.1)
#             demand = np.random.randint(180, 270)
#             revenue = demand * np.random.randint(800, 1500)
#             profit = revenue - cost
            
#             st.success("✅ Analysis Complete!")
#             st.subheader(f"Projected P&L for {origin} ↔ {destination}")
            
#             col1, col2, col3 = st.columns(3)
#             col1.metric("Projected Daily Revenue", f"${revenue:,.0f}")
#             col2.metric("Projected Daily Cost", f"${cost:,.0f}")
#             col3.metric("Projected Daily Profit", f"${profit:,.0f}")
            
#             if profit > 20000:
#                 st.success(f"**Recommendation:** High potential. This route is projected to be highly profitable.")
#             elif profit > 0:
#                 st.warning(f"**Recommendation:** Marginally profitable. Could be a strategic addition.")
#             else:
#                 st.error(f"**Recommendation:** Not profitable under current assumptions.")


# def predictive_maintenance_page():
#     """UI for Predictive Maintenance."""
#     st.header("🔧 Predictive Maintenance")
#     st.markdown("Forecast component failures to enable proactive maintenance scheduling and prevent AOG events.")

#     st.sidebar.header("Fleet-wide View")
#     num_at_risk = st.sidebar.slider("Number of 'At-Risk' Aircraft to Show", 3, 10, 5)
    
#     at_risk_data = {
#         'TailNumber': [f'N{np.random.randint(100,999)}UA' for _ in range(num_at_risk)],
#         'Component': np.random.choice(['Engine 1', 'APU', 'Brake-Assembly', 'TCAS-Unit'], num_at_risk),
#         'Predicted_RUL_days': sorted(np.random.randint(5, 45, num_at_risk))
#     }
#     at_risk_df = pd.DataFrame(at_risk_data)
#     st.sidebar.subheader("Top Maintenance Alerts")
#     st.sidebar.dataframe(at_risk_df, use_container_width=True)

#     st.subheader("Individual Aircraft Analysis")
#     tail_number = st.selectbox("Select Aircraft Tail Number", at_risk_df['TailNumber'].unique())
    
#     if tail_number:
#         sensor_df = create_mock_sensor_data()
#         component = at_risk_df[at_risk_df['TailNumber'] == tail_number]['Component'].iloc[0]
#         st.subheader(f"Live Sensor Data for {component} on {tail_number}")

#         fig = go.Figure()
#         fig.add_trace(go.Scatter(x=sensor_df['Timestamp'], y=sensor_df['EGT_C'], name='EGT'))
#         fig.add_trace(go.Scatter(x=sensor_df['Timestamp'], y=sensor_df['Vibration_mm_s']*100, name='Vibration (x100)', yaxis='y2'))
#         fig.update_layout(
#             title_text=f"Sensor Readings for {component}",
#             yaxis=dict(title="EGT (°C)"),
#             yaxis2=dict(title="Vibration (mm/s)", overlaying='y', side='right')
#         )
#         st.plotly_chart(fig, use_container_width=True)

#         if st.button("Predict Remaining Useful Life (RUL)", type="primary"):
#             with st.spinner("Analyzing data and running ML model..."):
#                 time.sleep(2)
#                 rul_days = at_risk_df[at_risk_df['TailNumber'] == tail_number]['Predicted_RUL_days'].iloc[0]
                
#                 st.success("✅ RUL Prediction Complete!")
#                 st.subheader("Prediction & Cost of Inaction")
                
#                 col1, col2 = st.columns(2)
#                 col1.metric("Predicted RUL", f"{rul_days} days")
                
#                 inaction_cost = 2500000
#                 prob_failure = 1 / (rul_days + 10)
#                 cost_inaction = int(inaction_cost * prob_failure * 10)
#                 col2.metric("Cost of 10-day Delay", f"${cost_inaction:,.0f}", help="Projected risk-adjusted cost of delaying maintenance by 10 days.")

#                 st.warning(f"**Recommendation:** Schedule maintenance for **{component}** on **{tail_number}** within the next **{rul_days - 5} days**.", icon="⚠️")


# def spare_parts_inventory_page():
#     """UI for Spare Parts Inventory Management."""
#     st.header("📦 Spare Parts Inventory Management")
#     st.markdown("Optimize inventory levels to minimize holding costs while guaranteeing part availability.")

#     st.sidebar.header("Scenario Controls")
#     service_level = st.sidebar.slider("Target Service Level", 95.0, 99.9, 99.0, 0.1, format="%.1f%%")
    
#     inventory_df = create_mock_inventory_data()
    
#     st.subheader("System-Wide Inventory Overview")
#     total_value = (inventory_df['OnHand'] * inventory_df['HoldingCost_USD'] * 20).sum()
#     total_holding_cost = inventory_df['HoldingCost_USD'].sum() * 12
#     col1, col2 = st.columns(2)
#     col1.metric("Total Inventory Value", f"${total_value:,.0f}")
#     col2.metric("Projected Annual Holding Cost", f"${total_holding_cost:,.0f}")
    
#     st.subheader("Optimize Inventory Policy by Location")
#     location = st.selectbox("Select Maintenance Station", inventory_df['Location'].unique())
    
#     if st.button(f"Optimize All Parts at {location}", type="primary"):
#         with st.spinner(f"Running Stochastic models for {location}..."):
#             time.sleep(3)
            
#             loc_df = inventory_df[inventory_df['Location'] == location].copy()
#             factor = 1 + (service_level - 99.0) / 10.0
#             loc_df['Optimal_OnHand'] = (loc_df['OnHand'] * 0.7 * factor).astype(int)
#             loc_df['New_HoldingCost'] = loc_df['Optimal_OnHand'] * loc_df['HoldingCost_USD']
#             loc_df['Old_HoldingCost'] = loc_df['OnHand'] * loc_df['HoldingCost_USD']
            
#             cost_savings = (loc_df['Old_HoldingCost'] - loc_df['New_HoldingCost']).sum()
            
#             st.success(f"Optimization Complete for {location}!")
#             st.metric(
#                 f"Projected Annual Holding Cost Savings at {location}",
#                 f"${cost_savings * 12:,.0f}",
#                 delta=f"-${(loc_df['OnHand'] - loc_df['Optimal_OnHand']).sum()} units"
#             )
            
#             st.dataframe(loc_df[['PartNumber', 'OnHand', 'Optimal_OnHand', 'HoldingCost_USD']], use_container_width=True)


# # --- Main App Router ---
# def main():
#     """Main function to route to the correct page."""
#     st.sidebar.image("https://placehold.co/400x100/FFFFFF/003262?text=United+Airlines")
#     st.sidebar.title("Project Horizon Modules")
    
#     page_options = {
#         "Home": home_page,
#         "Aircraft Route Optimization": aircraft_route_optimization_page,
#         "Crew Scheduling": crew_scheduling_page,
#         "Network Optimization": network_optimization_page,
#         "Predictive Maintenance": predictive_maintenance_page,
#         "Spare Parts Inventory": spare_parts_inventory_page
#     }
    
#     selection = st.sidebar.radio("Go to", list(page_options.keys()))
    
#     page = page_options[selection]
#     page()

# if __name__ == "__main__":
#     main()
