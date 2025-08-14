import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
# --- Add Gurobi for MILP ---
# In a real environment, you would need to have Gurobi installed and a license.
# pip install gurobipy
# import gurobipy
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

# --- Mock Data Generation (Enhanced for Production Features) ---
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
    for _ in range(30):
        origin = np.random.choice(origins)
        dest = np.random.choice(destinations)
        if origin == dest: continue
        
        aircraft = np.random.choice(list(aircraft_types.keys()))
        flight_hours = np.random.uniform(2.5, 11.0)
        pax = int(aircraft_types[aircraft]['capacity'] * np.random.uniform(0.7, 0.95) * demand_multiplier)
        
        data.append({
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

def create_mock_sensor_data():
    """Generates mock sensor data for predictive maintenance."""
    time_series = pd.to_datetime(pd.date_range(end=pd.Timestamp.now(), periods=100))
    data = {
        'Timestamp': time_series,
        'EGT_C': 550 + np.cumsum(np.random.randn(100) * 0.1) + np.linspace(0, 15, 100),
        'Vibration_mm_s': 1.2 + np.cumsum(np.random.randn(100) * 0.005) + np.linspace(0, 0.5, 100),
        'FuelFlow_kg_hr': 2500 + np.random.randn(100) * 20
    }
    return pd.DataFrame(data)

def create_mock_inventory_data():
    """Generates mock inventory data."""
    parts = ['APU-Generator', 'Brake-Assembly', 'TCAS-Unit', 'Weather-Radar']
    locations = ['SFO MRO', 'ORD MRO', 'IAH MRO']
    inventory = []
    for part in parts:
        for loc in locations:
            inventory.append({
                'PartNumber': part,
                'Location': loc,
                'OnHand': np.random.randint(5, 50),
                'HoldingCost_USD': np.random.randint(100, 500),
                'StockoutRisk': f"{np.random.uniform(1, 10):.1f}%"
            })
    return pd.DataFrame(inventory)

# --- Backend Solver Function ---
def run_fleet_assignment_milp(flights_df, aircraft_data, fuel_price_multiplier):
    """
    This function builds and solves the Fleet Assignment MILP model using Gurobi.
    NOTE: In this environment, the actual solver is not run. The code demonstrates
    the model formulation, and a simulated result is returned.
    """
    if not GUROBI_AVAILABLE:
        st.error("Gurobi is not installed. Returning a simulated result.")
        # Fallback to simulation if Gurobi is not available
        optimized_df = flights_df.copy()
        for i, row in optimized_df.iterrows():
            if row['Passengers'] < row['Capacity'] * 0.6:
                if '777' in row['AircraftType'] or '787' in row['AircraftType']:
                    optimized_df.loc[i, 'AircraftType'] = '737-MAX'
            elif row['Passengers'] > row['Capacity'] * 0.98:
                if '737' in row['AircraftType'] or 'A320' in row['AircraftType']:
                    optimized_df.loc[i, 'AircraftType'] = '787-9'
        return optimized_df

    # --- REAL GUROBI MODEL ---
    model = gp.Model("FleetAssignment")

    # 1. Decision Variables: x[f, t] = 1 if flight f is assigned to aircraft type t
    flight_indices = flights_df.index
    aircraft_keys = list(aircraft_data.keys())
    x = model.addVars(flight_indices, aircraft_keys, vtype=GRB.BINARY, name="assign")

    # 2. Objective Function: Minimize total operating cost
    # Cost includes a penalty for spilled passengers (demand > capacity)
    total_cost = gp.quicksum(
        x[f, t] * flights_df.loc[f, 'FlightHours'] * aircraft_data[t]['cost_per_hour'] * fuel_price_multiplier
        for f in flight_indices for t in aircraft_keys
    )
    spill_cost = gp.quicksum(
        x[f, t] * (flights_df.loc[f, 'Passengers'] - aircraft_data[t]['capacity']) * 500 # $500 penalty per spilled passenger
        for f in flight_indices for t in aircraft_keys if flights_df.loc[f, 'Passengers'] > aircraft_data[t]['capacity']
    )
    model.setObjective(total_cost + spill_cost, GRB.MINIMIZE)

    # 3. Constraints
    # a) Each flight must be covered by exactly one aircraft type
    for f in flight_indices:
        model.addConstr(gp.quicksum(x[f, t] for t in aircraft_keys) == 1, name=f"flight_coverage_{f}")

    # b) The number of aircraft of a type used cannot exceed the number available
    for t in aircraft_keys:
        model.addConstr(gp.quicksum(x[f, t] for f in flight_indices) <= aircraft_data[t]['count'], name=f"fleet_size_{t}")

    # 4. Optimize the model
    # model.optimize() # This line would run the solver in a real environment

    # 5. Extract results and return
    # In a real scenario, you would extract the solution from the model.
    # For this demo, we return a simulated result.
    st.info("Gurobi model was built but not solved. Displaying simulated results.")
    # (Same simulation logic as before for demonstration purposes)
    optimized_df = flights_df.copy()
    for i, row in optimized_df.iterrows():
        if row['Passengers'] < row['Capacity'] * 0.6:
            if '777' in row['AircraftType'] or '787' in row['AircraftType']:
                optimized_df.loc[i, 'AircraftType'] = '737-MAX'
        elif row['Passengers'] > row['Capacity'] * 0.98:
            if '737' in row['AircraftType'] or 'A320' in row['AircraftType']:
                optimized_df.loc[i, 'AircraftType'] = '787-9'
    return optimized_df


# --- UI Pages ---
def home_page():
    """The main landing page of the dashboard."""
    st.image("https://placehold.co/1200x300/003262/FFFFFF?text=Project+Horizon", use_container_width=True)
    st.title("Next-Generation Operations Optimization")
    st.markdown("""
        Welcome to the **Project Horizon** dashboard. This demo now includes the actual MILP model structure for Fleet Assignment using **Gurobi**.
    """)
    st.info("ℹ️ **Note:** This is an enhanced, production-style demo. Use the sidebar controls on each page to perform 'what-if' analysis and explore operational trade-offs.", icon="ℹ️")


def aircraft_route_optimization_page():
    """UI for Aircraft Route Optimization."""
    st.header("✈️ Aircraft Route Optimization (Fleet Assignment)")
    st.markdown("This module assigns the most profitable aircraft to each flight. It now uses a real **Gurobi MILP model** structure.")
    if not GUROBI_AVAILABLE:
        st.warning("Gurobi library not found. The optimization will use a simplified simulation.", icon="⚠️")

    st.sidebar.header("Scenario Controls")
    fuel_price_multiplier = st.sidebar.slider("Fuel Price Adjustment", 0.8, 1.5, 1.0, 0.1, help="Simulate changes in fuel price.")
    demand_multiplier = st.sidebar.slider("Demand Forecast Adjustment", 0.8, 1.5, 1.0, 0.1, help="Simulate changes in passenger demand.")
    
    flights_df, aircraft_data = create_mock_flight_data(demand_multiplier)
    
    st.subheader("Current Flight Schedule & Profitability")
    st.dataframe(flights_df, use_container_width=True)

    if st.button("Optimize Fleet Assignment", type="primary"):
        with st.spinner("Building Gurobi model and running solver..."):
            # --- THIS IS THE CALL TO THE REAL SOLVER FUNCTION ---
            optimized_df = run_fleet_assignment_milp(flights_df, aircraft_data, fuel_price_multiplier)
            
            # Recalculate costs/profits for the new assignments (simplified for demo)
            recalc_cost = (optimized_df['FlightHours'] * 8000 * fuel_price_multiplier).astype(int)
            optimized_df['OperatingCost'] = np.where(optimized_df['AircraftType'] != flights_df['AircraftType'], recalc_cost, optimized_df['OperatingCost'])
            optimized_df['Profit'] = optimized_df['Revenue'] - optimized_df['OperatingCost']
            
            original_profit = flights_df['Profit'].sum()
            optimized_profit = optimized_df['Profit'].sum()
            profit_change = optimized_profit - original_profit

            st.success("✅ Optimization Complete!")
            
            st.subheader("Optimization Impact Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Original Network Profit", f"${original_profit:,.0f}")
            col2.metric("Optimized Network Profit", f"${optimized_profit:,.0f}", delta=f"${profit_change:,.0f}")
            col3.metric("Aircraft Swaps Made", (optimized_df['AircraftType'] != flights_df['AircraftType']).sum())

            st.subheader("Optimized Schedule")
            st.dataframe(optimized_df, use_container_width=True)

def crew_scheduling_page():
    """UI for Crew Scheduling."""
    st.header("👥 Crew Scheduling Optimization")
    st.markdown("Generates cost-effective and legal crew pairings. Analyze trade-offs between cost and operational robustness.")

    st.sidebar.header("Scenario Controls")
    hotel_cost_multiplier = st.sidebar.slider("Hotel/Per Diem Cost Factor", 0.8, 1.5, 1.0, 0.1)
    robustness_pref = st.sidebar.select_slider("Prioritize", ["Lowest Cost", "Balanced", "Robustness"], "Balanced")
    
    flights_df, _ = create_mock_flight_data()
    flights_df = flights_df.head(15)
    st.subheader("Flights Requiring Crew Coverage")
    st.dataframe(flights_df, use_container_width=True)

    if st.button("Generate Optimal Crew Pairings", type="primary"):
        with st.spinner("Solving Set Covering model..."):
            time.sleep(4)
            
            cost_factor = 1.08 if robustness_pref == "Lowest Cost" else (1.12 if robustness_pref == "Balanced" else 1.18)
            num_pairings = 12 if robustness_pref == "Lowest Cost" else 10
            
            pairings = []
            for i in range(num_pairings):
                pairing_flights = flights_df.sample(n=np.random.randint(2, 4))
                cost = (pairing_flights['OperatingCost'].sum() * 0.15 + np.random.randint(1000, 5000)) * hotel_cost_multiplier
                pairings.append({
                    'PairingID': f'P{1000+i}',
                    'Flights': ', '.join(pairing_flights['FlightNumber'].tolist()),
                    'CrewBase': np.random.choice(['ORD', 'EWR', 'SFO']),
                    'DutyHours': round(pairing_flights['FlightHours'].sum() * 1.2, 1),
                    'TotalCost': cost
                })
            pairings_df = pd.DataFrame(pairings)
            total_pairing_cost = pairings_df['TotalCost'].sum()
            unoptimized_cost = total_pairing_cost * cost_factor

            st.success("✅ Crew Pairings Generated!")
            
            st.subheader("Pairing Cost & KPI Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Unoptimized Crew Cost", f"${unoptimized_cost:,.0f}")
            col2.metric("Optimized Crew Cost", f"${total_pairing_cost:,.0f}", delta=f"-${unoptimized_cost - total_pairing_cost:,.0f}", delta_color="inverse")
            col3.metric("Avg. Duty Hours / Pairing", f"{pairings_df['DutyHours'].mean():.1f}h")

            st.subheader("Generated Pairings")
            st.dataframe(pairings_df.style.format({'TotalCost': '${:,.2f}'}), use_container_width=True)


def network_optimization_page():
    """UI for Strategic Network Planning."""
    st.header("🌐 Network Optimization")
    st.markdown("Analyze the strategic value and profitability of adding new routes to the network.")
    
    st.subheader("Propose a New Route for Analysis")
    
    origins = ['ORD', 'EWR', 'SFO', 'DEN', 'IAH', 'LAX']
    potential_new_dests = ['BCN', 'DUB', 'ZRH', 'AMS', 'MEX', 'GRU']
    
    col1, col2 = st.columns(2)
    with col1:
        origin = st.selectbox("Select Origin Hub", origins)
    with col2:
        destination = st.selectbox("Select Potential New Destination", potential_new_dests)
        
    if st.button(f"Analyze Profitability of {origin} ↔ {destination}", type="primary"):
        with st.spinner("Running network profitability model..."):
            time.sleep(3)
            
            flight_hours = np.random.uniform(7, 12)
            cost = int(flight_hours * 8500 * 1.1)
            demand = np.random.randint(180, 270)
            revenue = demand * np.random.randint(800, 1500)
            profit = revenue - cost
            
            st.success("✅ Analysis Complete!")
            st.subheader(f"Projected P&L for {origin} ↔ {destination}")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Projected Daily Revenue", f"${revenue:,.0f}")
            col2.metric("Projected Daily Cost", f"${cost:,.0f}")
            col3.metric("Projected Daily Profit", f"${profit:,.0f}")
            
            if profit > 20000:
                st.success(f"**Recommendation:** High potential. This route is projected to be highly profitable.")
            elif profit > 0:
                st.warning(f"**Recommendation:** Marginally profitable. Could be a strategic addition.")
            else:
                st.error(f"**Recommendation:** Not profitable under current assumptions.")


def predictive_maintenance_page():
    """UI for Predictive Maintenance."""
    st.header("🔧 Predictive Maintenance")
    st.markdown("Forecast component failures to enable proactive maintenance scheduling and prevent AOG events.")

    st.sidebar.header("Fleet-wide View")
    num_at_risk = st.sidebar.slider("Number of 'At-Risk' Aircraft to Show", 3, 10, 5)
    
    at_risk_data = {
        'TailNumber': [f'N{np.random.randint(100,999)}UA' for _ in range(num_at_risk)],
        'Component': np.random.choice(['Engine 1', 'APU', 'Brake-Assembly', 'TCAS-Unit'], num_at_risk),
        'Predicted_RUL_days': sorted(np.random.randint(5, 45, num_at_risk))
    }
    at_risk_df = pd.DataFrame(at_risk_data)
    st.sidebar.subheader("Top Maintenance Alerts")
    st.sidebar.dataframe(at_risk_df, use_container_width=True)

    st.subheader("Individual Aircraft Analysis")
    tail_number = st.selectbox("Select Aircraft Tail Number", at_risk_df['TailNumber'].unique())
    
    if tail_number:
        sensor_df = create_mock_sensor_data()
        component = at_risk_df[at_risk_df['TailNumber'] == tail_number]['Component'].iloc[0]
        st.subheader(f"Live Sensor Data for {component} on {tail_number}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sensor_df['Timestamp'], y=sensor_df['EGT_C'], name='EGT'))
        fig.add_trace(go.Scatter(x=sensor_df['Timestamp'], y=sensor_df['Vibration_mm_s']*100, name='Vibration (x100)', yaxis='y2'))
        fig.update_layout(
            title_text=f"Sensor Readings for {component}",
            yaxis=dict(title="EGT (°C)"),
            yaxis2=dict(title="Vibration (mm/s)", overlaying='y', side='right')
        )
        st.plotly_chart(fig, use_container_width=True)

        if st.button("Predict Remaining Useful Life (RUL)", type="primary"):
            with st.spinner("Analyzing data and running ML model..."):
                time.sleep(2)
                rul_days = at_risk_df[at_risk_df['TailNumber'] == tail_number]['Predicted_RUL_days'].iloc[0]
                
                st.success("✅ RUL Prediction Complete!")
                st.subheader("Prediction & Cost of Inaction")
                
                col1, col2 = st.columns(2)
                col1.metric("Predicted RUL", f"{rul_days} days")
                
                inaction_cost = 2500000
                prob_failure = 1 / (rul_days + 10)
                cost_inaction = int(inaction_cost * prob_failure * 10)
                col2.metric("Cost of 10-day Delay", f"${cost_inaction:,.0f}", help="Projected risk-adjusted cost of delaying maintenance by 10 days.")

                st.warning(f"**Recommendation:** Schedule maintenance for **{component}** on **{tail_number}** within the next **{rul_days - 5} days**.", icon="⚠️")


def spare_parts_inventory_page():
    """UI for Spare Parts Inventory Management."""
    st.header("📦 Spare Parts Inventory Management")
    st.markdown("Optimize inventory levels to minimize holding costs while guaranteeing part availability.")

    st.sidebar.header("Scenario Controls")
    service_level = st.sidebar.slider("Target Service Level", 95.0, 99.9, 99.0, 0.1, format="%.1f%%")
    
    inventory_df = create_mock_inventory_data()
    
    st.subheader("System-Wide Inventory Overview")
    total_value = (inventory_df['OnHand'] * inventory_df['HoldingCost_USD'] * 20).sum()
    total_holding_cost = inventory_df['HoldingCost_USD'].sum() * 12
    col1, col2 = st.columns(2)
    col1.metric("Total Inventory Value", f"${total_value:,.0f}")
    col2.metric("Projected Annual Holding Cost", f"${total_holding_cost:,.0f}")
    
    st.subheader("Optimize Inventory Policy by Location")
    location = st.selectbox("Select Maintenance Station", inventory_df['Location'].unique())
    
    if st.button(f"Optimize All Parts at {location}", type="primary"):
        with st.spinner(f"Running Stochastic models for {location}..."):
            time.sleep(3)
            
            loc_df = inventory_df[inventory_df['Location'] == location].copy()
            factor = 1 + (service_level - 99.0) / 10.0
            loc_df['Optimal_OnHand'] = (loc_df['OnHand'] * 0.7 * factor).astype(int)
            loc_df['New_HoldingCost'] = loc_df['Optimal_OnHand'] * loc_df['HoldingCost_USD']
            loc_df['Old_HoldingCost'] = loc_df['OnHand'] * loc_df['HoldingCost_USD']
            
            cost_savings = (loc_df['Old_HoldingCost'] - loc_df['New_HoldingCost']).sum()
            
            st.success(f"Optimization Complete for {location}!")
            st.metric(
                f"Projected Annual Holding Cost Savings at {location}",
                f"${cost_savings * 12:,.0f}",
                delta=f"-${(loc_df['OnHand'] - loc_df['Optimal_OnHand']).sum()} units"
            )
            
            st.dataframe(loc_df[['PartNumber', 'OnHand', 'Optimal_OnHand', 'HoldingCost_USD']], use_container_width=True)


# --- Main App Router ---
def main():
    """Main function to route to the correct page."""
    st.sidebar.image("https://placehold.co/400x100/FFFFFF/003262?text=United+Airlines")
    st.sidebar.title("Project Horizon Modules")
    
    page_options = {
        "Home": home_page,
        "Aircraft Route Optimization": aircraft_route_optimization_page,
        "Crew Scheduling": crew_scheduling_page,
        "Network Optimization": network_optimization_page,
        "Predictive Maintenance": predictive_maintenance_page,
        "Spare Parts Inventory": spare_parts_inventory_page
    }
    
    selection = st.sidebar.radio("Go to", list(page_options.keys()))
    
    page = page_options[selection]
    page()

if __name__ == "__main__":
    main()