import plotly.graph_objects as go

def create_claim_waterfall(breakdown_dict, final_claim):
    """
    Generates a Plotly Waterfall chart showing how the final claim was calculated.
    Steps: Base Severity → Brand Tier → Fuel Adjust → Collision Severity → Age Depreciation → Final
    """
    base_cost    = breakdown_dict.get("base_severity", 0)
    brand_mult   = breakdown_dict.get("brand_multiplier", 1.0)
    fuel_mult    = breakdown_dict.get("fuel_multiplier", 1.0)
    col_mult     = breakdown_dict.get("collision_multiplier", 1.0)
    dep_factor   = breakdown_dict.get("depreciation_factor", 1.0)

    # Calculate step-by-step absolute INR impacts
    after_brand  = base_cost * brand_mult
    brand_impact = after_brand - base_cost

    after_fuel   = after_brand * fuel_mult
    fuel_impact  = after_fuel - after_brand

    after_col    = after_fuel * col_mult
    col_impact   = after_col - after_fuel

    after_dep    = after_col * dep_factor
    dep_impact   = after_dep - after_col  # negative (depreciation reduces value)

    fig = go.Figure(go.Waterfall(
        name="Claim Breakdown",
        orientation="v",
        measure=["relative", "relative", "relative", "relative", "relative", "total"],
        x=["Base Severity", "Brand Tier", "Fuel Adjust", "Collision Severity", "Age Depreciation", "Final Estimate"],
        textposition="outside",
        text=[
            f"₹{base_cost:,.0f}",
            f"+₹{brand_impact:,.0f}",
            f"+₹{fuel_impact:,.0f}",
            f"{'+'  if col_impact >= 0 else ''}₹{col_impact:,.0f}",
            f"₹{dep_impact:,.0f}",
            f"₹{final_claim:,.0f}"
        ],
        y=[base_cost, brand_impact, fuel_impact, col_impact, dep_impact, final_claim],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#ef553b"}},
        increasing={"marker": {"color": "#00cc96"}},
        totals={"marker": {"color": "#636efa"}}
    ))

    fig.update_layout(
        title="Repair Cost Estimation Breakdown",
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(color="#e0e0e0")
    )

    return fig