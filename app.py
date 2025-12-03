import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
import statsmodels.api as sm

st.set_page_config(page_title="Marketing Mix Modeling", layout="wide")

st.title("Marketing Mix Modeling (MMM)")
st.markdown("Upload media and backend performance data to analyze channel effectiveness with saturation and adstock effects.")

def clean_numeric(df, cols):
    """Clean numeric columns by removing commas, dollar signs, and converting to numeric"""
    for c in cols:
        if c in df.columns:
            df[c] = (df[c].astype(str)
                          .str.replace(",", "", regex=False)
                          .str.replace("$", "", regex=False)
                          .str.strip())
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df

def add_week_start_from_date(df):
    """Add canonical Week_Start column derived from Date or Week Start"""
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df["Week_Start"] = df["Date"] - pd.to_timedelta(df["Date"].dt.weekday, unit="D")
    elif "Week Start" in df.columns:
        df["Week_Start"] = pd.to_datetime(df["Week Start"], errors="coerce")
        df = df.dropna(subset=["Week_Start"])
    else:
        first_col = df.columns[0]
        df[first_col] = pd.to_datetime(df[first_col], errors="coerce")
        df = df.dropna(subset=[first_col])
        df["Week_Start"] = df[first_col] - pd.to_timedelta(df[first_col].dt.weekday, unit="D")
    return df

def apply_saturation(x, beta):
    """Apply saturation transformation: x_sat = x^beta (clip negative values to 0)"""
    x_clipped = np.clip(x, 0, None)
    return x_clipped ** beta

def apply_adstock(x_sat, decay):
    """Apply adstock transformation: s[t] = x_sat[t] + decay * s[t-1]"""
    adstocked = np.zeros_like(x_sat)
    adstocked[0] = x_sat[0]
    for t in range(1, len(x_sat)):
        adstocked[t] = x_sat[t] + decay * adstocked[t-1]
    return adstocked

st.sidebar.header("Configuration")

media_file = st.sidebar.file_uploader("Media / Ad Platform Data CSV", type=["csv"], key="media")

with open("media_template.csv", "rb") as f:
    st.sidebar.download_button(
        label="📥 Download Media Template",
        data=f,
        file_name="media_template.csv",
        mime="text/csv",
        help="Download a sample CSV showing the expected format for media data"
    )

backend_file = st.sidebar.file_uploader("Backend Performance CSV (POS / CRM / ecommerce)", type=["csv"], key="backend")

with open("backend_template.csv", "rb") as f:
    st.sidebar.download_button(
        label="📥 Download Backend Template",
        data=f,
        file_name="backend_template.csv",
        mime="text/csv",
        help="Download a sample CSV showing the expected format for backend data"
    )

if media_file is None or backend_file is None:
    st.info("👆 Please upload BOTH a media/ad platform CSV and a backend performance CSV to proceed.")
    
    st.markdown("""
    ### Two-File Upload Required
    
    **Media / Ad Platform CSV:**
    - Campaign-level data with `Date`, `Channel`, `Spend`
    - Platform-attributed conversions (Online Order, etc.)
    - Optional: Impressions, Clicks, Video Views, etc.
    
    **Backend Performance CSV (Truth):**
    - POS/CRM/ecommerce data with `Date`
    - True business KPIs: Online Order, Online Order Revenue, Store Traffic, etc.
    - This is the source of truth for your dependent variable
    
    **The app will:**
    1. Compute Week Start (Monday) from Date for both files
    2. Aggregate media by week + channel → pivot Spend to wide format
    3. Aggregate backend KPIs by week (truth)
    4. Merge backend + media into modeling dataset
    5. Use backend KPI as dependent variable (not platform-attributed conversions)
    6. Apply saturation and adstock transformations to media spend
    7. Fit OLS regression model
    8. Show attribution comparison: Platform vs Backend vs MMM
    """)
    st.stop()

media_df = pd.read_csv(media_file)
backend_df = pd.read_csv(backend_file)

st.header("📊 Preview: Raw Data")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Media / Ad Platform Data")
    st.dataframe(media_df.head(20), use_container_width=True)
    st.caption(f"{len(media_df):,} rows")

with col2:
    st.subheader("Backend Performance Data (Truth)")
    st.dataframe(backend_df.head(20), use_container_width=True)
    st.caption(f"{len(backend_df):,} rows")

media_numeric_cols = [
    "Spend", "Impressions", "Clicks", "Video Views", "Video Completions",
    "Online Order", "Online Order Revenue",
    "Store Traffic", "Store Visit Revenue",
    "Reservations", "Content Views", "Add To Carts", "Page Views"
]

backend_numeric_cols = [
    "Online Order", "Online Order Revenue",
    "Store Traffic", "Store Visit Revenue",
    "Reservations"
]

media_df = clean_numeric(media_df, media_numeric_cols)
backend_df = clean_numeric(backend_df, backend_numeric_cols)

media_df = add_week_start_from_date(media_df)
backend_df = add_week_start_from_date(backend_df)

media_df = media_df.drop(columns=["Week Start"], errors="ignore").rename(columns={"Week_Start": "Week Start"})
backend_df = backend_df.drop(columns=["Week Start"], errors="ignore").rename(columns={"Week_Start": "Week Start"})

if 'Channel' not in media_df.columns:
    st.error("'Channel' column not found in media CSV. Please ensure your media file has this column.")
    st.stop()

if 'Spend' not in media_df.columns:
    st.error("'Spend' column not found in media CSV. Please ensure your media file has this column for pivoting.")
    st.stop()

st.header("🏗️ Build MMM Dataset")

st.subheader("Step 1: Aggregate Media by Week + Channel")
available_media_cols = [col for col in media_numeric_cols if col in media_df.columns]
media_weekly_channel = media_df.groupby(["Week Start", "Channel"], as_index=False)[available_media_cols].sum()
media_weekly_channel = media_weekly_channel.sort_values(["Week Start", "Channel"])
st.dataframe(media_weekly_channel.head(15), use_container_width=True)
st.caption(f"{len(media_weekly_channel):,} week-channel combinations")

st.subheader("Step 2: Pivot Media Spend to Wide Format")
media_spend_wide = media_weekly_channel.pivot_table(
    index="Week Start",
    columns="Channel",
    values="Spend",
    aggfunc="sum"
).fillna(0)
media_spend_wide.columns = [f"Spend_{c}" for c in media_spend_wide.columns]
media_spend_wide = media_spend_wide.reset_index().sort_values("Week Start")
st.dataframe(media_spend_wide.head(10), use_container_width=True)
st.caption(f"One row per week: {len(media_spend_wide)} unique weeks")

st.subheader("Step 3: Aggregate Backend KPIs by Week")
kpi_cols = [c for c in backend_numeric_cols if c in backend_df.columns]
if len(kpi_cols) == 0:
    st.error("No backend KPI columns found. Expected at least one of: Online Order, Online Order Revenue, Store Traffic, Store Visit Revenue, Reservations")
    st.stop()

promo_cols = [c for c in ["Promo_Flag", "Promo"] if c in backend_df.columns]

for p in promo_cols:
    backend_df[p] = pd.to_numeric(backend_df[p].replace({"Y": 1, "N": 0, "Yes": 1, "No": 0, "TRUE": 1, "FALSE": 0, "True": 1, "False": 0}), errors="coerce").fillna(0).astype(int)

agg_dict = {c: "sum" for c in kpi_cols}
for p in promo_cols:
    agg_dict[p] = "max"

backend_weekly = backend_df.groupby("Week Start", as_index=False).agg(agg_dict)
backend_weekly = backend_weekly.sort_values("Week Start")
st.dataframe(backend_weekly.head(10), use_container_width=True)
st.caption(f"One row per week: {len(backend_weekly)} unique weeks")

st.subheader("Step 4: Merge Backend KPIs + Media Spend")
df_model = backend_weekly.merge(media_spend_wide, on="Week Start", how="left").fillna(0)
df_model = df_model.sort_values("Week Start").reset_index(drop=True)

df_model["Trend"] = np.arange(1, len(df_model) + 1)

df_model["Quarter_num"] = df_model["Week Start"].dt.quarter
quarter_dummies = pd.get_dummies(df_model["Quarter_num"], prefix="Q", drop_first=True).astype(int)
df_model = pd.concat([df_model, quarter_dummies], axis=1)

if "Promo_Flag" not in df_model.columns and "Promo" in df_model.columns:
    df_model.rename(columns={"Promo": "Promo_Flag"}, inplace=True)
if "Promo_Flag" not in df_model.columns:
    df_model["Promo_Flag"] = 0

st.dataframe(df_model.tail(20), use_container_width=True)
st.success(f"✅ Modeling dataset ready: {len(df_model)} weeks (one row per Monday)")
st.caption("Added control variables: Trend (time), Q_2/Q_3/Q_4 (seasonality), Promo_Flag (promotions)")

spend_cols = [col for col in df_model.columns if col.startswith('Spend_')]

if len(spend_cols) == 0:
    st.error("No Spend_ columns found after pivoting. Please check your media data.")
    st.stop()

st.header("🎯 Select Modeling Columns")

col1, col2 = st.columns(2)

with col1:
    available_kpis = [c for c in ["Online Order", "Online Order Revenue", "Store Traffic", "Store Visit Revenue", "Reservations"] if c in df_model.columns]
    if len(available_kpis) == 0:
        st.error("No backend KPIs found in modeling dataset.")
        st.stop()
    
    default_kpi = 'Online Order' if 'Online Order' in available_kpis else available_kpis[0]
    kpi_col = st.selectbox(
        "Dependent Variable (Backend KPI)",
        options=available_kpis,
        index=available_kpis.index(default_kpi) if default_kpi in available_kpis else 0,
        help="KPI from backend data (POS/CRM/ecommerce)"
    )
    st.caption("💡 KPI comes from backend data (POS/CRM/ecommerce), not from ad platform-reported conversions.")

with col2:
    media_cols = st.multiselect(
        "Media Variables (Channel Spend)",
        options=spend_cols,
        default=spend_cols,
        help="Channel-level Spend columns from media data"
    )
    st.caption("ℹ️ Media variables are channel-level Spend columns derived from your media file.")

if len(media_cols) == 0:
    st.warning("Please select at least one media variable to continue.")
    st.stop()

st.sidebar.header("Transform Parameters")
adstock_decay = st.sidebar.slider(
    "Adstock Decay (λ)",
    min_value=0.0,
    max_value=0.95,
    value=0.5,
    step=0.05,
    help="Carryover effect: how much of last week's impact carries over"
)

saturation_beta = st.sidebar.slider(
    "Saturation Exponent (β)",
    min_value=0.5,
    max_value=1.0,
    value=0.8,
    step=0.05,
    help="Diminishing returns: lower values = more saturation"
)

st.header("🔄 Transforms: Adstock & Saturation")

transformed_data = {}

for col in media_cols:
    x = df_model[col].astype(float).values
    saturated = apply_saturation(x, saturation_beta)
    adstocked = apply_adstock(saturated, adstock_decay)
    transformed_data[col] = adstocked

with st.expander("Show Adstock & Saturation Diagnostics"):
    for col in media_cols:
        x = df_model[col].astype(float).values
        st.subheader(f"{col}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_model['Week Start'],
            y=x,
            name='Original',
            mode='lines'
        ))
        fig.add_trace(go.Scatter(
            x=df_model['Week Start'],
            y=transformed_data[col],
            name='Transformed (Sat + Adstock)',
            mode='lines'
        ))
        fig.update_layout(
            title=f"{col}: Original vs Transformed",
            xaxis_title="Week",
            yaxis_title="Value",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

st.header("📈 Fit OLS Model")

X = pd.DataFrame(transformed_data)

X["Trend"] = df_model["Trend"].astype(float).values

for qcol in ["Q_2", "Q_3", "Q_4"]:
    if qcol in df_model.columns:
        X[qcol] = df_model[qcol].astype(float).values

if "Promo_Flag" in df_model.columns and df_model["Promo_Flag"].sum() > 0:
    X["Promo_Flag"] = df_model["Promo_Flag"].astype(float).values

X = X.apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)

y = pd.to_numeric(df_model[kpi_col], errors="coerce").fillna(0).astype(float).values

X_with_const = sm.add_constant(X).astype(float)

model = sm.OLS(y, X_with_const).fit()

st.subheader("Model Summary")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("R²", f"{model.rsquared:.4f}")
with col2:
    st.metric("Adjusted R²", f"{model.rsquared_adj:.4f}")
with col3:
    st.metric("F-statistic", f"{model.fvalue:.2f}")

st.caption("Model includes: adstock + saturation on media spend, plus controls for time trend, quarterly seasonality, and promotions (if Promo_Flag is provided).")

with st.expander("Show full regression output"):
    st.text(model.summary().as_text())

st.header("💰 Channel Contributions")

contributions = {}
for col in media_cols:
    coef = model.params[col]
    avg_contribution = coef * transformed_data[col].mean()
    contributions[col] = avg_contribution

intercept_contribution = model.params['const']
total_predicted = sum(contributions.values()) + intercept_contribution

contrib_df_full = pd.DataFrame({
    'Channel': list(contributions.keys()) + ['Intercept'],
    'Contribution': list(contributions.values()) + [intercept_contribution],
})
contrib_df_full['Share (%)'] = (contrib_df_full['Contribution'] / total_predicted * 100)

contrib_df_media = pd.DataFrame({
    'Channel': list(contributions.keys()),
    'Contribution': list(contributions.values()),
})
contrib_df_media['Share (%)'] = (contrib_df_media['Contribution'] / total_predicted * 100)

col1, col2 = st.columns([1, 1])

with col1:
    st.dataframe(contrib_df_full.style.format({
        'Contribution': '{:.2f}',
        'Share (%)': '{:.2f}'
    }), use_container_width=True)

with col2:
    fig = px.bar(
        contrib_df_media,
        x='Channel',
        y='Contribution',
        title='Channel Contributions (Media Only)',
        labels={'Contribution': 'Average Contribution to KPI'}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

st.header("📉 Actual vs Predicted")

y_pred = model.predict(X_with_const)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_model['Week Start'],
    y=y,
    name='Actual',
    mode='lines+markers',
    line=dict(color='blue')
))
fig.add_trace(go.Scatter(
    x=df_model['Week Start'],
    y=y_pred,
    name='Predicted',
    mode='lines+markers',
    line=dict(color='red', dash='dash')
))
fig.update_layout(
    title=f"Actual vs Predicted {kpi_col}",
    xaxis_title="Week Start",
    yaxis_title=kpi_col,
    height=400
)
st.plotly_chart(fig, use_container_width=True)

mae = np.mean(np.abs(y - y_pred))
rmse = np.sqrt(np.mean((y - y_pred)**2))

mask = (y > 0) & np.isfinite(y)
if mask.sum() > 0:
    mape = np.mean(np.abs((y[mask] - y_pred[mask]) / y[mask])) * 100
else:
    mape = np.nan

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("MAE", f"{mae:.2f}")
with col2:
    st.metric("RMSE", f"{rmse:.2f}")
with col3:
    if np.isnan(mape):
        st.metric("MAPE", "N/A")
    else:
        st.metric("MAPE", f"{mape:.2f}%")

st.header("🔍 Attribution: Platform vs Backend vs MMM")

if "Online Order" in media_weekly_channel.columns:
    platform_weekly = media_weekly_channel.pivot_table(
        index="Week Start",
        columns="Channel",
        values="Online Order",
        aggfunc="sum"
    ).fillna(0)
    platform_weekly.columns = [f"PlatOO_{c}" for c in platform_weekly.columns]
    platform_weekly = platform_weekly.reset_index()
    platform_weekly["Platform_Total_OO"] = platform_weekly[[c for c in platform_weekly.columns if c.startswith("PlatOO_")]].sum(axis=1)
    
    attrib_compare = backend_weekly.merge(platform_weekly[["Week Start", "Platform_Total_OO"]], on="Week Start", how="left").fillna(0)
    
    pred_df = pd.DataFrame({
        "Week Start": df_model["Week Start"],
        "MMM_Predicted": y_pred
    })
    attrib_compare = attrib_compare.merge(pred_df, on="Week Start", how="left")
    
    st.markdown("**Weekly Comparison: Backend (Truth) vs Platform-Attributed vs MMM-Predicted**")
    st.dataframe(attrib_compare.tail(20), use_container_width=True)
    st.caption("Backend KPIs (truth) vs ad platform-attributed online orders vs MMM-predicted KPI.")
    
    if "Online Order" in attrib_compare.columns:
        total_backend_oo = attrib_compare["Online Order"].sum()
        total_platform_oo = attrib_compare["Platform_Total_OO"].sum()
        total_mmm_pred = attrib_compare["MMM_Predicted"].sum()
        
        st.subheader("Total Attribution Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Backend Online Order (Truth)", f"{total_backend_oo:,.0f}")
        with col2:
            st.metric("Platform-Attributed OO", f"{total_platform_oo:,.0f}")
            over_under = ((total_platform_oo - total_backend_oo) / total_backend_oo * 100) if total_backend_oo > 0 else 0
            st.caption(f"Over/Under: {over_under:+.1f}%")
        with col3:
            st.metric("MMM Predicted Total", f"{total_mmm_pred:,.0f}")
            mmm_diff = ((total_mmm_pred - total_backend_oo) / total_backend_oo * 100) if total_backend_oo > 0 else 0
            st.caption(f"Diff from Truth: {mmm_diff:+.1f}%")
else:
    st.info("Platform-attributed conversions (Online Order) not found in media data. Attribution comparison skipped.")

st.sidebar.header("Budget Optimization")
enable_optimization = st.sidebar.checkbox("Enable Budget Reallocation")

if enable_optimization:
    st.header("💡 Budget Reallocation")
    st.info("⚠️ Note: This optimization uses saturation only (ignores adstock carryover) for simplicity. It's a one-step prediction.")
    
    last_week_spend = df_model[media_cols].iloc[-1].values
    total_last_spend = last_week_spend.sum()
    
    col1, col2 = st.columns(2)
    with col1:
        total_budget = st.number_input(
            "Total Weekly Budget",
            min_value=0.0,
            value=float(total_last_spend),
            step=1000.0
        )
    
    with col2:
        st.info(f"Last week's total spend: ${total_last_spend:,.2f}")
    
    col1, col2 = st.columns(2)
    with col1:
        lower_bound_mult = st.slider(
            "Lower Bound (multiple of current spend)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1
        )
    
    with col2:
        upper_bound_mult = st.slider(
            "Upper Bound (multiple of current spend)",
            min_value=1.0,
            max_value=3.0,
            value=2.0,
            step=0.1
        )
    
    def objective(new_spend):
        saturated = apply_saturation(new_spend, saturation_beta)
        predicted_kpi = model.params['const']
        for i, col in enumerate(media_cols):
            predicted_kpi += model.params[col] * saturated[i]
        return -predicted_kpi
    
    def budget_constraint(new_spend):
        return total_budget - new_spend.sum()
    
    bounds = [(lower_bound_mult * last_week_spend[i], 
              upper_bound_mult * last_week_spend[i]) 
             for i in range(len(media_cols))]
    
    constraints = {'type': 'eq', 'fun': budget_constraint}
    
    initial_guess = last_week_spend * (total_budget / total_last_spend)
    
    result = minimize(
        objective,
        initial_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    if result.success:
        optimized_spend = result.x
        optimized_kpi = -result.fun
        
        saturated_last = apply_saturation(last_week_spend, saturation_beta)
        current_predicted_kpi = model.params['const']
        for i, col in enumerate(media_cols):
            current_predicted_kpi += model.params[col] * saturated_last[i]
        
        channel_names = [col.replace('Spend_', '') for col in media_cols]
        
        comparison_df = pd.DataFrame({
            'Channel': channel_names,
            'Current Spend': last_week_spend,
            'Optimized Spend': optimized_spend,
            'Change': optimized_spend - last_week_spend,
            'Change (%)': ((optimized_spend - last_week_spend) / last_week_spend * 100)
        })
        
        st.subheader("Optimized Allocation")
        st.dataframe(comparison_df.style.format({
            'Current Spend': '${:.2f}',
            'Optimized Spend': '${:.2f}',
            'Change': '${:.2f}',
            'Change (%)': '{:.2f}%'
        }), use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Predicted KPI", f"{current_predicted_kpi:.2f}")
        with col2:
            st.metric("Optimized Predicted KPI", f"{optimized_kpi:.2f}")
        with col3:
            improvement = ((optimized_kpi - current_predicted_kpi) / current_predicted_kpi * 100)
            st.metric("Improvement", f"{improvement:.2f}%")
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Current Spend',
            x=channel_names,
            y=last_week_spend
        ))
        fig.add_trace(go.Bar(
            name='Optimized Spend',
            x=channel_names,
            y=optimized_spend
        ))
        fig.update_layout(
            title='Current vs Optimized Spend',
            xaxis_title='Channel',
            yaxis_title='Spend',
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Optimization failed. Try adjusting the bounds or budget constraints.")
