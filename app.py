import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="TCB Reliability Dashboard", layout="wide")

st.title("TCB Micro-Bump Reliability Dashboard")
st.caption("Explainable reliability risk screening for 3D stacking thermocompression bonding")

np.random.seed(7)
time = np.linspace(0, 10, 200)

# -------------------------------------------------
# Golden baseline profiles
# -------------------------------------------------
golden_temp = 25 + 225 * (1 - np.exp(-time / 2.5))
golden_z = 100 - 35 * (1 - np.exp(-time / 1.8))
golden_force = 10 + 40 * (1 - np.exp(-time / 1.5))

# Simulated model reconstruction
# In real deployment, this would come from the 1D-CNN autoencoder.
reconstructed_temp = golden_temp + np.random.normal(0, 1.0, len(time))
reconstructed_z = golden_z + np.random.normal(0, 0.4, len(time))
reconstructed_force = golden_force + np.random.normal(0, 0.6, len(time))

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.header("Simulate Bonding Condition")

failure_mode = st.sidebar.selectbox(
    "Select scenario",
    ["Healthy", "Non-wet open risk", "Solder bridging risk", "Mixed anomaly"]
)

alignment_offset = st.sidebar.slider("X/Y alignment offset (µm)", 0.0, 10.0, 2.0)
theta_offset = st.sidebar.slider("Theta offset (degree)", 0.0, 2.0, 0.3)

# -------------------------------------------------
# Generate actual signals
# -------------------------------------------------
if failure_mode == "Healthy":
    actual_temp = golden_temp + np.random.normal(0, 2, len(time))
    actual_z = golden_z + np.random.normal(0, 0.8, len(time))
    actual_force = golden_force + np.random.normal(0, 1, len(time))

elif failure_mode == "Non-wet open risk":
    actual_temp = golden_temp - 25 + np.random.normal(0, 3, len(time))
    actual_z = golden_z + 8 + np.random.normal(0, 1.2, len(time))
    actual_force = golden_force - 5 + np.random.normal(0, 1.5, len(time))

elif failure_mode == "Solder bridging risk":
    actual_temp = golden_temp + np.random.normal(0, 2, len(time))
    actual_z = golden_z - 9 + np.random.normal(0, 1.2, len(time))
    actual_force = golden_force + 10 + np.random.normal(0, 2, len(time))

else:
    actual_temp = golden_temp - 18 + np.random.normal(0, 4, len(time))
    actual_z = golden_z - 6 + np.random.normal(0, 1.8, len(time))
    actual_force = golden_force + 8 + np.random.normal(0, 2.5, len(time))

# -------------------------------------------------
# Risk calculation
# -------------------------------------------------
temp_error = np.mean(np.abs(actual_temp - reconstructed_temp))
z_error = np.mean(np.abs(actual_z - reconstructed_z))
force_error = np.mean(np.abs(actual_force - reconstructed_force))
alignment_error = alignment_offset * 2.5 + theta_offset * 10

raw_score = (
    temp_error * 1.2
    + z_error * 3.0
    + force_error * 1.5
    + alignment_error
)

risk_score = min(100, int(raw_score * 2.2))

if risk_score <= 30:
    risk_level = "LOW"
    action = "Continue production"
elif risk_score <= 75:
    risk_level = "MEDIUM"
    action = "Tag unit/lot for enhanced inspection"
else:
    risk_level = "HIGH"
    action = "Hold affected unit/lot and trigger engineer review"

# -------------------------------------------------
# Failure mode risk mapping
# -------------------------------------------------
bridging_raw = (
    max(0, reconstructed_z.mean() - actual_z.mean()) * 4.0
    + force_error * 1.2
    + alignment_offset * 2.0
    + theta_offset * 12.0
)

nonwet_raw = (
    max(0, reconstructed_temp.mean() - actual_temp.mean()) * 1.5
    + max(0, actual_z.mean() - reconstructed_z.mean()) * 3.0
    + max(0, reconstructed_force.mean() - actual_force.mean()) * 1.0
)

bridging_score = min(100, int(bridging_raw * 2.0))
nonwet_score = min(100, int(nonwet_raw * 2.0))

def level_from_score(score):
    if score <= 30:
        return "LOW"
    elif score <= 75:
        return "MEDIUM"
    else:
        return "HIGH"

bridging_level = level_from_score(bridging_score)
nonwet_level = level_from_score(nonwet_score)

if bridging_score > nonwet_score and bridging_score > 30:
    predicted_issue = "Solder bridging / short risk due to possible over-compression or misalignment"
elif nonwet_score > bridging_score and nonwet_score > 30:
    predicted_issue = "Non-wet open / cold joint risk due to insufficient thermal energy or bump collapse"
elif alignment_offset > 6 or theta_offset > 1.2:
    predicted_issue = "Alignment-related open or bridging risk"
else:
    predicted_issue = "No major bonding reliability risk detected"

# -------------------------------------------------
# Top dashboard overview
# -------------------------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Lot ID", "LOT-A123")
col2.metric("Reliability Risk Score", f"{risk_score}/100")
col3.metric("Risk Category", risk_level)

st.info(f"Predicted Reliability Risk Direction: {predicted_issue}")

if risk_level == "HIGH":
    st.error(f"Recommended Action: {action}")
elif risk_level == "MEDIUM":
    st.warning(f"Recommended Action: {action}")
else:
    st.success(f"Recommended Action: {action}")

# -------------------------------------------------
# Signal comparison charts
# -------------------------------------------------
st.subheader("Signal Comparison: Actual vs Golden Baseline vs Model Reconstruction")

def plot_signal(title, actual, golden, reconstructed, ylabel):
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(time, golden, linestyle="--", label="Golden Baseline")
    ax.plot(time, reconstructed, linestyle=":", label="Model Reconstruction")
    ax.plot(time, actual, label="Actual Signal")

    # Highlight deviation region between actual and reconstructed
    ax.fill_between(
    time,
    actual,
    reconstructed,
    color="red",
    alpha=0.20,
    label="Deviation Area"
)

    ax.set_title(title)
    ax.set_xlabel("Time during TCB cycle")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)

c1, c2, c3 = st.columns(3)

with c1:
    plot_signal(
        "Temperature Profile",
        actual_temp,
        golden_temp,
        reconstructed_temp,
        "Temperature (°C)"
    )

with c2:
    plot_signal(
        "Z-axis Displacement",
        actual_z,
        golden_z,
        reconstructed_z,
        "Z Position / Bond Height"
    )

with c3:
    plot_signal(
        "Bonding Force / VCM Current",
        actual_force,
        golden_force,
        reconstructed_force,
        "Force Proxy"
    )

# -------------------------------------------------
# Key risk driver breakdown
# -------------------------------------------------
st.subheader("Key Risk Driver Breakdown")

raw_contributors = pd.DataFrame({
    "Risk Driver": [
        "Z-axis displacement deviation",
        "Temperature profile deviation",
        "Bonding force / VCM deviation",
        "Alignment / theta offset"
    ],
    "Raw Score": [
        z_error * 3.0,
        temp_error * 1.2,
        force_error * 1.5,
        alignment_error
    ]
})

total_score = raw_contributors["Raw Score"].sum()

if total_score > 0:
    raw_contributors["Contribution (%)"] = (
        raw_contributors["Raw Score"] / total_score * 100
    )
else:
    raw_contributors["Contribution (%)"] = 0

contributors = raw_contributors.sort_values(
    by="Contribution (%)",
    ascending=False
)

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(contributors["Risk Driver"], contributors["Contribution (%)"])
ax.set_ylabel("Contribution to Reliability Risk (%)")
ax.set_xlabel("Risk Driver")
ax.set_title("Signal Deviation Contribution")
ax.set_ylim(0, 100)

plt.xticks(rotation=0, ha="center", fontsize=8)
plt.tight_layout()

st.pyplot(fig)

st.write("Main signal drivers behind the elevated risk:")
for _, row in contributors.head(3).iterrows():
    st.write(f"- {row['Risk Driver']}: {row['Contribution (%)']:.1f}%")

# -------------------------------------------------
# Failure mode separation table
# -------------------------------------------------
st.subheader("Failure Mode Risk Mapping")

failure_mode_table = pd.DataFrame({
    "Failure Mode": [
        "Solder Bridging / Short",
        "Non-Wet Open / Cold Joint"
    ],
    "Risk Score": [
        f"{bridging_score}/100",
        f"{nonwet_score}/100"
    ],
    "Risk Level": [
        bridging_level,
        nonwet_level
    ],
    "Main Evidence": [
        "Z-axis over-travel, high force deviation, alignment/theta offset",
        "Slow thermal response, insufficient heating, high final bond height"
    ]
})

st.table(failure_mode_table)

# -------------------------------------------------
# Risk trend monitoring
# -------------------------------------------------
st.subheader("Risk Trend by Bonding Cycle / Lot")

cycle_ids = list(range(1, 11))

base_risks = np.clip(
    np.linspace(max(5, risk_score - 25), risk_score, 10)
    + np.random.normal(0, 5, 10),
    0,
    100
)

trend_data = pd.DataFrame({
    "Cycle": cycle_ids,
    "Reliability Risk Score": base_risks,
    "Temperature Deviation": np.random.normal(temp_error, 2, 10),
    "Z-axis Deviation": np.random.normal(z_error, 1, 10),
    "Force Deviation": np.random.normal(force_error, 1.5, 10)
})

st.line_chart(trend_data.set_index("Cycle")[["Reliability Risk Score"]])

trend_col1, trend_col2, trend_col3 = st.columns(3)

trend_col1.metric("Latest Risk", f"{risk_score}/100")
trend_col2.metric("10-Cycle Avg Risk", f"{trend_data['Reliability Risk Score'].mean():.1f}/100")

if trend_data["Reliability Risk Score"].iloc[-1] > trend_data["Reliability Risk Score"].iloc[0] + 15:
    trend_status = "Drifting Up"
else:
    trend_status = "Stable"

trend_col3.metric("Trend Status", trend_status)

if trend_status == "Drifting Up":
    st.warning(
        "Process drift detected: reliability risk has increased over recent bonding cycles. "
        "Possible causes include bonding head wear, heater instability, alignment calibration drift, "
        "chuck contamination, or thermal contact degradation."
    )
else:
    st.success("No severe process drift detected over recent bonding cycles.")

# -------------------------------------------------
# Engineering interpretation
# -------------------------------------------------
st.subheader("Engineering Interpretation")

if bridging_score > nonwet_score and bridging_score > 30:
    st.write("""
    The current bonding cycle shows excessive Z-axis displacement and/or elevated alignment offset.
    This indicates possible over-compression and asymmetric solder squeeze-out, increasing the risk
    of solder bridging between adjacent micro-bumps.
    """)

elif nonwet_score > bridging_score and nonwet_score > 30:
    st.write("""
    The current bonding cycle shows delayed thermal response and/or insufficient bump collapse.
    This indicates possible incomplete solder wetting, increasing the risk of non-wet open or cold joint formation.
    """)

elif alignment_offset > 6 or theta_offset > 1.2:
    st.write("""
    The current bonding cycle shows alignment-related deviation before bonding.
    This may cause off-centre compression, increasing the risk of open joints or solder bridging.
    """)

else:
    st.write("""
    The bonding cycle closely follows the expected healthy bonding behaviour.
    No major reliability risk is detected from the available TCB process signals.
    """)

# -------------------------------------------------
# Historical comparison
# -------------------------------------------------
st.subheader("Historical Similar Case Comparison")

if risk_level == "HIGH":
    similar_cases = pd.DataFrame({
        "Outcome": ["Defect Associated", "No Defect Found"],
        "Percentage": [78, 22]
    })
    st.write("Similar historical bonding cycles showed strong association with reliability-related defects.")
elif risk_level == "MEDIUM":
    similar_cases = pd.DataFrame({
        "Outcome": ["Defect Associated", "No Defect Found"],
        "Percentage": [35, 65]
    })
    st.write("Similar historical bonding cycles showed moderate association with reliability-related defects.")
else:
    similar_cases = pd.DataFrame({
        "Outcome": ["Defect Associated", "No Defect Found"],
        "Percentage": [8, 92]
    })
    st.write("Similar historical bonding cycles were mostly within normal production limits.")

st.bar_chart(similar_cases.set_index("Outcome"))

# -------------------------------------------------
# Suggested process adjustment
# -------------------------------------------------
st.subheader("Suggested Process Adjustment")

if bridging_score > nonwet_score and bridging_score > 30:
    st.write("""
    Recommended adjustment:
    - Review bonding force setting
    - Check Z-axis compression limit
    - Verify X/Y/theta alignment calibration
    - Inspect affected unit or lot for possible solder bridging
    """)

elif nonwet_score > bridging_score and nonwet_score > 30:
    st.write("""
    Recommended adjustment:
    - Review temperature ramp rate
    - Confirm peak bonding temperature
    - Check thermal transfer stability
    - Inspect affected unit or lot for non-wet open or cold joint risk
    """)

elif alignment_offset > 6 or theta_offset > 1.2:
    st.write("""
    Recommended adjustment:
    - Recalibrate alignment stage
    - Check X/Y/theta offset before bonding
    - Perform enhanced optical inspection
    """)

else:
    st.write("""
    Recommended adjustment:
    - No immediate process correction required
    - Continue normal monitoring
    """)