import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="TCB Reliability Dashboard", layout="wide")

# -------------------------------------------------
# Simple status card helper
# -------------------------------------------------
def card(title, text, style="info"):
    hide_title_list = [
        "Engineering Interpretation",
        "Recommended Adjustment",
        "Historical Comparison"
    ]

    if title in hide_title_list:
        message = text
    else:
        message = f"### {title}\n\n{text}"

    if style == "good":
        st.success(message)
    elif style == "warn":
        st.warning(message)
    elif style == "bad":
        st.error(message)
    else:
        st.info(message)


# -------------------------------------------------
# Title
# -------------------------------------------------
st.title("TCB Micro-Bump Reliability Dashboard")
st.caption("Explainable reliability risk screening for 3D stacking thermocompression bonding")

np.random.seed(7)
time = np.linspace(0, 10, 200)

# -------------------------------------------------
# Sidebar: Golden Profile Context
# -------------------------------------------------
st.sidebar.header("Golden Profile Context")

tool_id = st.sidebar.selectbox(
    "Tool ID",
    ["TCB-01", "TCB-02", "TCB-03"]
)

recipe_id = st.sidebar.selectbox(
    "Recipe ID",
    ["Recipe-R1", "Recipe-R2"]
)

product_family = st.sidebar.selectbox(
    "Product / Stack Family",
    ["Stack-A", "Stack-B"]
)

baseline_status = "Engineer-approved baseline"

# -------------------------------------------------
# Golden Profile: context-specific healthy envelope
# -------------------------------------------------
tool_factor = {
    "TCB-01": 0.0,
    "TCB-02": 2.0,
    "TCB-03": -2.0
}[tool_id]

recipe_factor = {
    "Recipe-R1": 0.0,
    "Recipe-R2": 5.0
}[recipe_id]

product_factor = {
    "Stack-A": 0.0,
    "Stack-B": 3.0
}[product_family]

# Golden baseline curves
golden_temp = 25 + (225 + recipe_factor) * (1 - np.exp(-time / 2.5)) + tool_factor
golden_z = 100 - (35 + product_factor) * (1 - np.exp(-time / 1.8))
golden_force = 10 + (40 + recipe_factor) * (1 - np.exp(-time / 1.5))

# Healthy variation envelope
temp_sigma = 5
z_sigma = 2
force_sigma = 3

temp_upper = golden_temp + temp_sigma
temp_lower = golden_temp - temp_sigma

z_upper = golden_z + z_sigma
z_lower = golden_z - z_sigma

force_upper = golden_force + force_sigma
force_lower = golden_force - force_sigma

# Simulated model reconstruction
# In real deployment, this comes from the 1D-CNN autoencoder.
reconstructed_temp = golden_temp + np.random.normal(0, 1.0, len(time))
reconstructed_z = golden_z + np.random.normal(0, 0.4, len(time))
reconstructed_force = golden_force + np.random.normal(0, 0.6, len(time))

# -------------------------------------------------
# Sidebar: Simulate Bonding Condition
# -------------------------------------------------
st.sidebar.header("Simulate Bonding Condition")

failure_mode = st.sidebar.selectbox(
    "Select scenario",
    [
        "Healthy",
        "Non-wet open risk",
        "Solder bridging risk",
        "VCM-only anomaly",
        "Mixed anomaly"
    ]
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

elif failure_mode == "VCM-only anomaly":
    actual_temp = golden_temp + np.random.normal(0, 2, len(time))
    actual_z = golden_z + np.random.normal(0, 0.8, len(time))
    actual_force = golden_force + 14 + np.random.normal(0, 2.2, len(time))

else:
    actual_temp = golden_temp - 18 + np.random.normal(0, 4, len(time))
    actual_z = golden_z - 6 + np.random.normal(0, 1.8, len(time))
    actual_force = golden_force + 8 + np.random.normal(0, 2.5, len(time))

# -------------------------------------------------
# Signal errors
# -------------------------------------------------
def envelope_deviation(actual, lower, upper):
    below = np.maximum(lower - actual, 0)
    above = np.maximum(actual - upper, 0)
    return np.mean(below + above)


# Reconstruction error = model anomaly evidence
temp_recon_error = np.mean(np.abs(actual_temp - reconstructed_temp))
z_recon_error = np.mean(np.abs(actual_z - reconstructed_z))
force_recon_error = np.mean(np.abs(actual_force - reconstructed_force))

# Envelope error = how far signal leaves healthy operating range
temp_envelope_error = envelope_deviation(actual_temp, temp_lower, temp_upper)
z_envelope_error = envelope_deviation(actual_z, z_lower, z_upper)
force_envelope_error = envelope_deviation(actual_force, force_lower, force_upper)

# Combined error for risk scoring
temp_error = 0.7 * temp_recon_error + 0.3 * temp_envelope_error
z_error = 0.7 * z_recon_error + 0.3 * z_envelope_error
force_error = 0.7 * force_recon_error + 0.3 * force_envelope_error

alignment_error = alignment_offset * 2.5 + theta_offset * 10

# Directional indicators
z_overtravel = reconstructed_z.mean() - actual_z.mean()
z_undertravel = actual_z.mean() - reconstructed_z.mean()
thermal_lag = reconstructed_temp.mean() - actual_temp.mean()
force_high = actual_force.mean() - reconstructed_force.mean()
force_low = reconstructed_force.mean() - actual_force.mean()

# Thresholds for consistency logic
z_abnormal = z_error > 4.0
force_abnormal = force_error > 6.0
temp_abnormal = temp_error > 8.0
alignment_abnormal = alignment_offset > 6.0 or theta_offset > 1.2

z_overtravel_detected = z_overtravel > 3.0
z_undertravel_detected = z_undertravel > 3.0
force_high_detected = force_high > 5.0

# -------------------------------------------------
# Mechanical Signal Consistency Check
# -------------------------------------------------
if force_abnormal and z_abnormal:
    mechanical_consistency_status = "Consistent mechanical bonding anomaly detected"
    mechanical_consistency_detail = (
        "VCM/force deviation and Z-axis displacement deviation are both abnormal. "
        "This suggests the abnormal force response is consistent with actual bonding mechanics, "
        "not just tool-side behaviour."
    )
    mechanical_consistency_style = "bad"

elif force_abnormal and not z_abnormal:
    mechanical_consistency_status = "Possible tool or actuator issue"
    mechanical_consistency_detail = (
        "VCM/force deviation is abnormal, but Z-axis displacement remains within the healthy envelope. "
        "This weakens direct micro-bump reliability evidence and suggests checking VCM calibration, "
        "actuator health, friction, or controller stability."
    )
    mechanical_consistency_style = "warn"

elif z_abnormal and not force_abnormal:
    mechanical_consistency_status = "Possible displacement-driven bonding anomaly"
    mechanical_consistency_detail = (
        "Z-axis displacement is abnormal, but VCM/force response does not strongly support it. "
        "Inspect final bond height trend and verify displacement sensing."
    )
    mechanical_consistency_style = "warn"

else:
    mechanical_consistency_status = "Mechanical signals consistent with healthy profile"
    mechanical_consistency_detail = (
        "VCM/force response and Z-axis displacement are both within the expected healthy process envelope."
    )
    mechanical_consistency_style = "good"

# -------------------------------------------------
# Failure mode risk mapping with cross-validation
# -------------------------------------------------
bridging_raw = 0

if z_overtravel_detected:
    bridging_raw += z_overtravel * 5.0

if force_high_detected and z_overtravel_detected:
    bridging_raw += force_high * 1.5
elif force_high_detected and not z_overtravel_detected:
    bridging_raw += force_high * 0.25

if alignment_abnormal:
    bridging_raw += alignment_offset * 2.0 + theta_offset * 12.0

nonwet_raw = 0

if thermal_lag > 5.0:
    nonwet_raw += thermal_lag * 2.0

if z_undertravel_detected:
    nonwet_raw += z_undertravel * 4.0

if force_low > 3.0 and z_undertravel_detected:
    nonwet_raw += force_low * 1.0

bridging_score = min(95, int(bridging_raw * 1.7))
nonwet_score = min(95, int(nonwet_raw * 1.8))


def level_from_score(score):
    if score <= 30:
        return "LOW"
    elif score <= 75:
        return "MEDIUM"
    else:
        return "HIGH"


bridging_level = level_from_score(bridging_score)
nonwet_level = level_from_score(nonwet_score)

# Overall risk score
raw_score = (
    temp_error * 1.2
    + z_error * 3.0
    + force_error * 1.0
    + alignment_error
)

# If VCM is abnormal but Z is normal, reduce overclaiming direct reliability failure.
if force_abnormal and not z_abnormal:
    raw_score *= 0.75

risk_score = min(95, int(raw_score * 2.2))

if risk_score <= 30:
    risk_level = "LOW"
    action = "Continue production"
elif risk_score <= 75:
    risk_level = "MEDIUM"
    action = "Tag unit/lot for enhanced inspection"
else:
    risk_level = "HIGH"
    action = "Hold affected unit/lot and trigger engineer review"

if bridging_score > nonwet_score and bridging_score > 30:
    predicted_issue = "Solder bridging / short risk supported by mechanical signal cross-validation"
elif nonwet_score > bridging_score and nonwet_score > 30:
    predicted_issue = "Non-wet open / cold joint risk due to insufficient thermal energy or bump collapse"
elif force_abnormal and not z_abnormal:
    predicted_issue = "VCM/force anomaly detected, but Z-axis remains normal; possible tool or actuator issue"
elif alignment_abnormal:
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

if risk_level == "HIGH":
    card("Predicted Reliability Risk Direction", predicted_issue, "bad")
    card("Recommended Action", action, "bad")
elif risk_level == "MEDIUM":
    card("Predicted Reliability Risk Direction", predicted_issue, "warn")
    card("Recommended Action", action, "warn")
else:
    card("Predicted Reliability Risk Direction", predicted_issue, "good")
    card("Recommended Action", action, "good")

# -------------------------------------------------
# Golden Profile Basis table
# -------------------------------------------------
st.subheader("Golden Profile Basis")

profile_table = pd.DataFrame({
    "Profile Item": [
        "Tool ID",
        "Recipe ID",
        "Product Family",
        "Reference Data",
        "Baseline Status",
        "Update Policy"
    ],
    "Value": [
        tool_id,
        recipe_id,
        product_family,
        "Verified healthy TCB cycles",
        baseline_status,
        "Automatically suggested, engineer-approved release"
    ]
})

st.table(profile_table)

# -------------------------------------------------
# Signal comparison charts
# -------------------------------------------------
st.subheader("Signal Comparison: Actual vs Golden Baseline vs Model Reconstruction")


def plot_signal(title, actual, golden, reconstructed, lower, upper, ylabel):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    # Dark modern background
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#161A23")

    # Healthy envelope
    ax.fill_between(
        time,
        lower,
        upper,
        color="#4A5568",
        alpha=0.24,
        label="Healthy Envelope"
    )

    # Golden Baseline = orange
    ax.plot(
        time,
        golden,
        linestyle="--",
        linewidth=2.2,
        color="#F77737",
        label="Golden Baseline"
    )

    # Model Reconstruction = pink
    ax.plot(
        time,
        reconstructed,
        linestyle=":",
        linewidth=2.4,
        color="#E1306C",
        label="Model Reconstruction"
    )

    # Actual Signal = green
    ax.plot(
        time,
        actual,
        linewidth=2.5,
        color="#25D366",
        label="Actual Signal"
    )

    # Deviation area = red
    ax.fill_between(
        time,
        actual,
        reconstructed,
        color="#FF4D4F",
        alpha=0.24,
        label="Deviation Area"
    )

    ax.set_title(title, color="white", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time during TCB cycle", color="white")
    ax.set_ylabel(ylabel, color="white")

    ax.grid(True, color="#2D3748", alpha=0.38, linestyle="--", linewidth=0.7)
    ax.tick_params(colors="white")

    for spine in ax.spines.values():
        spine.set_color("#A0AEC0")

    legend = ax.legend(
        fontsize=8,
        facecolor="#1A202C",
        edgecolor="#4A5568",
        labelcolor="white"
    )

    for text in legend.get_texts():
        text.set_color("white")

    plt.tight_layout()
    st.pyplot(fig)


c1, c2, c3 = st.columns(3)

with c1:
    plot_signal(
        "Temperature Profile",
        actual_temp,
        golden_temp,
        reconstructed_temp,
        temp_lower,
        temp_upper,
        "Temperature (°C)"
    )

with c2:
    plot_signal(
        "Z-axis Displacement",
        actual_z,
        golden_z,
        reconstructed_z,
        z_lower,
        z_upper,
        "Z Position / Bond Height"
    )

with c3:
    plot_signal(
        "Bonding Force / VCM Current",
        actual_force,
        golden_force,
        reconstructed_force,
        force_lower,
        force_upper,
        "Force Proxy"
    )

# -------------------------------------------------
# Golden Profile Governance
# -------------------------------------------------
st.subheader("Golden Profile Governance")

card(
    "Engineer-Approved Baseline",
    "The Golden Profile is automatically constructed from verified healthy cycles, but it should not be released blindly. Engineers review and approve the baseline to prevent slow tool drift from being learned as normal behavior.",
    "info"
)

# -------------------------------------------------
# Mechanical consistency check
# -------------------------------------------------
st.subheader("Mechanical Signal Consistency Check")

card(
    mechanical_consistency_status,
    mechanical_consistency_detail,
    mechanical_consistency_style
)

mechanical_table = pd.DataFrame({
    "Signal Condition": [
        "VCM abnormal + Z-axis abnormal",
        "VCM abnormal + Z-axis normal",
        "Z-axis abnormal + VCM normal",
        "Both normal"
    ],
    "Dashboard Interpretation": [
        "Consistent mechanical bonding anomaly detected",
        "Possible tool/actuator issue — verify VCM calibration",
        "Possible displacement-driven bonding anomaly — inspect bond height trend",
        "Mechanical signals consistent with healthy profile"
    ]
})

st.table(mechanical_table)

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
        force_error * 1.0,
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

fig, ax = plt.subplots(figsize=(10, 4.2))

fig.patch.set_facecolor("#0F1117")
ax.set_facecolor("#161A23")

bar_colors = ["#25D366", "#F77737", "#E1306C", "#833AB4"]

ax.bar(
    contributors["Risk Driver"],
    contributors["Contribution (%)"],
    color=bar_colors[:len(contributors)]
)

ax.set_ylabel("Contribution to Reliability Risk (%)", color="white")
ax.set_xlabel("Risk Driver", color="white")
ax.set_title("Signal Deviation Contribution", color="white", fontsize=12, fontweight="bold")
ax.set_ylim(0, 100)

ax.grid(True, axis="y", color="#2D3748", alpha=0.38, linestyle="--", linewidth=0.7)
ax.tick_params(colors="white")

for spine in ax.spines.values():
    spine.set_color("#A0AEC0")

plt.xticks(rotation=0, ha="center", fontsize=8)
plt.tight_layout()

st.pyplot(fig)

# -------------------------------------------------
# Main signal drivers table
# -------------------------------------------------
st.subheader("Main Signal Drivers Behind Elevated Risk")

driver_table = contributors.copy()

driver_table["Rank"] = [str(i) for i in range(1, len(driver_table) + 1)]
driver_table["Contribution (%)"] = driver_table["Contribution (%)"].apply(lambda x: f"{x:.1f}%")

driver_table = driver_table[["Rank", "Risk Driver", "Contribution (%)"]]

st.dataframe(
    driver_table,
    hide_index=True,
    use_container_width=True
)

# -------------------------------------------------
# Failure mode separation table
# -------------------------------------------------
st.subheader("Failure Mode Risk Mapping")

if bridging_score > 75:
    bridging_evidence = "Z-axis over-travel + low bond height behaviour + VCM support + alignment/theta offset"
elif bridging_score > 30:
    bridging_evidence = "Partial mechanical evidence; verify Z-axis and final bond height"
else:
    bridging_evidence = "No strong bridging evidence from cross-validated mechanical signals"

if nonwet_score > 75:
    nonwet_evidence = "Delayed thermal response + insufficient bump collapse + weak force support"
elif nonwet_score > 30:
    nonwet_evidence = "Partial thermal or collapse evidence; inspect non-wet open risk"
else:
    nonwet_evidence = "No strong non-wet open evidence from thermal and Z-axis signals"

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
        bridging_evidence,
        nonwet_evidence
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
    card(
        "Process Drift Warning",
        "Reliability risk has increased over recent bonding cycles. Possible causes include bonding head wear, heater instability, alignment calibration drift, chuck contamination, or thermal contact degradation.",
        "warn"
    )
else:
    card(
        "Trend Status",
        "No severe process drift detected over recent bonding cycles.",
        "good"
    )

# -------------------------------------------------
# Engineering interpretation
# -------------------------------------------------
st.subheader("Engineering Interpretation")

if force_abnormal and not z_abnormal:
    card(
        "Engineering Interpretation",
        "VCM/force response is abnormal, but Z-axis displacement remains within the expected healthy envelope. This suggests the event may be related to tool-side behaviour such as actuator friction, calibration drift, or controller instability rather than direct micro-bump collapse.",
        "warn"
    )

elif bridging_score > nonwet_score and bridging_score > 30:
    card(
        "Engineering Interpretation",
        "The current bonding cycle shows cross-validated mechanical evidence: Z-axis over-travel is supported by abnormal VCM/force behaviour and/or alignment offset. This indicates possible over-compression and asymmetric solder squeeze-out, increasing the risk of solder bridging between adjacent micro-bumps.",
        "bad" if bridging_score > 75 else "warn"
    )

elif nonwet_score > bridging_score and nonwet_score > 30:
    card(
        "Engineering Interpretation",
        "The current bonding cycle shows delayed thermal response and/or insufficient bump collapse. This indicates possible incomplete solder wetting, increasing the risk of non-wet open or cold joint formation.",
        "bad" if nonwet_score > 75 else "warn"
    )

elif alignment_abnormal:
    card(
        "Engineering Interpretation",
        "The current bonding cycle shows alignment-related deviation before bonding. This may cause off-centre compression, increasing the risk of open joints or solder bridging.",
        "warn"
    )

else:
    card(
        "Engineering Interpretation",
        "The bonding cycle closely follows the expected healthy bonding behaviour. No major reliability risk is detected from the available TCB process signals.",
        "good"
    )

# -------------------------------------------------
# Historical comparison
# -------------------------------------------------
st.subheader("Historical Similar Case Comparison")

if risk_level == "HIGH":
    similar_cases = pd.DataFrame({
        "Outcome": ["Defect Associated", "No Defect Found"],
        "Percentage": [78, 22]
    })
    historical_text = "Similar historical bonding cycles showed strong association with reliability-related defects."
    historical_style = "bad"
elif risk_level == "MEDIUM":
    similar_cases = pd.DataFrame({
        "Outcome": ["Defect Associated", "No Defect Found"],
        "Percentage": [35, 65]
    })
    historical_text = "Similar historical bonding cycles showed moderate association with reliability-related defects."
    historical_style = "warn"
else:
    similar_cases = pd.DataFrame({
        "Outcome": ["Defect Associated", "No Defect Found"],
        "Percentage": [8, 92]
    })
    historical_text = "Similar historical bonding cycles were mostly within normal production limits."
    historical_style = "good"

card("Historical Comparison", historical_text, historical_style)
st.bar_chart(similar_cases.set_index("Outcome"))

# -------------------------------------------------
# Suggested process adjustment
# -------------------------------------------------
st.subheader("Suggested Process Adjustment")

if force_abnormal and not z_abnormal:
    card(
        "Recommended Adjustment",
        "Verify VCM calibration, inspect actuator friction or motor behaviour, and monitor the next bonding cycles. Do not immediately classify this as micro-bump reliability failure unless Z-axis or bond-height deviation also appears.",
        "warn"
    )

elif bridging_score > nonwet_score and bridging_score > 30:
    card(
        "Recommended Adjustment",
        "Review bonding force setting, check Z-axis compression limit, verify X/Y/theta alignment calibration, and inspect affected unit or lot for possible solder bridging.",
        "bad" if bridging_score > 75 else "warn"
    )

elif nonwet_score > bridging_score and nonwet_score > 30:
    card(
        "Recommended Adjustment",
        "Review temperature ramp rate, confirm peak bonding temperature, check thermal transfer stability, and inspect affected unit or lot for non-wet open or cold joint risk.",
        "bad" if nonwet_score > 75 else "warn"
    )

elif alignment_abnormal:
    card(
        "Recommended Adjustment",
        "Recalibrate alignment stage, check X/Y/theta offset before bonding, and perform enhanced optical inspection.",
        "warn"
    )

else:
    card(
        "Recommended Adjustment",
        "No immediate process correction required. Continue normal monitoring.",
        "good"
    )
