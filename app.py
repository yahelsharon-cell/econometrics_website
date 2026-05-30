import streamlit as st
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import io

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Data Analytics", layout="wide")

# ── Dark theme ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #0E1117; color: #F2F2F7; }
[data-testid="stMetricValue"] { color: #0A84FF !important; }
.metric-card {
    background-color: #1C1C1E;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #2C2C2E;
}
.reg-table {
    width: 100%; border-collapse: collapse; font-size: 13.5px;
    background-color: #1C1C1E; border-radius: 10px; overflow: hidden;
}
.reg-table th {
    background-color: #2C2C2E; color: #8E8E93; font-size: 11px;
    font-weight: 600; text-transform: uppercase; letter-spacing: 0.6px;
    padding: 10px 14px; text-align: left;
}
.reg-table td {
    padding: 10px 14px; color: #F2F2F7;
    border-bottom: 1px solid #2C2C2E; font-variant-numeric: tabular-nums;
}
.reg-table tr:last-child td { border-bottom: none; }
.sig { color: #0A84FF; font-weight: 700; }
.insig { color: #636366; }
.prof-box {
    background-color: #1C1C1E; border-left: 3px solid #0A84FF;
    border-radius: 0 10px 10px 0; padding: 16px 20px; margin-top: 8px;
}
.prof-box p { color: #F2F2F7 !important; margin: 4px 0; font-size: 14px; line-height: 1.6; }
.prof-box .label { color: #636366 !important; font-size: 11px;
    text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📂 Data Import")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file:
        df_raw = pd.read_csv(uploaded_file)
        st.session_state['df_raw'] = df_raw
        st.success(f"Dataset loaded! ({len(df_raw):,} rows)")

    # ── Situation filter (only shown if column exists) ────────────────────
    if 'df_raw' in st.session_state:
        df_raw = st.session_state['df_raw']

        sit_col = next(
            (c for c in df_raw.columns if c.lower() == 'situation'), None
        )

        if sit_col:
            unique_situations = sorted(df_raw[sit_col].dropna().unique().tolist())
            default_idx = unique_situations.index('all') if 'all' in unique_situations else 0

            chosen = st.selectbox(
                "Filter by Game Situation",
                options=unique_situations,
                index=default_idx,
            )
            st.session_state['df'] = df_raw[df_raw[sit_col] == chosen].reset_index(drop=True)
            st.caption(f"{len(st.session_state['df']):,} rows after filtering.")
        else:
            st.session_state['df'] = df_raw

    if 'df' in st.session_state:
        df = st.session_state['df']
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        st.markdown("---")
        st.markdown("### 🧮 Regression Setup")

        y_var = st.selectbox("Dependent variable (Y)", options=numeric_cols, index=0)
        x_vars = st.multiselect(
            "Independent variables (X)",
            options=[c for c in numeric_cols if c != y_var],
            default=[c for c in numeric_cols if c != y_var][:2],
        )

        label_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        hover_options = ['None (Use Row Index)'] + label_cols
        hover_label = st.selectbox("Hover Label (Optional)", options=hover_options)

        st.session_state['y_var'] = y_var
        st.session_state['x_vars'] = x_vars
        st.session_state['hover_label'] = hover_label

# ── Main content ──────────────────────────────────────────────────────────────
st.title("📊 Data Analytics Dashboard")

tab1, tab2, tab3 = st.tabs(["📈 Descriptives", "🔧 Variable Builder", "🧮 Regression"])

# ════════════════════════════════════════════════════════════════════════════
#  TAB 1 — DESCRIPTIVES
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    if 'df' not in st.session_state:
        st.warning("Upload a CSV file in the sidebar to get started.")
    else:
        df = st.session_state['df']

        # ── Summary metrics ──────────────────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Rows",    len(df))
        c2.metric("Total Columns", len(df.columns))
        c3.metric("Null Values",   int(df.isna().sum().sum()))

        # ── Clean data button ────────────────────────────────────────────────
        rows_before = len(df)
        if st.button("🧹 Clean Data  (drop rows with NaN)"):
            df_clean = df.dropna()
            rows_dropped = rows_before - len(df_clean)
            st.session_state['df'] = df_clean
            df = df_clean
            if rows_dropped > 0:
                st.success(f"Removed {rows_dropped:,} row(s) containing NaN values. "
                           f"{len(df_clean):,} rows remain.")
            else:
                st.info("No NaN values found — dataset is already clean.")

        # ── Data preview ─────────────────────────────────────────────────────
        st.markdown("### Data Preview")
        st.dataframe(df.head(20), use_container_width=True)

        # ── Correlation heatmap ──────────────────────────────────────────────
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) >= 2:
            st.markdown("### Correlation Matrix")

            MAX_COLS = 15
            if len(numeric_cols) > MAX_COLS:
                st.warning(
                    f"Your dataset has {len(numeric_cols)} numeric columns. "
                    f"The heatmap is limited to the first {MAX_COLS} to avoid memory issues."
                )
            plot_cols = numeric_cols[:MAX_COLS]
            corr = df[plot_cols].corr()

            fig, ax = plt.subplots(figsize=(max(6, len(plot_cols)), max(4, len(plot_cols) * 0.8)))
            fig.patch.set_facecolor('#0E1117')
            ax.set_facecolor('#0E1117')

            sns.heatmap(
                corr,
                annot=True,
                fmt=".2f",
                cmap="Blues",
                linewidths=0.4,
                linecolor='#2C2C2E',
                ax=ax,
                annot_kws={"size": 9, "color": "#F2F2F7"},
                cbar_kws={"shrink": 0.8},
                xticklabels=plot_cols,
                yticklabels=plot_cols,
            )
            ax.tick_params(colors='#8E8E93', labelsize=9)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', color='#8E8E93')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, color='#8E8E93')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

# ════════════════════════════════════════════════════════════════════════════
#  TAB 2 — VARIABLE BUILDER
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    if 'df' not in st.session_state:
        st.warning("Upload a CSV file in the sidebar to get started.")
    else:
        df = st.session_state['df']
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        st.subheader("Variable Builder")
        st.caption(
            "Combine or transform existing numeric columns to create a new variable. "
            "Once created, it will be available in the Regression tab."
        )

        if not numeric_cols:
            st.error("No numeric columns found in the dataset.")
        else:
            # ── Inputs ───────────────────────────────────────────────────────
            col_name, col_op = st.columns(2)
            with col_name:
                new_var_name = st.text_input("New Variable Name", placeholder="e.g., log_income")
            with col_op:
                operation = st.selectbox(
                    "Operation",
                    options=[
                        "Add  (A + B)",
                        "Subtract  (A − B)",
                        "Multiply  (A × B)",
                        "Divide  (A ÷ B)",
                        "Natural Log  (ln A)",
                    ],
                )

            is_unary = operation.startswith("Natural Log")

            col_a, col_b = st.columns(2)
            with col_a:
                var_a = st.selectbox("Variable A", options=numeric_cols, key="vb_var_a")
            with col_b:
                if not is_unary:
                    b_options = [c for c in numeric_cols if c != var_a]
                    var_b = st.selectbox("Variable B", options=b_options, key="vb_var_b")
                else:
                    st.markdown(
                        "<div style='padding-top:28px;color:#636366;font-size:13px;'>"
                        "Applies ln() to Variable A only.</div>",
                        unsafe_allow_html=True,
                    )
                    var_b = None

            # ── Create button ─────────────────────────────────────────────────
            if st.button("➕  Create Variable", type="primary"):
                name = new_var_name.strip()
                if not name:
                    st.error("Please enter a name for the new variable.")
                elif name in df.columns:
                    st.warning(f"A column named **{name}** already exists. Choose a different name.")
                else:
                    try:
                        if operation.startswith("Add"):
                            df[name] = df[var_a] + df[var_b]
                        elif operation.startswith("Subtract"):
                            df[name] = df[var_a] - df[var_b]
                        elif operation.startswith("Multiply"):
                            df[name] = df[var_a] * df[var_b]
                        elif operation.startswith("Divide"):
                            if (df[var_b] == 0).any():
                                st.error(
                                    f"**{var_b}** contains zeros — division by zero would produce "
                                    "infinite values. Choose a different variable or clean the data first."
                                )
                                st.stop()
                            df[name] = df[var_a] / df[var_b]
                        elif is_unary:
                            if (df[var_a] <= 0).any():
                                st.error(
                                    f"**{var_a}** contains zero or negative values. "
                                    "Natural log requires strictly positive values."
                                )
                                st.stop()
                            df[name] = np.log(df[var_a])

                        st.session_state['df'] = df
                        st.success(
                            f"✅ **{name}** created successfully and added to the dataset. "
                            "It is now available as a variable in the Regression tab."
                        )
                        st.rerun()

                    except Exception as e:
                        st.error(f"Failed to create variable: {e}")

            # ── Created variables table ───────────────────────────────────────
            original_cols = set(st.session_state['df_raw'].columns) if 'df_raw' in st.session_state else set()
            custom_cols   = [c for c in df.columns if c not in original_cols]

            if custom_cols:
                st.markdown("---")
                st.markdown("#### Created Variables")
                preview = df[custom_cols].describe().T[['mean', 'min', 'max', 'std']].round(4)
                st.dataframe(preview, use_container_width=True)

                with st.expander("Remove a created variable"):
                    to_remove = st.selectbox("Select variable to remove", options=custom_cols, key="vb_remove")
                    if st.button("🗑  Remove Variable", key="vb_remove_btn"):
                        df = st.session_state['df'].drop(columns=[to_remove])
                        st.session_state['df'] = df
                        st.success(f"**{to_remove}** removed.")
                        st.rerun()

# ════════════════════════════════════════════════════════════════════════════
#  TAB 3 — REGRESSION
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    if 'df' not in st.session_state:
        st.warning("Upload data first using the sidebar.")
    elif not st.session_state.get('x_vars'):
        st.info("Select at least one independent variable (X) in the sidebar.")
    else:
        df          = st.session_state['df']
        y_var       = st.session_state['y_var']
        x_vars      = st.session_state['x_vars']
        hover_label = st.session_state.get('hover_label', 'None (Use Row Index)')

        st.subheader("OLS Regression Analysis")
        st.caption(f"**Y:** {y_var}   |   **X:** {', '.join(x_vars)}")

        run = st.button("▶  Run Analysis", type="primary")

        if run:
            try:
                # ── 1. Define columns and build reg_data ─────────────────────
                selected_label = None if hover_label == 'None (Use Row Index)' else hover_label
                cols_to_clean  = [y_var] + x_vars + ([selected_label] if selected_label else [])
                reg_data       = df[cols_to_clean].dropna()

                if len(reg_data) < len(x_vars) + 2:
                    st.error(f"Not enough complete observations ({len(reg_data)}) "
                             f"for {len(x_vars)} predictor(s).")
                    st.stop()

                # ── 2. Summary stats on reg_data (only chosen variables) ──────
                summary_stats = reg_data[[y_var] + x_vars].describe().T[['mean', 'min', 'max', '50%', 'std']]
                summary_stats = summary_stats.rename(columns={'50%': 'median'})
                st.markdown("### 📈 Variable Summary Statistics")
                st.dataframe(summary_stats, use_container_width=True)

                # ── 3. Fit OLS on reg_data ────────────────────────────────────
                Y     = reg_data[y_var]
                X     = sm.add_constant(reg_data[x_vars])
                model = sm.OLS(Y, X).fit()

                # ── Fit metrics ──────────────────────────────────────────────
                st.markdown("#### Model Fit")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("R²",           f"{model.rsquared:.4f}")
                m2.metric("Adj. R²",      f"{model.rsquared_adj:.4f}")
                m3.metric("F-statistic",  f"{model.fvalue:.3f}")
                m4.metric("Prob (F)",     f"{model.f_pvalue:.4f}")
                m5.metric("Observations", f"{int(model.nobs):,}")

                # ── Coefficient table ────────────────────────────────────────
                st.markdown("#### Coefficients")
                ci = model.conf_int()
                rows_html = ""
                for var in model.params.index:
                    p         = model.pvalues[var]
                    stars     = ("***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "")
                    sig_class = "sig" if stars else "insig"
                    rows_html += f"""<tr>
                      <td><b>{var}</b></td>
                      <td>{model.params[var]:.5f}</td>
                      <td>{model.bse[var]:.5f}</td>
                      <td>{model.tvalues[var]:.3f}</td>
                      <td class="{sig_class}">{p:.4f} {stars}</td>
                      <td>[{ci.loc[var,0]:.4f}, {ci.loc[var,1]:.4f}]</td>
                    </tr>"""

                st.markdown(f"""
                <table class="reg-table">
                  <thead><tr>
                    <th>Variable</th><th>Coef</th><th>Std Err</th>
                    <th>t-stat</th><th>P &gt; |t|</th><th>95% CI</th>
                  </tr></thead>
                  <tbody>{rows_html}</tbody>
                </table>
                <p style="font-size:11px;color:#636366;margin-top:6px;">
                  *** p&lt;0.001 &nbsp;** p&lt;0.01 &nbsp;* p&lt;0.05
                </p>""", unsafe_allow_html=True)

                # ── Professor's Summary ──────────────────────────────────────
                st.markdown("#### Professor's Summary")
                summary_lines = []
                for var in x_vars:
                    coef      = model.params[var]
                    p         = model.pvalues[var]
                    direction = "increases" if coef > 0 else "decreases"
                    sig_note  = (f"This effect is **statistically significant** (p = {p:.4f})."
                                 if p < 0.05 else
                                 f"This effect is **not statistically significant** (p = {p:.4f}).")
                    summary_lines.append(
                        f"<p>• For every 1-unit increase in <b>{var}</b>, "
                        f"<b>{y_var}</b> {direction} by <b>{abs(coef):.4f}</b> units. {sig_note}</p>"
                    )
                r2_pct  = model.rsquared * 100
                overall = (f"<p>Overall, the model explains <b>{r2_pct:.1f}%</b> of the "
                           f"variance in <b>{y_var}</b> (R² = {model.rsquared:.4f}).</p>")
                st.markdown(
                    '<div class="prof-box"><p class="label">Plain-English Interpretation</p>'
                    + "".join(summary_lines) + overall + '</div>',
                    unsafe_allow_html=True
                )

                # ── Residual diagnostics ─────────────────────────────────────
                st.markdown("#### Residual Diagnostics")
                dw = sm.stats.stattools.durbin_watson(model.resid)
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("Durbin-Watson",  f"{dw:.3f}")
                d2.metric("AIC",            f"{model.aic:.2f}")
                d3.metric("BIC",            f"{model.bic:.2f}")
                d4.metric("Log-Likelihood", f"{model.llf:.2f}")

                # ── Scatter plot: Y vs first X (interactive Plotly) ───────────
                first_x = x_vars[0]
                st.markdown(f"#### Scatter Plot — {y_var} vs {first_x}")

                fig2 = px.scatter(
                    reg_data,
                    x=first_x,
                    y=y_var,
                    trendline="ols",
                    hover_name=selected_label if selected_label else None,
                    hover_data={first_x: True, y_var: True} if not selected_label else None,
                    custom_data=[reg_data.index] if not selected_label else None,
                    labels={first_x: first_x, y_var: y_var},
                )
                if not selected_label:
                    fig2.update_traces(
                        hovertemplate="Row: %{customdata[0]}<br>"
                                      + f"{first_x}: %{{x}}<br>"
                                      + f"{y_var}: %{{y}}<extra></extra>",
                        selector=dict(mode='markers'),
                    )
                fig2.update_traces(
                    marker=dict(color='#0A84FF', opacity=0.6, size=7),
                    selector=dict(mode='markers'),
                )
                fig2.update_traces(
                    line=dict(color='#FF453A', width=2),
                    selector=dict(type='scatter', mode='lines'),
                )
                fig2.update_layout(
                    paper_bgcolor='#0E1117',
                    plot_bgcolor='#1C1C1E',
                    font_color='#F2F2F7',
                    xaxis=dict(gridcolor='#2C2C2E', zerolinecolor='#3A3A3C'),
                    yaxis=dict(gridcolor='#2C2C2E', zerolinecolor='#3A3A3C'),
                    margin=dict(l=40, r=20, t=30, b=40),
                )
                st.plotly_chart(fig2, use_container_width=True)

                # ── Export button ────────────────────────────────────────────
                st.markdown("#### Download Results")

                # Build CSV export from coefficient table
                export_rows = []
                for var in model.params.index:
                    p     = model.pvalues[var]
                    stars = ("***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "")
                    export_rows.append({
                        "Variable":   var,
                        "Coef":       round(model.params[var], 6),
                        "Std_Err":    round(model.bse[var], 6),
                        "t_stat":     round(model.tvalues[var], 4),
                        "P_value":    round(p, 6),
                        "Sig":        stars,
                        "CI_lower":   round(ci.loc[var, 0], 6),
                        "CI_upper":   round(ci.loc[var, 1], 6),
                    })
                export_df = pd.DataFrame(export_rows)

                # Prepend model-level stats as a header block in a text file
                txt_summary = (
                    f"OLS Regression Results\n"
                    f"{'='*50}\n"
                    f"Dependent Variable : {y_var}\n"
                    f"Independent Vars   : {', '.join(x_vars)}\n"
                    f"Observations       : {int(model.nobs)}\n"
                    f"R²                 : {model.rsquared:.6f}\n"
                    f"Adj. R²            : {model.rsquared_adj:.6f}\n"
                    f"F-statistic        : {model.fvalue:.4f}  (p = {model.f_pvalue:.6f})\n"
                    f"AIC                : {model.aic:.4f}\n"
                    f"BIC                : {model.bic:.4f}\n"
                    f"Log-Likelihood     : {model.llf:.4f}\n"
                    f"Durbin-Watson      : {dw:.4f}\n"
                    f"{'='*50}\n\n"
                    + export_df.to_string(index=False)
                    + "\n\n*** p<0.001  ** p<0.01  * p<0.05\n"
                )

                col_csv, col_txt = st.columns(2)

                with col_csv:
                    st.download_button(
                        label="⬇  Download as CSV",
                        data=export_df.to_csv(index=False).encode('utf-8'),
                        file_name=f"regression_{y_var}.csv",
                        mime="text/csv",
                    )

                with col_txt:
                    st.download_button(
                        label="⬇  Download as TXT",
                        data=txt_summary.encode('utf-8'),
                        file_name=f"regression_{y_var}.txt",
                        mime="text/plain",
                    )

            except Exception as e:
                st.error(f"Regression failed: {e}")
                st.info("Common causes: multicollinear variables, non-numeric data, "
                        "or too few observations after dropping nulls.")
