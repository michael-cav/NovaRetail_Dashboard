"""NovaRetail | Customer Intelligence Dashboard — Streamlit Community Cloud ready."""
from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="NovaRetail | Customer Intelligence", page_icon="🛍️", layout="wide", initial_sidebar_state="expanded")

INK = "#17324D"
TEAL = "#087E8B"
CORAL = "#E66A53"
GOLD = "#E9AD4C"
COLORS = [TEAL, CORAL, "#6587B0", GOLD, "#9071B1", "#5A9D72", "#7B8794"]
SEGMENT_COLORS = {"Promising": "#087E8B", "Growth": "#5A9D72", "Stable": "#6587B0", "Decline": "#E66A53", "Unknown": "#9CA3AF", "Ambiguous": "#E9AD4C"}

st.markdown("""<style>
[data-testid="stAppViewContainer"] {background:#F6F8FB;}
[data-testid="stSidebar"] {background:#F0F5F8;}
.block-container {padding-top:2.0rem; padding-bottom:3rem; max-width:1500px;}
h1,h2,h3 {color:#17324D; letter-spacing:-.03em;}
[data-testid="stMetric"] {background:white; border:1px solid #E3EAF0; border-radius:16px; padding:16px 18px; box-shadow:0 3px 14px #17324D09;}
[data-testid="stMetricLabel"] {color:#526579;}
[data-testid="stMetricValue"] {color:#17324D;}
[data-testid="stPlotlyChart"] {background:white; border-radius:14px; border:1px solid #E3EAF0; padding:5px;}
.hero {background:linear-gradient(115deg,#17324D,#126778); color:white; padding:26px 30px; border-radius:18px; margin-bottom:22px;}
.hero h1 {color:white; margin:0; font-size:2rem;}
.hero p {color:#E6F5F5; margin:.4rem 0 0;}
.small-note {color:#66788A; font-size:.88rem;}
</style>""", unsafe_allow_html=True)

DATA_PATH = Path(__file__).parent / "NR_dataset.xlsx"
CATEGORIES = {"grocery": "Groceries", "grocery items": "Groceries", "groceries": "Groceries", "home appliance": "Home Appliances", "home appliances": "Home Appliances"}

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="data", engine="openpyxl")
    expected = ["label", "CustomerID", "TransactionID", "TransactionDate", "ProductCategory", "PurchaseAmount", "CustomerAgeGroup", "CustomerGender", "CustomerRegion", "CustomerSatisfaction", "RetailChannel"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError("Required dataset columns missing: " + ", ".join(missing))
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")
    df["PurchaseAmount"] = pd.to_numeric(df["PurchaseAmount"], errors="coerce")
    df["CustomerSatisfaction"] = pd.to_numeric(df["CustomerSatisfaction"], errors="coerce")
    for col in ["label", "CustomerRegion", "RetailChannel", "ProductCategory", "CustomerAgeGroup", "CustomerGender"]:
        df[col] = df[col].astype("string").str.strip().replace("", pd.NA)
    df["label"] = df["label"].fillna("Unknown")
    df["CategoryClean"] = df["ProductCategory"].fillna("Unknown").map(lambda v: CATEGORIES.get(str(v).lower(), str(v)))
    df["CustomerKey"] = df["CustomerID"].astype("string").fillna("Unknown")
    return df


def customer_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One customer per row. Segment = latest date's label if unambiguous; else Ambiguous."""
    if df.empty:
        return pd.DataFrame(columns=["CustomerID", "Revenue", "Purchase Rows", "Last Purchase", "Average Satisfaction", "Latest Segment", "Days Since Latest", "Possible Retention Flag"])
    base = df.groupby("CustomerKey", dropna=False).agg(
        Revenue=("PurchaseAmount", "sum"),
        **{"Purchase Rows": ("CustomerKey", "size"), "Last Purchase": ("TransactionDate", "max"), "Average Satisfaction": ("CustomerSatisfaction", "mean")},
    ).reset_index().rename(columns={"CustomerKey": "CustomerID"})
    latest = df.merge(df.groupby("CustomerKey")["TransactionDate"].max().rename("_last"), left_on="CustomerKey", right_index=True, how="left")
    latest = latest[latest["TransactionDate"].eq(latest["_last"]) | (latest["TransactionDate"].isna() & latest["_last"].isna())]
    segs = latest.groupby("CustomerKey")["label"].agg(lambda s: s.iloc[0] if s.nunique(dropna=False) == 1 else "Ambiguous")
    base["Latest Segment"] = base["CustomerID"].map(segs).fillna("Unknown")
    max_day = df["TransactionDate"].max()
    base["Days Since Latest"] = (max_day - base["Last Purchase"]).dt.days if pd.notna(max_day) else np.nan
    base["Possible Retention Flag"] = np.where((base["Latest Segment"] == "Decline") | (base["Average Satisfaction"] <= 2), "Review", "—")
    return base.sort_values("Revenue", ascending=False)


def style_chart(fig, height=350):
    fig.update_layout(height=height, margin=dict(l=16, r=16, t=45, b=18), paper_bgcolor="white", plot_bgcolor="white", font=dict(color=INK), title_font_size=16, legend_title_text="", colorway=COLORS)
    fig.update_xaxes(showgrid=False, linecolor="#D7E2EA")
    fig.update_yaxes(gridcolor="#ECF0F4", zeroline=False)
    return fig


def bar_chart(df, group, title, horizontal=False):
    amounts = df.groupby(group, dropna=False, as_index=False)["PurchaseAmount"].sum().sort_values("PurchaseAmount", ascending=False)
    if horizontal:
        amounts = amounts.sort_values("PurchaseAmount", ascending=True)
        fig = px.bar(amounts, x="PurchaseAmount", y=group, orientation="h", title=title, color_discrete_sequence=[TEAL], labels={"PurchaseAmount": "Revenue ($)", group: ""})
        fig.update_xaxes(tickprefix="$")
    else:
        fig = px.bar(amounts, x=group, y="PurchaseAmount", title=title, color_discrete_sequence=[TEAL], labels={"PurchaseAmount": "Revenue ($)", group: ""})
        fig.update_yaxes(tickprefix="$")
    fig.update_traces(hovertemplate="%{x}<br>%{y}<extra></extra>" if not horizontal else "%{y}<br>$%{x:,.2f}<extra></extra>")
    return style_chart(fig)


def get_secret(name: str) -> str:
    env = os.environ.get(name, "")
    if env:
        return env
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""


def analytics_snapshot(df: pd.DataFrame, customer: pd.DataFrame) -> dict:
    def revenue_for(col):
        return {str(k): round(float(v), 2) for k, v in df.groupby(col, dropna=False)["PurchaseAmount"].sum().items()}
    return {
        "scope": "Active sidebar filters only; values are drawn from the uploaded dataset.",
        "total_revenue": round(float(df["PurchaseAmount"].sum()), 2),
        "recorded_rows": int(len(df)),
        "unique_customers": int(df["CustomerKey"].nunique()),
        "average_recorded_row_value": round(float(df["PurchaseAmount"].mean()), 2) if df["PurchaseAmount"].notna().any() else None,
        "revenue_by_record_segment": revenue_for("label"),
        "revenue_by_region": revenue_for("CustomerRegion"),
        "revenue_by_product_category": revenue_for("CategoryClean"),
        "revenue_by_channel": revenue_for("RetailChannel"),
        "segment_row_count": {str(k): int(v) for k, v in df["label"].value_counts().items()},
        "customer_latest_segments": {str(k): int(v) for k, v in customer["Latest Segment"].value_counts().items()},
        "average_satisfaction_by_record_segment": {str(k): round(float(v), 2) for k, v in df.groupby("label")["CustomerSatisfaction"].mean().dropna().items()},
        "top_customers": [{"customer_id": str(r["CustomerID"]), "revenue": round(float(r["Revenue"]), 2), "latest_segment": str(r["Latest Segment"])} for _, r in customer.head(10).iterrows()],
        "date_min": str(df["TransactionDate"].min().date()) if df["TransactionDate"].notna().any() else None,
        "date_max": str(df["TransactionDate"].max().date()) if df["TransactionDate"].notna().any() else None,
    }


def local_answer(question: str, snapshot: dict) -> str:
    """Transparent no-key backup, not represented as a generative AI model."""
    q = question.lower()
    data_choices = [("region", "revenue_by_region"), ("product", "revenue_by_product_category"), ("categor", "revenue_by_product_category"), ("channel", "revenue_by_channel"), ("segment", "revenue_by_record_segment")]
    for keyword, key in data_choices:
        if keyword in q:
            d = snapshot[key]
            ranking = sorted(d.items(), key=lambda x: x[1], reverse=True)
            if not ranking:
                return "There are no matching revenue records for this selection."
            return "Revenue by {} (active filters):\n\n".format(keyword) + "\n".join(f"- **{k}**: ${v:,.2f}" for k, v in ranking) + "\n\nHighest recorded revenue: **{}**. This is descriptive, not proof of future growth.".format(ranking[0][0])
    if "risk" in q or "declin" in q or "retention" in q:
        return "The **Decline** label can flag records for retention review. Counts by record segment: {}. Satisfaction and recent activity may add context, but this short historical dataset alone cannot prove future churn.".format(snapshot["segment_row_count"])
    if "customer" in q or "top" in q:
        top = snapshot["top_customers"][:5]
        return "Top customers by recorded revenue for the active filters:\n\n" + "\n".join(f"- Customer {c['customer_id']}: ${c['revenue']:,.2f}" for c in top)
    if "revenue" in q or "sales" in q or "total" in q:
        return f"The selected records total **${snapshot['total_revenue']:,.2f}** across **{snapshot['recorded_rows']} purchase rows** and **{snapshot['unique_customers']} unique customers**."
    return "Without an AI API key, I can answer basic questions about revenue, regions, segments, products, channels, top customers, and decline risk. For open-ended questions, configure an OpenAI API key in Streamlit secrets."


def ai_answer(question, snapshot, api_key, model):
    from openai import OpenAI
    client = OpenAI(api_key=api_key, timeout=25.0)
    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        max_tokens=450,
        messages=[
            {"role": "system", "content": "You are NovaRetail's careful business analytics assistant. Answer only with facts present in the attached aggregate snapshot of currently filtered data. Show numbers and units; when information isn't available say so. Never invent trends, causation, purchases, segment transitions, or customer attributes. Distinguish record-level segment grouping from latest customer-level grouping. Dataset dates are historical; present conclusions as exploratory. Keep concise. Do not infer details outside the snapshot."},
            {"role": "user", "content": "Filtered dataset aggregate snapshot:\n" + json.dumps(snapshot, default=str) + "\n\nQuestion: " + question},
        ],
    )
    return response.choices[0].message.content

if not DATA_PATH.exists():
    st.error("The data file NR_dataset.xlsx is missing. Place it next to app.py in the GitHub repository.")
    st.stop()
try:
    raw = load_data(str(DATA_PATH))
except Exception as exc:
    st.error(f"Unable to read dataset: {exc}")
    st.stop()

st.markdown('<div class="hero"><h1>NovaRetail · Customer Intelligence</h1><p>Revenue performance, customer segments, retention signals, and growth opportunities</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.title("Explore the data")
    st.caption("All selections update the dashboard and chatbot together.")
    if st.button("↺ Reset filters", use_container_width=True):
        for key in ("regions", "segments", "channels", "categories", "dates"):
            st.session_state.pop(key, None)
        st.rerun()
    def multi(col, label, key):
        return st.multiselect(label, sorted(raw[col].dropna().astype(str).unique()), key=key, placeholder="All")
    regions = multi("CustomerRegion", "Region", "regions")
    segments = multi("label", "Record segment", "segments")
    channels = multi("RetailChannel", "Retail channel", "channels")
    categories = multi("CategoryClean", "Product category", "categories")
    date_values = raw["TransactionDate"].dropna()
    date_range = None
    if not date_values.empty:
        low, high = date_values.min().date(), date_values.max().date()
        date_range = st.date_input("Purchase date range", value=(low, high), min_value=low, max_value=high, key="dates")
    st.divider()
    st.caption("Source: NR_dataset.xlsx · Historical purchase records. No synthetic transactions are added.")

filtered = raw.copy()
for col, values in [("CustomerRegion", regions), ("label", segments), ("RetailChannel", channels), ("CategoryClean", categories)]:
    if values:
        filtered = filtered[filtered[col].isin(values)]
if date_range is not None and isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    first, last = date_range
    filtered = filtered[filtered["TransactionDate"].dt.date.between(first, last)]

if filtered.empty:
    st.warning("No records match these filters. Clear some selections or use Reset filters.")
    st.stop()

customers = customer_summary(filtered)
snapshot = analytics_snapshot(filtered, customers)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Recorded revenue", f"${filtered['PurchaseAmount'].sum():,.2f}")
k2.metric("Unique customers", f"{filtered['CustomerKey'].nunique():,}")
k3.metric("Recorded purchase rows", f"{len(filtered):,}")
k4.metric("Average purchase row", f"${filtered['PurchaseAmount'].mean():,.2f}")

st.caption("All metrics use the selected purchase rows; repeated transaction IDs are not automatically removed. Counts describe the filtered historical extract, not NovaRetail's nationwide totals.")

overview, growth, retention, detail, chat, notes = st.tabs(["📊 Overview", "📈 Revenue & growth", "🧭 Retention", "👥 Customers", "💬 AI assistant", "ℹ️ Data notes"])

with overview:
    left, right = st.columns(2)
    with left:
        st.plotly_chart(bar_chart(filtered, "label", "Revenue by record segment"), use_container_width=True)
    with right:
        by_month = filtered.dropna(subset=["TransactionDate"]).copy()
        by_month["Day"] = by_month["TransactionDate"].dt.date
        by_month = by_month.groupby("Day", as_index=False)["PurchaseAmount"].sum().sort_values("Day")
        fig = px.line(by_month, x="Day", y="PurchaseAmount", markers=True, title="Recorded revenue over time", labels={"PurchaseAmount": "Revenue ($)", "Day": "Purchase date"})
        fig.update_traces(line_color=TEAL, marker_color=CORAL)
        fig.update_yaxes(tickprefix="$")
        st.plotly_chart(style_chart(fig), use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(bar_chart(filtered, "CustomerRegion", "Revenue by region"), use_container_width=True)
    with right:
        st.plotly_chart(bar_chart(filtered, "RetailChannel", "Revenue by channel"), use_container_width=True)

with growth:
    st.subheader("Where is revenue concentrated?")
    a, b = st.columns(2)
    with a:
        st.plotly_chart(bar_chart(filtered, "CategoryClean", "Revenue by product category", horizontal=True), use_container_width=True)
    with b:
        top = customers.head(10).sort_values("Revenue")
        fig = px.bar(top, x="Revenue", y="CustomerID", orientation="h", title="Top 10 customers by recorded revenue", color_discrete_sequence=[CORAL], labels={"Revenue": "Revenue ($)", "CustomerID": "Customer ID"})
        fig.update_xaxes(tickprefix="$")
        st.plotly_chart(style_chart(fig), use_container_width=True)
    st.info("**Business use:** Compare category, region, and channel strengths, then test targeted campaigns. Revenue concentration alone does not establish profitability or a causal growth opportunity.")

with retention:
    st.subheader("Monitor possible retention signals")
    st.caption("The provided segments are existing classifications. This app does not independently predict churn.")
    a, b, c = st.columns(3)
    a.metric("Decline-labeled rows", int((filtered["label"] == "Decline").sum()))
    b.metric("Customers to review", int((customers["Possible Retention Flag"] == "Review").sum()))
    c.metric("Average satisfaction", f"{filtered['CustomerSatisfaction'].mean():.2f} / 5" if filtered["CustomerSatisfaction"].notna().any() else "N/A")
    a, b = st.columns(2)
    with a:
        counts = filtered["label"].value_counts().rename_axis("Segment").reset_index(name="Rows")
        fig = px.bar(counts, x="Segment", y="Rows", color="Segment", color_discrete_map=SEGMENT_COLORS, title="Purchase rows by labeled segment")
        st.plotly_chart(style_chart(fig), use_container_width=True)
    with b:
        satisfaction = filtered.groupby("label", as_index=False)["CustomerSatisfaction"].mean().dropna()
        fig = px.bar(satisfaction, x="label", y="CustomerSatisfaction", title="Average satisfaction by record segment", labels={"label": "Segment", "CustomerSatisfaction": "Satisfaction (1–5)"}, color_discrete_sequence=[TEAL])
        fig.update_yaxes(range=[0, 5])
        st.plotly_chart(style_chart(fig), use_container_width=True)
    st.markdown("#### Customers flagged for review")
    st.dataframe(customers[customers["Possible Retention Flag"] == "Review"], hide_index=True, use_container_width=True)
    st.caption("Review flag = latest unambiguous segment is Decline OR selected rows' average satisfaction ≤ 2. Days since latest purchase are relative to the latest date in the **filtered extract**, not today. Filtering can change each customer's apparent latest segment.")

with detail:
    st.subheader("Customer-level drilldown")
    query = st.text_input("Search customer ID", placeholder="Example: 12345")
    displayed = customers[customers["CustomerID"].str.contains(query, case=False, regex=False, na=False)] if query else customers
    st.dataframe(displayed, hide_index=True, use_container_width=True, column_config={"Revenue": st.column_config.NumberColumn(format="$%.2f"), "Average Satisfaction": st.column_config.NumberColumn(format="%.2f")})
    st.download_button("Download filtered customer summary CSV", displayed.to_csv(index=False).encode("utf-8"), "novaretail_customer_summary.csv", "text/csv")
    st.caption("A customer's 'Latest Segment' is based on their latest selected purchase date. If multiple different labels exist on that date, it is 'Ambiguous'. Segment assignments are not necessarily permanent.")

with chat:
    st.subheader("Ask about your selected data")
    api_key = get_secret("OPENAI_API_KEY")
    model = get_secret("OPENAI_MODEL") or "gpt-4o-mini"
    if api_key:
        st.success("AI mode is available. Responses are grounded in aggregate values from your filtered data.")
    else:
        st.info("**Basic analytics mode:** Answers to common questions are available without a key. To activate the generative AI chatbot, add OPENAI_API_KEY to Streamlit secrets. The app remains functional without it.")
    st.caption("Try: 'Which region has the highest revenue?', 'Show sales by product category', 'Which customers spent the most?', or 'Which segment may need retention attention?'")
    filter_fingerprint = json.dumps({"regions": regions, "segments": segments, "channels": channels, "categories": categories, "dates": str(date_range)})
    if st.session_state.get("chat_scope") != filter_fingerprint:
        st.session_state["chat_scope"] = filter_fingerprint
        st.session_state["messages"] = []
    for m in st.session_state.get("messages", []):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    q = st.chat_input("Ask a question about these results...")
    if q:
        st.session_state["messages"].append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            try:
                answer = ai_answer(q, snapshot, api_key, model) if api_key else local_answer(q, snapshot)
            except Exception as exc:
                answer = f"The AI provider could not complete the request ({type(exc).__name__}). Here is a basic data-based answer:\n\n" + local_answer(q, snapshot)
            st.markdown(answer)
        st.session_state["messages"].append({"role": "assistant", "content": answer})

with notes:
    st.subheader("Data quality and definitions")
    st.markdown("""- **Source:** Uploaded `NR_dataset.xlsx`, sheet `data`. The source file is never modified.
- **Unit of analysis:** One Excel row is a recorded purchase line, not necessarily a unique transaction.
- **Transaction IDs:** IDs can repeat. These rows are retained because identical IDs do not independently establish duplicate sales.
- **Segment:** Original row-level label; blanks are presented as `Unknown`.
- **Category normalization:** `Grocery`, `Grocery Items`, and `Groceries` are combined into `Groceries`; variants of `Home Appliance(s)` are combined.
- **Unique customers:** Distinct `CustomerID` among the rows selected by the filters.
- **Latest customer segment:** Latest date within the selected records, with `Ambiguous` when several labels disagree on that same date.
- **Retention:** A descriptive prioritization, **not** a trained churn model. Satisfaction and segment flags are only clues.
- **Revenue:** Sum of purchase amounts; costs and profit margin are not provided.
- **Dates:** Historical extract. Recency is measured relative to the filtered dataset's latest purchase date.
- **Chat privacy:** With an AI key configured, selected aggregate metrics and the user's question are sent to the configured model provider; row-level customer identifiers are not included beyond top customer IDs. Check institutional data-sharing requirements before enabling this feature.""")
    with st.expander("Data quality diagnostics"):
        st.write("Raw records:", len(raw))
        st.write("Missing segment labels:", int((raw["label"] == "Unknown").sum()))
        st.write("Distinct transaction IDs:", int(raw["TransactionID"].nunique()))
        st.write("Fully identical records:", int(raw.duplicated().sum()))
        st.write("Customer IDs with more than one recorded segment:", int((raw.groupby("CustomerKey")["label"].nunique() > 1).sum()))
        st.dataframe(pd.DataFrame({"Raw category": raw["ProductCategory"], "Displayed category": raw["CategoryClean"]}).drop_duplicates().sort_values("Raw category"), hide_index=True, use_container_width=True)
