import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns

from ml_models import (
    DELAY_TRAIN_MAX_YEAR,
    predict_category,
    predict_delay,
    train_category_models,
    train_delay_models,
)

# Page setup
st.set_page_config(page_title="ArmsLens", page_icon="🔍", layout="wide")
st.title("🔍 ArmsLens")
st.markdown("#### AI-Powered Analysis and Prediction of Global Arms Trade (2000–2023)")

# =========================
# 📥 Load & Clean Data
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("data/global_arms_transfer_2000_2023.csv")
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

    # "orderYrEst" is a True/False flag: was the order year ESTIMATED (not exactly known)?
    if "orderyrest" in df.columns:
        df["order_year_estimated"] = df["orderyrest"].astype(str).str.lower().eq("true")
        df = df.drop(columns=["orderyrest"])

    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
        df["order_year"] = df["order_date"].dt.year
        df["order_date"] = df["order_date"].dt.date

    if "delivery_date" in df.columns:
        df["delivery_date"] = pd.to_datetime(df["delivery_date"], errors="coerce")
        df["delivery_date"] = df["delivery_date"].dt.date

    df["order_year"] = pd.to_numeric(df["order_year"], errors="coerce").astype("Int64")

    return df

df = load_data()

# =========================
# 📄 Raw Data Preview + Download
# =========================
with st.expander("📄 Click to preview raw CSV data (uncleaned)"):
    raw_df = pd.read_csv("data/global_arms_transfer_2000_2023.csv")
    st.write(raw_df.head())

    csv_data = raw_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Raw CSV",
        data=csv_data,
        file_name="global_arms_transfer_raw.csv",
        mime="text/csv"
    )

# =========================
# ✅ Cleaned Data Preview
# =========================
st.subheader("✅ Cleaned Data Preview")
st.dataframe(df.head(), use_container_width=True)
st.markdown(f"**📊 Full Dataset Dimensions:** `{df.shape[0]} rows × {df.shape[1]} columns`")

# =========================
# 🧰 Sidebar Filters
# =========================
st.sidebar.header("🧰 Filter Data")

years = sorted(df["order_year"].dropna().unique())
selected_years = st.sidebar.multiselect("Select Year(s)", years)

categories = sorted(df["category"].dropna().unique()) if "category" in df.columns else []
selected_categories = st.sidebar.multiselect("Select Category", categories)

sources = sorted(df["source"].dropna().unique()) if "source" in df.columns else []
selected_sources = st.sidebar.multiselect("Select Source Country", sources)

targets = sorted(df["target"].dropna().unique()) if "target" in df.columns else []
selected_targets = st.sidebar.multiselect("Select Target Country", targets)

# =========================
# 🎯 Apply Filters
# =========================
if selected_years or selected_categories or selected_sources or selected_targets:
    filtered_df = df.copy()
    if selected_years:
        filtered_df = filtered_df[filtered_df["order_year"].isin(selected_years)]
    if selected_categories:
        filtered_df = filtered_df[filtered_df["category"].isin(selected_categories)]
    if selected_sources:
        filtered_df = filtered_df[filtered_df["source"].isin(selected_sources)]
    if selected_targets:
        filtered_df = filtered_df[filtered_df["target"].isin(selected_targets)]

    st.subheader(f"🎯 Filtered Results — ({len(filtered_df)}) rows matched")
    st.dataframe(filtered_df.head(), use_container_width=True)
    st.markdown(f"**📐 Filtered Dataset Dimensions:** `{filtered_df.shape[0]} rows × {filtered_df.shape[1]} columns`")

    # 📥 Download filtered data
    csv_filtered = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Filtered Data as CSV",
        data=csv_filtered,
        file_name="filtered_arms_trade_data.csv",
        mime="text/csv"
    )
else:
    filtered_df = df
    st.info("ℹ️ No filters selected yet — please choose at least one filter from the sidebar.")

# =========================
# ❓ Arms Trade Questions
# =========================
st.header("❓ Arms Trade Q&A — Data-Driven Insights")

# All Q&A charts below use `qdf`: the filtered data when filters are set, otherwise the full dataset.
filters_active = bool(selected_years or selected_categories or selected_sources or selected_targets)
if filters_active and filtered_df.empty:
    st.warning("⚠️ No rows match the selected filters — showing charts for the full dataset instead.")
    qdf = df
else:
    qdf = filtered_df
if filters_active and not filtered_df.empty:
    st.caption(f"🎯 Charts below reflect your sidebar filters ({len(qdf):,} of {len(df):,} deals).")
else:
    st.caption(f"Charts below use the full dataset ({len(df):,} deals). Use the sidebar to filter them.")

MIN_DEALS = 10  # averages are only shown for countries/models with enough deals to be meaningful


def show_fig(fig):
    """Render a Matplotlib figure, then close it to free memory."""
    st.pyplot(fig)
    plt.close(fig)


def has_data(data, min_rows=1):
    """Show a friendly message instead of an error when filters leave too little data."""
    if len(data) < min_rows:
        st.info("Not enough data for the current filters to answer this question.")
        return False
    return True
# =========================
# Q1: Top 10 Exporters Chart (Altair)
# =========================
with st.expander("🚢 Q1. What are the top 10 exporting countries (2000–2023)?"):
    top_sources = (
        qdf["source"]
        .value_counts()
        .head(10)
        .reset_index()
    )
    top_sources.columns = ["source_country", "trade_count"]
    
    st.dataframe(top_sources)
    
    chart1 = alt.Chart(top_sources).mark_bar().encode(
        x=alt.X("trade_count:Q", title="Number of Trades"),
        y=alt.Y("source_country:N", sort="-x", title="Country"),
        tooltip=["source_country", "trade_count"],
        color=alt.value("steelblue")
    ).properties(
        height=400,
        width=700
    )
    st.altair_chart(chart1, use_container_width=True)

# =========================
# 📥 Q2: Top 10 Importers 
# =========================
with st.expander("📥 Q2. What are the top 10 importing countries (2000–2023)?"):
    top_targets = (
        qdf["target"]
        .value_counts()
        .head(10)
        .reset_index()
    )
    top_targets.columns = ["target_country", "import_count"]
    
    st.dataframe(top_targets)

    chart2 = alt.Chart(top_targets).mark_bar().encode(
        x=alt.X("import_count:Q", title="Number of Imports"),
        y=alt.Y("target_country:N", sort="-x", title="Country"),
        tooltip=["target_country", "import_count"],
        color=alt.value("steelblue")
    ).properties(
        height=400,
        width=700
    )
    st.altair_chart(chart2, use_container_width=True)


# =========================
# 💣 Q3: What proportion of arms traded belongs to each weapon category?
# =========================
with st.expander("💣 Q3: Which weapon categories were ordered the most?"):
    st.markdown("This lollipop chart shows the total quantity of arms traded by weapon category.")
    
    cat_df = qdf[qdf["category"].notna()].groupby("category")["quantity"].sum().sort_values(ascending=True).reset_index()
    
    # Show data as a table
    st.dataframe(cat_df.rename(columns={"category": "Weapon Category", "quantity": "Total Quantity"}), use_container_width=True)

    # Plot lollipop chart
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hlines(y=cat_df["category"], xmin=0, xmax=cat_df["quantity"], color="skyblue", linewidth=2)
    ax.plot(cat_df["quantity"], cat_df["category"], "o", markersize=6, color="navy")
    ax.set_xlabel("Total Quantity")
    ax.set_ylabel("Category")
    ax.set_title("Weapon Categories by Total Quantity (Lollipop View)")
    show_fig(fig)
# =========================
# 📈 Q4: Annual Trend of Weapon Orders
# =========================
with st.expander("📈 Q4: What is the annual trend of weapon orders over time?"):
    st.markdown("This line plot shows how total weapon quantities have changed year by year.")

    yearly = qdf.groupby("order_year")["quantity"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.lineplot(data=yearly, x="order_year", y="quantity", marker="o", linewidth=2.5, color="steelblue", ax=ax)
    
    # Annotate values on selected peaks
    for i in range(0, len(yearly), max(len(yearly)//10, 1)):  # annotate every few years
        row = yearly.iloc[i]
        ax.text(row["order_year"], row["quantity"], f'{int(row["quantity"]):,}', fontsize=7, ha='center', va='bottom')

    ax.set_title("Yearly Trend of Arms Orders (2000–2023)", fontsize=12)
    ax.set_xlabel("Order Year")
    ax.set_ylabel("Total Quantity")
    ax.tick_params(axis='x', rotation=45)
    sns.despine()
    show_fig(fig)
# =========================
# ("🌐 Q5: Heatmap of arms transfers between top 5 countries"):

with st.expander("🌐 Q5: Arms transfers between top 5 countries"):
    st.markdown("This heatmap shows total arms transferred (by quantity) between the top 5 source and target countries.")

    # Select top 5 source and target countries
    top_sources = qdf["source"].value_counts().head(5).index
    top_targets = qdf["target"].value_counts().head(5).index

    # Filter and pivot
    heat_df = qdf[qdf["source"].isin(top_sources) & qdf["target"].isin(top_targets)]
    pivot_df = heat_df.groupby(["source", "target"])["quantity"].sum().reset_index()
    heatmap_data = pivot_df.pivot(index="source", columns="target", values="quantity").fillna(0)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(5.5, 4))
    sns.heatmap(heatmap_data, cmap="YlGnBu", annot=True, fmt=".0f", linewidths=0.3, cbar_kws={"shrink": 0.7})
    ax.set_title("Top 5 Source vs Target Arms Transfers", fontsize=11)
    show_fig(fig)

# =========================
# 🚚 Q6: Years with Highest Transfers 

# =========================
with st.expander("🚚 Q6: Which years saw the highest number of arms transfers?"):
    st.markdown("This chart shows the number of transfer records per year (not quantity).")

    year_counts = qdf["order_year"].value_counts().sort_index().reset_index()
    year_counts.columns = ["order_year", "transfer_count"]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(year_counts["order_year"], year_counts["transfer_count"], marker='o', linestyle='-', color='orange')
    ax.set_title("Number of Arms Transfers by Year", fontsize=14)
    ax.set_xlabel("Order Year")
    ax.set_ylabel("Number of Transfers")
    ax.grid(True)
    plt.xticks(rotation=45)
    show_fig(fig)

# =========================
# 🧮 Q7: Avg Quantity per Country 
# =========================
with st.expander("🧮 Q7: What is the average quantity of arms traded per transaction for each country?"):
    st.markdown(f"Only exporters with at least **{MIN_DEALS} deals** are included, so a country with 1–2 huge deals does not distort the ranking.")

    # Get top 5 countries by average quantity (with enough deals)
    avg_qty_df = (
        qdf.groupby("source")["quantity"]
        .agg(["mean", "count"])
        .query("count >= @MIN_DEALS")
        .sort_values("mean", ascending=False)
        .head(5)
        .reset_index()
    )
    avg_qty_df.columns = ["Source Country", "Average Quantity", "Number of Deals"]

    if has_data(avg_qty_df):
        st.dataframe(avg_qty_df.style.format({"Average Quantity": "{:.2f}"}), use_container_width=True)

        # Filter data for those top 5 countries
        top5 = avg_qty_df["Source Country"].tolist()
        plot_df = qdf[qdf["source"].isin(top5) & (qdf["quantity"] > 0)]

        # Scatterplot (log scale: quantities range from 1 to thousands)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=plot_df, x="quantity", y="source", hue="source", s=60, alpha=0.6, ax=ax, legend=False)
        ax.set_xscale("log")
        ax.set_title("Quantity per Transaction for Top 5 Exporters (by Average Quantity)")
        ax.set_xlabel("Quantity per deal (log scale)")
        ax.set_ylabel("Source Country")
        ax.grid(True, linestyle="--", alpha=0.3)
        show_fig(fig)

# =========================
# 🟢 Q8: Which countries had the highest average arms quantity per trade?
# =========================
with st.expander("🟢 Q8: Which countries had the highest average arms quantity per trade?"):

    st.markdown("""
    This bubble chart shows:
    - **X-axis:** Number of trades  
    - **Y-axis:** Country  
    - **Bubble size:** Average quantity per trade
    """)


    st.caption(f"Only exporters with at least {MIN_DEALS} deals are included.")

    # Compute average quantity and trade count
    country_stats = (
        qdf.groupby("source")
        .agg(trade_count=("quantity", "count"), avg_quantity=("quantity", "mean"))
        .query("trade_count >= @MIN_DEALS")
        .sort_values(by="avg_quantity", ascending=False)
        .head(10)
        .reset_index()
    )

    if has_data(country_stats):
        st.dataframe(country_stats.style.format({"avg_quantity": "{:.1f}", "trade_count": "{:,.0f}"}))

        # Plot: bubble area scaled so the largest bubble is a fixed size
        fig, ax = plt.subplots(figsize=(10, 6))
        sizes = country_stats["avg_quantity"] / country_stats["avg_quantity"].max() * 2000
        ax.scatter(
            country_stats["trade_count"],
            country_stats["source"],
            s=sizes,
            alpha=0.6,
            color="steelblue",
            edgecolors="black"
        )

        for _, row in country_stats.iterrows():
            ax.text(
                row["trade_count"], row["source"],
                f'{row["avg_quantity"]:.0f}', va='center', ha='center', fontsize=8, color='black'
            )

        ax.set_xscale("log")
        ax.set_xlabel("Number of Trades (log scale)")
        ax.set_ylabel("Country")
        ax.set_title("Top 10 Exporters by Avg Quantity per Trade (bubble size = avg quantity)", fontsize=12)
        ax.margins(x=0.15, y=0.08)
        show_fig(fig)

# =========================
# 🔫 Q9: What are the top traded designations (weapons/models)?
# =========================
with st.expander("🔫 Q9: What are the most frequently traded weapon designations?"):

    st.markdown("""
This chart shows the **top 10 weapon designations** by the number of trades.  
It uses a donut-style chart to visualize the share of each designation.
""")

    # Prepare data
    top_designations = (
        qdf["designation"]
        .value_counts()
        .drop(labels=["-", "unknown", "n/a", "other"], errors="ignore")
        .head(10)
        .reset_index()
    )
    top_designations.columns = ["designation", "trade_count"]

    st.dataframe(top_designations)

    # Donut Chart
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        top_designations["trade_count"],
        labels=top_designations["designation"],
        autopct="%1.1f%%",
        startangle=140,
        wedgeprops=dict(width=0.4),  # Makes it a donut
        textprops=dict(color="black"),
        colors=plt.cm.tab10.colors
    )

    ax.set_title("Top 10 Most Traded Weapon Designations", fontsize=13)
    show_fig(fig)
# =========================
with st.expander("🧩 Q10: How many unique weapon designations were traded from 2000 to 2023?"):
    
    st.markdown("""
This insight reveals how **diverse** the global arms trade is in terms of weapon models or designations.
    """)

    unique_designations = qdf["designation"].dropna().str.lower().str.strip()
    unique_designations = unique_designations[~unique_designations.isin(["-", "unknown", "n/a", "other"])]
    total_unique = unique_designations.nunique()

    st.success(f"🔢 Total unique weapon designations traded: **{total_unique:,}**")

# =========================
with st.expander("🗓️ Q11: In which year did the highest number of different countries import arms?"):

    st.markdown("""
This insight reveals the year in which the **largest number of unique countries** participated as arms importers.  
It indicates the **peak of global importing activity**.
""")

    year_target_counts = qdf.dropna(subset=["order_year", "target"]).groupby("order_year")["target"].nunique()
    if has_data(year_target_counts):
        top_year = int(year_target_counts.idxmax())
        top_count = year_target_counts.max()
        st.success(f"📅 The year **{top_year}** had the most diverse arms import activity, with **{top_count} countries** importing arms.")

# =========================
with st.expander("⏳ Q12: What is the average delivery delay (in years) by exporting country?"):

    st.markdown(f"Shows the **average delivery delay** between order and delivery per country "
                f"(only exporters with at least **{MIN_DEALS} delivered deals**).")

    delay_df = qdf.copy()
    delay_df["delivery_date"] = pd.to_datetime(delay_df["delivery_date"], errors="coerce")
    delay_df["order_date"] = pd.to_datetime(delay_df["order_date"], errors="coerce")
    delay_df["delay_years"] = (delay_df["delivery_date"] - delay_df["order_date"]).dt.days / 365

    delay_avg = (
        delay_df.groupby("source")["delay_years"]
        .agg(["mean", "count"])
        .query("count >= @MIN_DEALS")
        .sort_values("mean", ascending=False)
        .head(10)
        .reset_index()
        .rename(columns={"mean": "delay_years", "count": "delivered_deals"})
    )

    if has_data(delay_avg):
        st.dataframe(delay_avg.style.format({"delay_years": "{:.2f}"}))

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=delay_avg, y="source", x="delay_years", hue="source", palette="mako", legend=False, ax=ax)
        ax.set_title("Average Delivery Delay (in Years) by Exporting Country")
        ax.set_xlabel("Average Delay (Years)")
        ax.set_ylabel("Exporting Country")
        show_fig(fig)
# =========================
with st.expander("🔁 Q13: Which countries consistently trade specific categories?"):

    st.markdown("This heatmap shows how many times each country exported different **weapon categories**.")

    cat_matrix = qdf.pivot_table(
        index="source", columns="category", values="quantity", aggfunc="count"
    ).fillna(0)

    top_sources = qdf["source"].value_counts().head(10).index
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(cat_matrix.loc[top_sources], annot=True, fmt=".0f", cmap="YlGnBu", ax=ax)
    ax.set_title("Export Frequency of Weapon Categories by Top Countries")
    show_fig(fig)
# =========================
with st.expander("📊 Q14: Are certain categories becoming more or less popular over time?"):

    st.markdown("This **stacked area chart** shows how category popularity changed over time (by quantity).")

    cat_trend = qdf[qdf["category"].notna()].groupby(["order_year", "category"])["quantity"].sum().reset_index()

    # Show table before plotting
    st.dataframe(cat_trend.rename(columns={
        "order_year": "Year",
        "category": "Weapon Category",
        "quantity": "Total Quantity"
    }), use_container_width=True)

    pivot_area = cat_trend.pivot(index="order_year", columns="category", values="quantity").fillna(0)

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot_area.plot.area(ax=ax, cmap="tab20", linewidth=0)
    ax.set_title("Weapon Category Popularity Over Time (by Quantity)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Total Quantity")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    show_fig(fig)

# =========================
with st.expander("🤝  Q15: What are the most common exporter-importer pairs?"):

    st.markdown("This **bar chart** shows the top 10 exporter → importer country pairs by number of trades.")

    # Get top 10 pairs
    pair_df = (
        qdf.groupby(["source", "target"])
        .size()
        .reset_index(name="trade_count")
        .sort_values("trade_count", ascending=False)
        .head(10)
    )
    pair_df["pair"] = pair_df["source"] + " → " + pair_df["target"]

    if has_data(pair_df):
        chart15 = alt.Chart(pair_df).mark_bar(color="steelblue").encode(
            x=alt.X("trade_count:Q", title="Number of Trades"),
            y=alt.Y("pair:N", sort="-x", title="Exporter → Importer"),
            tooltip=["source", "target", "trade_count"],
        ).properties(height=400)
        st.altair_chart(chart15, use_container_width=True)

# =========================
with st.expander("📦 Q16: Which countries import the highest quantity per transaction?"):
    st.markdown("This reveals countries that typically import in **larger batches**, based on average quantity per deal.")

    st.caption(f"Only importers with at least {MIN_DEALS} deals are included.")

    top_avg_imports = (
        qdf[qdf["quantity"].notna()]
        .groupby("target")["quantity"]
        .agg(["mean", "count"])
        .query("count >= @MIN_DEALS")
        .sort_values("mean", ascending=False)
        .head(10)
        .reset_index()
    )
    top_avg_imports.columns = ["Importing Country", "Avg Quantity per Transaction", "Number of Deals"]

    if has_data(top_avg_imports):
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=top_avg_imports, x="Avg Quantity per Transaction", y="Importing Country",
                    hue="Importing Country", palette="crest", legend=False, ax=ax)
        ax.set_title("Top 10 Importers by Avg Quantity per Transaction")
        show_fig(fig)
# =========================
with st.expander("🛠️ Q17: What are the most commonly traded weapon categories by decade?"):
    st.markdown("""
This reveals how the **top traded weapon categories** have changed across decades:  
- 📅 **2000s** (2000–2009)  
- 📅 **2010s** (2010–2019)  
- 📅 **2020s** (2020–2023)
""")

    df_decade = qdf.copy()
    df_decade = df_decade[df_decade["order_year"].notna()]
    df_decade["decade"] = pd.cut(df_decade["order_year"],
                                  bins=[1999, 2009, 2019, 2024],
                                  labels=["2000s", "2010s", "2020s"])

    top_by_decade = (
        df_decade[df_decade["category"].notna()]
        .groupby(["decade", "category"])["quantity"]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top_by_decade, x="quantity", y="category", hue="decade", ax=ax)
    ax.set_title("Top Traded Weapon Categories by Decade")
    ax.set_xlabel("Total Quantity")
    ax.set_ylabel("Weapon Category")
    ax.legend(title="Decade", loc="lower right")
    show_fig(fig)
# =========================
with st.expander("🏗️ Q18: How many arms deals had missing delivery dates?"):
    st.markdown("Helps identify **data gaps** or possibly delayed or cancelled shipments.")
    
    missing_count = qdf["delivery_date"].isna().sum()
    total = len(qdf)
    percent = (missing_count / total) * 100

    st.success(f"❌ Missing Delivery Dates: **{missing_count:,}** out of **{total:,}** records (**{percent:.2f}%**).")
# =========================
with st.expander("🛠️ Q19: Which weapon designations had the longest average delivery delay?"):
    st.markdown("Focuses on **logistical or production delays** by designation (model).")

    delay_df = qdf.copy()
    delay_df["delivery_date"] = pd.to_datetime(delay_df["delivery_date"], errors="coerce")
    delay_df["order_date"] = pd.to_datetime(delay_df["order_date"], errors="coerce")
    delay_df["delay_years"] = (delay_df["delivery_date"] - delay_df["order_date"]).dt.days / 365

    MIN_MODEL_DEALS = 5
    st.caption(f"Names are cleaned (e.g. 'NH-90 NFH --' and 'NH-90 NFH -' count as one model) and only "
               f"models with at least {MIN_MODEL_DEALS} delivered deals are included.")

    # Clean designation names: the raw data adds trailing dashes/spaces to variants of the same model
    delay_df["designation_clean"] = delay_df["designation"].str.replace(r"[\s\-]+$", "", regex=True).str.strip()

    top_delays = (
        delay_df[delay_df["designation_clean"].notna() & (delay_df["designation_clean"] != "")]
        .groupby("designation_clean")["delay_years"]
        .agg(["mean", "count"])
        .query("count >= @MIN_MODEL_DEALS")
        .sort_values("mean", ascending=False)
        .head(10)
        .reset_index()
        .rename(columns={"designation_clean": "designation", "mean": "delay_years", "count": "delivered_deals"})
    )

    if has_data(top_delays):
        # Show table
        st.dataframe(top_delays.style.format({"delay_years": "{:.2f}"}))

        # Bar chart
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(
            data=top_delays,
            y="designation",
            x="delay_years",
            hue="designation",
            palette="crest",
            legend=False,
            ax=ax
        )
        ax.set_title("Top 10 Designations by Average Delivery Delay (in Years)")
        ax.set_xlabel("Average Delay (Years)")
        ax.set_ylabel("Weapon Designation")
        show_fig(fig)

# =========================
with st.expander("🎯 Q20: What is the most frequently traded weapon category per country?"):
    st.markdown("This shows each country's **top exported category**, reflecting their specialization.")

    cat_mode = (
        qdf[qdf["category"].notna()]
        .groupby("source")["category"]
        .agg(lambda x: x.mode()[0] if not x.mode().empty else None)
        .reset_index()
        .rename(columns={"category": "Most Frequent Exported Category"})
    )

    st.dataframe(cat_mode)
# =========================
with st.expander("🥧 Q21: Which weapon categories dominate the overall quantity of arms traded?"):
    st.markdown("""
    This **pie chart** shows the **top weapon categories** by total quantity.
    Smaller categories are grouped under "**Remaining categories**" for better clarity.

    ⚠️ Quantity counts units, so one missile and one ship both count as 1 — this is why Missiles dominate.
    """)

    # Sum quantity by category
    category_totals = (
        qdf[qdf["category"].notna()]
        .groupby("category")["quantity"]
        .sum()
        .sort_values(ascending=False)
    )

    # Top 5 + group the rest. (Not called "Other": the dataset already has a real category named "Other".)
    top_n = 5
    top_categories = category_totals[:top_n]
    other_sum = category_totals[top_n:].sum()

    pie_data = top_categories
    if other_sum > 0:
        pie_data = pd.concat([top_categories, pd.Series({"Remaining categories": other_sum})])

    # Plot
    fig, ax = plt.subplots(figsize=(8, 7))  # Larger size
    wedges, texts, autotexts = ax.pie(
        pie_data,
        labels=pie_data.index,
        autopct="%1.1f%%",
        startangle=140,
        labeldistance=1.15,  # Push labels outwards
        pctdistance=0.75,    # Place % inside
        textprops={"fontsize": 9},
        colors=plt.cm.Set2.colors
    )
    ax.set_title("Top 5 Weapon Categories by Quantity", fontsize=13)
    show_fig(fig)
# =========================
with st.expander("🎻 Q22: How does the quantity of arms traded vary across weapon categories?"):

    st.markdown("""
    This **violin plot** shows the distribution of arms trade quantities across each **weapon category**.  
    It helps us understand whether categories have **consistent trade sizes** or **wide variability**.
    """)

    # Filter valid categories and reasonable quantity values
    cat_violin = qdf[qdf["category"].notna() & (qdf["quantity"] > 0)].copy()

    # Remove outliers beyond 95th percentile for clearer visualization
    upper_limit = cat_violin["quantity"].quantile(0.95)
    cat_violin = cat_violin[cat_violin["quantity"] <= upper_limit]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=cat_violin, x="quantity", y="category", hue="category", palette="Pastel1", legend=False, ax=ax)
    ax.set_title("Distribution of Arms Quantity by Weapon Category", fontsize=13)
    ax.set_xlabel("Quantity per Transaction")
    ax.set_ylabel("Weapon Category")
    show_fig(fig)

# =====================================================================
# 🤖 MACHINE LEARNING SECTION
# =====================================================================
st.header("🤖 Machine Learning Models")
st.markdown("""
Beyond describing the past, two supervised ML models learn patterns from the data to make **predictions**:

| | Model A | Model B |
|---|---|---|
| **Task** | Regression | Classification (NLP) |
| **Predicts** | Delivery delay (years) | Weapon category |
| **Input** | Exporter, importer, category, quantity, order year | Weapon description text |
| **Best model** | Random Forest | TF-IDF + Logistic Regression |
""")


# Models are trained once and cached (st.cache_resource), not on every click.
# The leading underscore tells Streamlit not to hash the DataFrame argument.
@st.cache_resource(show_spinner="Training delivery delay models...")
def get_delay_models(_df):
    return train_delay_models(_df)


@st.cache_resource(show_spinner="Training category classifier...")
def get_category_models(_df):
    return train_category_models(_df)


# Highlight for the best score: dark green background + white bold text,
# readable in both light and dark themes.
BEST_STYLE = "background-color: #2e7d32; color: white; font-weight: bold;"

# A "fragment" reruns ONLY its own function when its widgets are used,
# instead of re-running the whole app (all 22 charts). This keeps the
# page from greying out on every prediction.
# (Named st.experimental_fragment in Streamlit 1.33–1.36, st.fragment from 1.37.)
fragment = getattr(st, "fragment", None) or st.experimental_fragment


@fragment
def delay_prediction_box(delay):
    with st.form("delay_form"):
        f1, f2, f3 = st.columns(3)
        src_list, tgt_list = delay["sources"], delay["targets"]
        p_source = f1.selectbox("Exporter", src_list,
                                index=src_list.index("United States") if "United States" in src_list else 0)
        p_target = f2.selectbox("Importer", tgt_list,
                                index=tgt_list.index("India") if "India" in tgt_list else 0)
        p_category = f3.selectbox("Weapon category", delay["categories"])
        f4, f5, f6 = st.columns(3)
        p_qty = f4.number_input("Quantity", min_value=1, max_value=25000, value=10)
        p_year = f5.number_input("Order year", min_value=2000, max_value=2030, value=2018)
        p_est = f6.checkbox("Order year is an estimate")
        submitted = st.form_submit_button("Predict delivery delay")

    if submitted:
        res = predict_delay(delay["model"], p_source, p_target, p_category, p_qty, p_year, p_est)
        st.success(
            f"📦 Predicted delay: **{res['prediction']:.1f} years** "
            f"→ expected delivery around **{p_year + round(res['prediction'])}**"
        )
        st.caption(f"Likely range (10th–90th percentile of the forest's trees): "
                   f"{res['low']:.1f} – {res['high']:.1f} years")
        if p_year > DELAY_TRAIN_MAX_YEAR:
            st.warning(f"The model was trained on orders up to {DELAY_TRAIN_MAX_YEAR}; "
                       "predictions for later years are an extrapolation.")


@fragment
def category_prediction_box(cat):
    user_text = st.text_input("Description", value="anti-ship missile")
    if user_text.strip():
        top = predict_category(cat["model"], user_text)
        st.success(f"Predicted category: **{top.loc[0, 'Category']}** "
                   f"({top.loc[0, 'Probability']:.0%} confidence)")
        st.bar_chart(top.set_index("Category"))


tab_delay, tab_cat = st.tabs(["⏳ A. Delivery Delay Predictor", "🧠 B. Weapon Category Classifier"])

# ---------------------------------------------------------------------
# MODEL A: Delivery Delay Predictor
# ---------------------------------------------------------------------
with tab_delay:
    st.subheader("⏳ Predict how long an arms deal will take to deliver")
    delay = get_delay_models(df)

    st.markdown(f"""
- **Target:** `delivery_date − order_date` in years
- **Features:** exporter, importer, weapon category, log(quantity), order year, order-year-estimated flag
- **Data:** {delay['n_train']:,} training deals / {delay['n_test']:,} test deals (80/20 split), orders 2000–{DELAY_TRAIN_MAX_YEAR}
""")

    # --- Model comparison ---
    st.markdown("#### 📊 Model Comparison (on unseen test data)")
    metrics = delay["metrics"]
    st.dataframe(
        metrics.style.format({"MAE (years)": "{:.2f}", "RMSE (years)": "{:.2f}", "R²": "{:.3f}"})
        .highlight_min(subset=["MAE (years)", "RMSE (years)"], props=BEST_STYLE)
        .highlight_max(subset=["R²"], props=BEST_STYLE),
        use_container_width=True, hide_index=True,
    )

    base_mae = metrics.loc[metrics["Model"].str.startswith("Baseline"), "MAE (years)"].iloc[0]
    rf_mae = metrics.loc[metrics["Model"] == "Random Forest", "MAE (years)"].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Random Forest MAE", f"{rf_mae:.2f} years")
    c2.metric("Baseline MAE", f"{base_mae:.2f} years")
    c3.metric("Error reduction vs baseline", f"{(1 - rf_mae / base_mae) * 100:.0f}%")

    with st.expander("ℹ️ What do these metrics mean?"):
        st.markdown("""
- **MAE (Mean Absolute Error):** on average, how many years the prediction is off by. Lower is better.
- **RMSE (Root Mean Squared Error):** like MAE but punishes big mistakes more. Lower is better.
- **R²:** share of the variation in delay that the model explains (1 = perfect, 0 = no better than the average).
- **Baseline:** a "dumb" model that always predicts the average delay. A real model must beat it.
""")

    # --- Charts ---
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 🎯 Predicted vs Actual")
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.scatter(delay["y_test"], delay["y_pred"], alpha=0.25, s=12, color="steelblue")
        lim = max(delay["y_test"].max(), delay["y_pred"].max())
        ax.plot([0, lim], [0, lim], "r--", linewidth=1, label="Perfect prediction")
        ax.set_xlabel("Actual delay (years)")
        ax.set_ylabel("Predicted delay (years)")
        ax.legend()
        show_fig(fig)

    with col_b:
        st.markdown("#### 🔑 Feature Importance")
        imp = delay["importance"]
        fig, ax = plt.subplots(figsize=(5, 4.5))
        sns.barplot(data=imp, x="Importance", y="Feature", color="teal", ax=ax)
        ax.set_xlabel("Increase in error when feature is shuffled (years)")
        ax.set_ylabel("")
        show_fig(fig)

    # --- Interactive prediction ---
    st.markdown("#### 🔮 Try it: predict a new deal")
    delay_prediction_box(delay)

    with st.expander("⚠️ Design decisions & limitations"):
        st.markdown(f"""
- **Why only orders up to {DELAY_TRAIN_MAX_YEAR}?** The data ends in 2023. A deal ordered in 2022 that takes 5 years is not
  delivered yet, so it has no delivery date. Recent orders would therefore only show *short* delays and teach the model a
  false pattern ("recent = fast"). This is called **right-censoring**. Training on older orders gives each deal at least ~5 years to be delivered.
- **Missing delivery dates (~16%)** are excluded: the true delay is unknown.
- **R² is modest**: real delays also depend on politics, sanctions, production queues and contract terms, which are not in the data.
- Countries with fewer than 10 deals are grouped as "infrequent" so the model does not overfit to rare countries.
""")

# ---------------------------------------------------------------------
# MODEL B: Weapon Category Classifier
# ---------------------------------------------------------------------
with tab_cat:
    st.subheader("🧠 Classify a weapon's category from its description (NLP)")
    cat = get_category_models(df)

    st.markdown(f"""
- **Input text:** weapon description + designation (e.g. *"portable sam strela-2"*)
- **Output:** one of {cat['confusion'].shape[0]} categories (Aircraft, Missiles, Ships...)
- **Text → numbers:** TF-IDF on single words and word pairs
- **Data:** {cat['n_unique']:,} unique weapon texts → {cat['n_train']:,} train / {cat['n_test']:,} test (stratified 80/20)
""")
    st.info("🔍 **Avoiding data leakage:** the raw data repeats the same weapon thousands of times "
            "(only 193 unique descriptions in 10,520 rows). Duplicates were removed **before** splitting, "
            "so the model is tested on texts it has never seen. Otherwise it would memorise them and show a fake 100% accuracy.")

    st.markdown("#### 📊 Model Comparison")
    st.dataframe(
        cat["metrics"].style.format({"Accuracy": "{:.1%}", "Macro F1": "{:.3f}"})
        .highlight_max(subset=["Accuracy", "Macro F1"], props=BEST_STYLE),
        use_container_width=True, hide_index=True,
    )
    with st.expander("ℹ️ What do these metrics mean?"):
        st.markdown("""
- **Accuracy:** % of test weapons classified correctly.
- **Macro F1:** average F1-score across all categories, treating each category equally. Important because
  the classes are imbalanced (Aircraft has thousands of rows, Satellites only a handful).
- **Precision:** of everything predicted as "Missiles", how many really were. **Recall:** of all real missiles, how many were found.
""")

    col_c, col_d = st.columns([3, 2])
    with col_c:
        st.markdown("#### 🧩 Confusion Matrix (Logistic Regression)")
        fig, ax = plt.subplots(figsize=(7, 5.5))
        sns.heatmap(cat["confusion"], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_xlabel("Predicted category")
        ax.set_ylabel("Actual category")
        plt.xticks(rotation=45, ha="right")
        show_fig(fig)
    with col_d:
        st.markdown("#### 📋 Per-Category Report")
        report = cat["report"].drop(index=["accuracy"], errors="ignore")
        st.dataframe(report.style.format({"precision": "{:.2f}", "recall": "{:.2f}",
                                          "f1-score": "{:.2f}", "support": "{:.0f}"}),
                     use_container_width=True)

    st.markdown("#### 🔮 Try it: type a weapon description")
    category_prediction_box(cat)
    st.caption("Examples to try: *fighter aircraft*, *main battle tank*, *frigate*, *air search radar*, *turbofan*, *self-propelled gun*")

# =========================
# 🏁 End of Dashboard
# =========================