import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns

# Page setup
st.set_page_config(page_title="Arms Trade Dashboard", layout="wide")
st.title("🔍 Global Arms Trade Analysis (2000–2023)")

# =========================
# 📥 Load & Clean Data
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("data/global_arms_transfer_2000_2023.csv")
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

    if "orderyrest" in df.columns:
        df.rename(columns={"orderyrest": "order_year_est"}, inplace=True)

    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
        df["order_year"] = df["order_date"].dt.year
        df["order_date"] = df["order_date"].dt.date

    if "delivery_date" in df.columns:
        df["delivery_date"] = pd.to_datetime(df["delivery_date"], errors="coerce")
        df["delivery_date"] = df["delivery_date"].dt.date

    if "order_year_est" in df.columns:
        df["order_year_est"] = pd.to_numeric(df["order_year_est"], errors="coerce").astype("Int64")
        df["order_year"] = df["order_year"].fillna(df["order_year_est"])

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
    st.info("ℹ️ No filters selected yet — please choose at least one filter from the sidebar.")
# =========================
# ❓ Arms Trade Questions
# =========================
st.header("❓ Arms Trade Q&A — Data-Driven Insights")
# =========================
# Q1: Top 10 Exporters Chart (Altair)
# =========================
with st.expander("🚢 Q1. What are the top 10 exporting countries (2000–2023)?"):
    top_sources = (
        df["source"]
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
        color=alt.Color("source_country:N", legend=None)
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
        df["target"]
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
        color=alt.Color("target_country:N", legend=None)
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
    
    cat_df = df[df["category"].notna()].groupby("category")["quantity"].sum().sort_values(ascending=True).reset_index()
    
    # Show data as a table
    st.dataframe(cat_df.rename(columns={"category": "Weapon Category", "quantity": "Total Quantity"}), use_container_width=True)

    # Plot lollipop chart
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hlines(y=cat_df["category"], xmin=0, xmax=cat_df["quantity"], color="skyblue", linewidth=2)
    ax.plot(cat_df["quantity"], cat_df["category"], "o", markersize=6, color="navy")
    ax.set_xlabel("Total Quantity")
    ax.set_ylabel("Category")
    ax.set_title("Weapon Categories by Total Quantity (Lollipop View)")
    st.pyplot(fig)
# =========================
# 📈 Q4: Annual Trend of Weapon Orders (Altair)
# =========================
with st.expander("📈 Q4: What is the annual trend of weapon orders over time?"):
    st.markdown("This line plot shows how total weapon quantities have changed year by year.")

    yearly = df.groupby("order_year")["quantity"].sum().reset_index()

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
    st.pyplot(fig)
# =========================
# ("🌐 Q5: Heatmap of arms transfers between top 5 countries"):

with st.expander("🌐 Q5: Arms transfers between top 5 countries"):
    st.markdown("This heatmap shows total arms transferred (by quantity) between the top 5 source and target countries.")

    # Select top 5 source and target countries
    top_sources = df["source"].value_counts().head(5).index
    top_targets = df["target"].value_counts().head(5).index

    # Filter and pivot
    heat_df = df[df["source"].isin(top_sources) & df["target"].isin(top_targets)]
    pivot_df = heat_df.groupby(["source", "target"])["quantity"].sum().reset_index()
    heatmap_data = pivot_df.pivot(index="source", columns="target", values="quantity").fillna(0)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(5.5, 4))
    sns.heatmap(heatmap_data, cmap="YlGnBu", annot=True, fmt=".0f", linewidths=0.3, cbar_kws={"shrink": 0.7})
    ax.set_title("Top 5 Source vs Target Arms Transfers", fontsize=11)
    st.pyplot(fig)

# =========================
# 🚚 Q6: Years with Highest Transfers (Altair)
# =========================
with st.expander("🚚 Q6: Which years saw the highest number of arms transfers?"):
    st.markdown("This chart shows the number of transfer records per year (not quantity).")

    year_counts = df["order_year"].value_counts().sort_index().reset_index()
    year_counts.columns = ["order_year", "transfer_count"]

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(year_counts["order_year"], year_counts["transfer_count"], marker='o', linestyle='-', color='orange')
    ax.set_title("Number of Arms Transfers by Year", fontsize=14)
    ax.set_xlabel("Order Year")
    ax.set_ylabel("Number of Transfers")
    ax.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# =========================
# 🧮 Q7: Avg Quantity per Country (Altair)
# =========================
with st.expander("🧮 Q7: What is the average quantity of arms traded per transaction for each country?"):
    # Get top 5 countries by average quantity
    avg_qty_df = (
        df.groupby("source")["quantity"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
        .reset_index()
    )
    avg_qty_df.columns = ["Source Country", "Average Quantity"]
    st.dataframe(avg_qty_df.style.format({"Average Quantity": "{:.2f}"}), use_container_width=True)

    # Filter data for those top 5 countries
    top5 = avg_qty_df["Source Country"].tolist()
    plot_df = df[df["source"].isin(top5)]

    # Scatterplot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=plot_df, x="quantity", y="source", hue="source", s=60, alpha=0.7, ax=ax)
    ax.set_title("Scatter Plot: Quantity per Transaction for Top 5 Exporting Countries")
    ax.set_xlabel("Quantity")
    ax.set_ylabel("Source Country")
    ax.legend(title="Country", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.3)

    st.pyplot(fig)

# =========================
# 📈 Q8: Which countries had the largest growth in arms exports between 2000 and 2023?
# =========================
with st.expander("🟢 Q8: Which countries had the highest average arms quantity per trade?"):

    st.markdown("""
    This bubble chart shows:
    - **X-axis:** Number of trades  
    - **Y-axis:** Country  
    - **Bubble size:** Average quantity per trade
    """)


    # Compute average quantity and trade count
    country_stats = (
        df.groupby("source")
        .agg(trade_count=("quantity", "count"), avg_quantity=("quantity", "mean"))
        .sort_values(by="avg_quantity", ascending=False)
        .head(10)
        .reset_index()
    )

    st.dataframe(country_stats.style.format({"avg_quantity": "{:.1f}", "trade_count": "{:,.0f}"}))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(
        country_stats["trade_count"],
        country_stats["source"],
        s=country_stats["avg_quantity"] * 2,  # Scale bubble size
        alpha=0.6,
        c=range(len(country_stats)),  # Add color variation
        cmap="viridis",
        edgecolors="black"
    )

    for i, row in country_stats.iterrows():
        ax.text(
            row["trade_count"], row["source"],
            f'{row["avg_quantity"]:.0f}', va='center', ha='center', fontsize=8, color='black'
        )

    ax.set_xlabel("Number of Trades")
    ax.set_ylabel("Country")
    ax.set_title("Top 10 Exporters by Avg Quantity per Trade (Bubble Chart)", fontsize=12)
    st.pyplot(fig)

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
        df["designation"]
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
    st.pyplot(fig)
# =========================
with st.expander("🧩 Q10: How many unique weapon designations were traded from 2000 to 2023?"):
    
    st.markdown("""
This insight reveals how **diverse** the global arms trade is in terms of weapon models or designations.
    """)

    unique_designations = df["designation"].dropna().str.lower().str.strip()
    unique_designations = unique_designations[~unique_designations.isin(["-", "unknown", "n/a", "other"])]
    total_unique = unique_designations.nunique()

    st.success(f"🔢 Total unique weapon designations traded: **{total_unique:,}**")

# =========================
with st.expander("🗓️ Q11: In which year did the highest number of different countries import arms?"):

    st.markdown("""
This insight reveals the year in which the **largest number of unique countries** participated as arms importers.  
It indicates the **peak of global importing activity**.
""")

    df["order_year"] = pd.to_numeric(df["order_year"], errors="coerce")
    year_target_counts = df.dropna(subset=["order_year", "target"]).groupby("order_year")["target"].nunique()
    top_year = year_target_counts.idxmax()
    top_count = year_target_counts.max()

    st.success(f"📅 The year **{top_year}** had the most diverse arms import activity, with **{top_count} countries** importing arms.")

# =========================
with st.expander("⏳ Q12: What is the average delivery delay (in years) by exporting country?"):

    st.markdown("Shows the **average delivery delay** between order and delivery per country.")

    delay_df = df.copy()
    delay_df["delivery_date"] = pd.to_datetime(delay_df["delivery_date"], errors="coerce")
    delay_df["order_date"] = pd.to_datetime(delay_df["order_date"], errors="coerce")
    delay_df["delay_years"] = (delay_df["delivery_date"] - delay_df["order_date"]).dt.days / 365

    delay_avg = (
        delay_df.groupby("source")["delay_years"]
        .mean()
        .dropna()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    st.dataframe(delay_avg.style.format({"delay_years": "{:.2f}"}))

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=delay_avg, y="source", x="delay_years", palette="mako", ax=ax)
    ax.set_title("Average Delivery Delay (in Years) by Exporting Country")
    ax.set_xlabel("Average Delay (Years)")
    ax.set_ylabel("Exporting Country")
    st.pyplot(fig)
# =========================
with st.expander("🔁 Q13: Which countries consistently trade specific categories?"):

    st.markdown("This heatmap shows how many times each country exported different **weapon categories**.")

    cat_matrix = df.pivot_table(
        index="source", columns="category", values="quantity", aggfunc="count"
    ).fillna(0)

    top_sources = df["source"].value_counts().head(10).index
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(cat_matrix.loc[top_sources], annot=True, fmt=".0f", cmap="YlGnBu", ax=ax)
    ax.set_title("Export Frequency of Weapon Categories by Top Countries")
    st.pyplot(fig)
# =========================
with st.expander("📊 Q14: Are certain categories becoming more or less popular over time?"):

    st.markdown("This **stacked area chart** shows how category popularity changed over time (by quantity).")

    cat_trend = df[df["category"].notna()].groupby(["order_year", "category"])["quantity"].sum().reset_index()

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
    st.pyplot(fig)

# =========================
with st.expander("🤝  Q15: What are the most common exporter-importer pairs?"):

    st.markdown("This **heatmap** highlights the top 10 exporter-importer country pairs by number of trades.")

    # Get top 10 pairs
    pair_df = (
        df.groupby(["source", "target"])
        .size()
        .reset_index(name="trade_count")
        .sort_values("trade_count", ascending=False)
        .head(10)
    )

    # Create pivot table for heatmap
    heat_df = pair_df.pivot(index="source", columns="target", values="trade_count").fillna(0)

    # Plot heatmap using seaborn
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(heat_df, annot=True, fmt=".0f", cmap="Blues", linewidths=0.5, ax=ax)
    ax.set_title("Top 10 Exporter-Importer Pairs — Heatmap", fontsize=12)
    ax.set_xlabel("Importer")
    ax.set_ylabel("Exporter")
    st.pyplot(fig)

# =========================
with st.expander("📦 Q16: Which countries import the highest quantity per transaction?"):
    st.markdown("This reveals countries that typically import in **larger batches**, based on average quantity per deal.")

    top_avg_imports = (
        df[df["quantity"].notna()]
        .groupby("target")["quantity"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    top_avg_imports.columns = ["Importing Country", "Avg Quantity per Transaction"]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=top_avg_imports, x="Avg Quantity per Transaction", y="Importing Country", palette="crest", ax=ax)
    ax.set_title("Top 10 Importers by Avg Quantity per Transaction")
    st.pyplot(fig)
# =========================
with st.expander("🛠️ Q17: What are the most commonly traded weapon categories by decade?"):
    st.markdown("""
This reveals how the **top traded weapon categories** have changed across decades:  
- 📅 **2000s** (2000–2009)  
- 📅 **2010s** (2010–2019)  
- 📅 **2020s** (2020–2023)
""")

    df_decade = df.copy()
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
    st.pyplot(fig)
# =========================
with st.expander("🏗️ Q18: How many arms deals had missing delivery dates?"):
    st.markdown("Helps identify **data gaps** or possibly delayed or cancelled shipments.")
    
    missing_count = df["delivery_date"].isna().sum()
    total = len(df)
    percent = (missing_count / total) * 100

    st.success(f"❌ Missing Delivery Dates: **{missing_count:,}** out of **{total:,}** records (**{percent:.2f}%**).")
# =========================
with st.expander("🛠️ Q19: Which weapon designations had the longest average delivery delay?"):
    st.markdown("Focuses on **logistical or production delays** by designation (model).")

    delay_df = df.copy()
    delay_df["delivery_date"] = pd.to_datetime(delay_df["delivery_date"], errors="coerce")
    delay_df["order_date"] = pd.to_datetime(delay_df["order_date"], errors="coerce")
    delay_df["delay_years"] = (delay_df["delivery_date"] - delay_df["order_date"]).dt.days / 365

    top_delays = (
        delay_df[delay_df["designation"].notna()]
        .groupby("designation")["delay_years"]
        .mean()
        .dropna()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    # Show table
    st.dataframe(top_delays.style.format({"delay_years": "{:.2f}"}))

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=top_delays,
        y="designation",
        x="delay_years",
        palette="crest",
        ax=ax
    )
    ax.set_title("Top 10 Designations by Average Delivery Delay (in Years)")
    ax.set_xlabel("Average Delay (Years)")
    ax.set_ylabel("Weapon Designation")
    st.pyplot(fig)

# =========================
with st.expander("🎯 Q20: What is the most frequently traded weapon category per country?"):
    st.markdown("This shows each country's **top exported category**, reflecting their specialization.")

    cat_mode = (
        df[df["category"].notna()]
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
    Smaller categories are grouped under "**Other**" for better clarity.
    """)

    # Sum quantity by category
    category_totals = (
        df[df["category"].notna()]
        .groupby("category")["quantity"]
        .sum()
        .sort_values(ascending=False)
    )

    # Top 5 + group rest as 'Other'
    top_n = 5
    top_categories = category_totals[:top_n]
    other_sum = category_totals[top_n:].sum()

    pie_data = pd.concat([top_categories, pd.Series({"Other": other_sum})])

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
    ax.set_title("Top Weapon Categories by Quantity (with Other)", fontsize=13)
    st.pyplot(fig)
# =========================
with st.expander("🎻 Q22: How does the quantity of arms traded vary across weapon categories?"):

    st.markdown("""
    This **violin plot** shows the distribution of arms trade quantities across each **weapon category**.  
    It helps us understand whether categories have **consistent trade sizes** or **wide variability**.
    """)

    # Filter valid categories and reasonable quantity values
    cat_violin = df[df["category"].notna() & (df["quantity"] > 0)].copy()

    # Remove outliers beyond 95th percentile for clearer visualization
    upper_limit = cat_violin["quantity"].quantile(0.95)
    cat_violin = cat_violin[cat_violin["quantity"] <= upper_limit]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=cat_violin, x="quantity", y="category", palette="Pastel1", ax=ax)
    ax.set_title("Distribution of Arms Quantity by Weapon Category", fontsize=13)
    ax.set_xlabel("Quantity per Transaction")
    ax.set_ylabel("Weapon Category")
    st.pyplot(fig)

# =========================
# 🏁 End of Dashboard
# =========================