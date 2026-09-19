import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

plt.rcParams["figure.dpi"] = 120

# ---------------------- 模拟数据生成 ----------------------
@st.cache_data
def gen_data():
    import numpy as np
    np.random.seed(42)
    n = 2000
    district_list = ["东城","北城","西城","南城"]
    reno_list = ["毛坯","简装","精装","豪装"]

    data = {
        "district": np.random.choice(district_list, size=n),
        "renovation": np.random.choice(reno_list, size=n),
        "area": np.random.normal(110,30,n).clip(50,200),
    }
    df = pd.DataFrame(data)

    # 基础价格系数
    district_coef = {"东城":3.2, "北城":2.8, "西城":2.4, "南城":2.0}
    reno_coef = {"毛坯":1.0, "简装":1.15, "精装":1.30, "豪装":1.55}

    df["base_price"] = df.apply(lambda x:
        x["area"] * district_coef[x["district"]] * reno_coef[x["renovation"]], axis=1
    )
    df["price"] = (df["base_price"] + np.random.normal(0,22,n)).round(0).clip(60,520)
    return df

df = gen_data()

# ---------------------- 侧边筛选器 ----------------------
st.set_page_config(page_title="房产数据分析看板", layout="wide")
st.title("🏠 房产数据分析看板")

with st.sidebar:
    st.header("筛选条件")
    sel_district = st.multiselect("选择城区", df["district"].unique(), default=df["district"].unique())
    sel_reno = st.multiselect("装修类型", df["renovation"].unique(), default=df["renovation"].unique())
    area_min, area_max = st.slider("面积区间", int(df["area"].min()), int(df["area"].max()),
                                   (int(df["area"].min()), int(df["area"].max())))

filtered_df = df[
    (df["district"].isin(sel_district)) &
    (df["renovation"].isin(sel_reno)) &
    (df["area"] >= area_min) & (df["area"] <= area_max)
].copy()

st.markdown(f"> 当前筛选后房源总数：**{len(filtered_df)}**")

# ---------------------- 基础统计指标 ----------------------
col1,col2,col3,col4 = st.columns(4)
col1.metric("房源数量", len(filtered_df))
col2.metric("均价(万元)", round(filtered_df["price"].mean(),1))
col3.metric("平均面积(㎡)", round(filtered_df["area"].mean(),1))
col4.metric("中位数房价", round(filtered_df["price"].median(),1))

# ---------------------- 1.面积-房价散点图 matplotlib ----------------------
st.subheader("面积与房价关系")
fig_scatter, ax_scatter = plt.subplots(figsize=(10,5))
ax_scatter.scatter(filtered_df["area"], filtered_df["price"], alpha=0.4, s=12)
ax_scatter.set_xlabel("Area (㎡)")
ax_scatter.set_ylabel("Price (10k yuan)")
ax_scatter.set_title("Area‑Price Distribution")
ax_scatter.grid(linestyle="--", alpha=0.5)
st.pyplot(fig_scatter)

# ---------------------- 2.各城区房价对比箱线图 matplotlib ----------------------
st.subheader("各城区房价对比")
districts_order = ['东城', '北城', '西城', '南城']
valid_districts = [d for d in districts_order if d in filtered_df['district'].unique()]
if valid_districts:
    data_to_plot = [filtered_df[filtered_df['district'] == d]['price'] for d in valid_districts]
    fig_dist, ax_dist = plt.subplots(figsize=(10,5))
    bp = ax_dist.boxplot(
        data_to_plot,
        tick_labels=["East","North","West","South"],
        patch_artist=True
    )
    colors = ['#ff9999','#66B2ff','#99ff99','#ffcc99']
    for patch,color in zip(bp["boxes"], colors[:len(data_to_plot)]):
        patch.set_facecolor(color)
    ax_dist.set_ylabel("Price (10k yuan)")
    ax_dist.set_title("Price by District")
    ax_dist.grid(axis='y', linestyle="--", alpha=0.5)
    st.pyplot(fig_dist)
else:
    st.warning("筛选后无可用城区数据")

# ---------------------- 3.装修程度与房价柱状图 matplotlib ----------------------
st.subheader("装修程度与房价")
renovation_order = ['毛坯','简装','精装','豪装']
reno_eng = ["Raw","Simple","Fine","Luxury"]
reno_stats = filtered_df.groupby("renovation")["price"].agg(["mean","std"]).reindex(renovation_order).dropna()

fig_reno, ax_reno = plt.subplots(figsize=(10,5))
bars = ax_reno.bar(
    reno_eng,
    reno_stats["mean"],
    yerr=reno_stats["std"],
    capsize=6,
    color=["#8b8682","#cdbe70","#ffd700","#8860b0"],
    edgecolor="black"
)
ax_reno.set_ylabel("Average Price (10k yuan)")
ax_reno.set_title("Price by Renovation Condition")
ax_reno.grid(axis="y", linestyle="--", alpha=0.5)
for bar in bars:
    h = bar.get_height()
    ax_reno.text(bar.get_x()+bar.get_width()/2, h+5, f"{h:.0f}", ha="center")
st.pyplot(fig_reno)

# ---------------------- 4.地段 × 装修交互热力图 plotly美化版 ----------------------
st.subheader("地段与装修程度交互分析")
pivot = filtered_df.pivot_table(
    values="price",
    index="district",
    columns="renovation",
    aggfunc="mean",
    fill_value=0
)

rename_idx = {"东城":"东城","北城":"北城","西城":"西城","南城":"南城"}
rename_col = {"毛坯":"毛坯","简装":"简装","精装":"精装","豪装":"豪装"}
pivot.index = [rename_idx.get(i,i) for i in pivot.index]
pivot.columns = [rename_col.get(c,c) for c in pivot.columns]

fig_pivot = px.imshow(
    pivot,
    text_auto=".0f",
    color_continuous_scale="YlOrRd",
    labels={"color":"平均房价（万元）"},
    title="各地段 × 装修程度平均房价热力图"
)
fig_pivot.update_layout(
    font={"family":"Noto Sans SC, sans‑serif", "size":14},
    xaxis_title="装修程度",
    yaxis_title="城区",
    title_x=0.5,
    margin={"l":80,"r":80,"t":80,"b":80},
    coloraxis_colorbar={
        "title_font":{"size":13},
        "tickfont":{"size":12}
    }
)
fig_pivot.update_xaxes(tickfont_size=13)
fig_pivot.update_yaxes(tickfont_size=13)
st.plotly_chart(fig_pivot, use_container_width=True)

# ---------------------- 原始数据预览 ----------------------
with st.expander("查看原始数据"):
    st.dataframe(filtered_df.reset_index(drop=True))
