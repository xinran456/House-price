import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# 页面配置
st.set_page_config(
    page_title="房地产数据分析驾驶舱",
    page_icon="🏠",
    layout='wide',
    initial_sidebar_state='expanded'
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


@st.cache_data(ttl=3600)
def generate_house_data(n=2000):
    """生成模拟的二手房交易数据"""
    np.random.seed(2024)
    # 数值型特征
    area = np.random.normal(130, 30, n).astype(int)
    area = np.clip(area, a_min=60, a_max=250)
    bedrooms = np.random.choice(a=[1, 2, 3, 4, 5], p=[0.05, 0.30, 0.40, 0.20, 0.05], size=n)
    bathrooms = np.random.choice(a=[1, 2, 3], p=[0.30, 0.55, 0.15], size=n)

    age = np.random.exponential(10, n).astype(int)
    age = np.clip(age, a_min=0, a_max=40)
    distance_to_center = np.round(np.random.exponential(4, n) + 0.5, decimals=1)

    # 分类特征
    districts = np.random.choice(a=['东城', '西城', '南城', '北城'], p=[0.30, 0.25, 0.20, 0.25], size=n)
    floors = np.random.choice(a=['低', '中', '高'], p=[0.30, 0.40, 0.30], size=n)
    renovations = np.random.choice(a=['毛坯', '简装', '精装', '豪装'], p=[0.10, 0.30, 0.40, 0.20], size=n)

    # 系数映射
    districts_coef = {'东城': 80, '西城': 40, '南城': 20, '北城': 60}
    floors_coef = {'低': -15, '中': 0, '高': 20}
    reno_coef = {'毛坯': -30, '简装': -10, '精装': 10, '豪装': 40}

    district_factor = np.array([districts_coef[d] for d in districts])
    floor_factor = np.array([floors_coef[d] for d in floors])
    reno_factor = np.array([reno_coef[d] for d in renovations])

    # 生成价格 （线性组合 + 噪声）
    base_price = (area * 1.5 + bedrooms * 10 + bathrooms * 8
                  - age * 1.2 - distance_to_center * 5 +
                  district_factor + floor_factor + reno_factor
                  )
    # 修复normal函数关键字参数语法错误
    noise = np.random.normal(loc=0, scale=0.12 * base_price)
    price = np.round(base_price + noise, decimals=0).astype(int)
    price = np.clip(price, a_min=50, a_max=800)

    df = pd.DataFrame({
        'price': price,
        'area': area,
        'bedrooms': bedrooms,
        'age': age,
        'distance_to_center': distance_to_center,
        'floor': floors,
        'district': districts,
        'renovation': renovations,
    })
    return df


# 加载数据
with st.spinner('正在加载模拟数据'):
    df = generate_house_data()

st.sidebar.title('筛选控制')
# 城区多选
selected_districts = st.sidebar.multiselect(label='城区',
                                            options=sorted(df['district'].unique()),
                                            default=sorted(df['district'].unique())
                                            )
# 装修
selected_renovations = st.sidebar.multiselect(label='装修程度',
                                              options=sorted(df['renovation'].unique()),
                                              default=sorted(df['renovation'].unique()),
                                              )
# 楼层多选 【修复label复制粘贴错误】
selected_floors = st.sidebar.multiselect(label='楼层',
                                         options=sorted(df['floor'].unique()),
                                         default=sorted(df['floor'].unique()),
                                         )
# 价格范围
price_min, price_max = int(df['price'].min()), int(df['price'].max())
price_range = st.sidebar.slider("总价范围（万元）",
                                min_value=price_min,
                                max_value=price_max,
                                value=(price_min, price_max)
                                )
# 面积范围
area_min, area_max = int(df['area'].min()), int(df['area'].max())
area_range = st.sidebar.slider("面积范围（㎡）",
                               min_value=area_min,
                               max_value=area_max,
                               value=(area_min, area_max)
                               )
# 房龄范围
age_min, age_max = int(df['age'].min()), int(df['age'].max())
age_range = st.sidebar.slider("房龄范围（年）",
                              min_value=age_min,
                              max_value=age_max,
                              value=(age_min, age_max)
                              )
# 应用筛选
filtered_df = df[
    (df['district'].isin(selected_districts)) &
    (df['renovation'].isin(selected_renovations)) &
    (df['floor'].isin(selected_floors)) &
    (df['price'] >= price_range[0]) &
    (df['price'] <= price_range[1]) &
    (df['age'] >= age_range[0]) &
    (df['age'] <= age_range[1]) &
    (df['area'] >= area_range[0]) &
    (df['area'] <= area_range[1])
]

# KPI 指标卡片
st.title('房地产市场价格分析驾驶舱')
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label='总房源数', value=f'{len(filtered_df)}套')
with col2:
    st.metric(label='平均价格', value=f'{filtered_df["price"].mean():.0f}万元')
with col3:
    st.metric(label='中位价格', value=f'{filtered_df["price"].median():.0f}万元')

with col4:
    most_expensive = filtered_df.groupby('district')['price'].mean().idxmax()
    # 修复metric参数缺失value的语法错误
    st.metric(label="最贵的区域", value=most_expensive)

st.divider()

# 展示筛选后数据表
st.subheader("筛选后房源数据")
st.dataframe(filtered_df, use_container_width=True)



#9.19
#9.19
#9.19
#9.19
# 图表区域
# 1. 价格分布（直方图 + 箱线图）
# 专门适配 Streamlit 云！本地也照样能用，两边都不会崩
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei"]
plt.rcParams["axes.unicode_minus"] = False # 解决负号变方框
st.subheader("房价的分布概览")
fig,axes = plt.subplots(nrows=1, ncols=2,figsize=(12,4))
axes[0].hist(filtered_df['price'],bins=30,color='skyblue',edgecolor='black',alpha=0.7)
axes[0].axvline(filtered_df['price'].mean(),color='red',linestyle='--',
                label=f'均值{filtered_df["price"].mean():.0f}万')
axes[0].axvline(filtered_df['price'].median(),color='green',linestyle='--',
                label=f'中位数{filtered_df["price"].median():.0f}万')
axes[0].set_title('房价分布直方图')
axes[0].set_xlabel('总价（万元）')
axes[0].set_ylabel('频数')
axes[0].legend()

axes[1].boxplot(filtered_df['price'],vert=True,patch_artist=True,
                boxprops=dict(color='lightgreen'))
axes[1].set_title('房价箱线图')
axes[1].set_ylabel('总价（万元）')

st.pyplot(fig)

# 2. 相关性热力图【已修复bathrooms缺失导致KeyError】
# 专门适配 Streamlit 云！本地也照样能用，两边都不会崩
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei"]
plt.rcParams["axes.unicode_minus"] = False # 解决负号变方框
st.subheader("数值特征相关性")
numeric_cols = ['price','area','bedrooms','bathrooms','age','distance_to_center']
# 自动过滤掉df中不存在的字段，防止KeyError
numeric_cols = [col for col in numeric_cols if col in filtered_df.columns]

if len(numeric_cols) >=2: #至少2个字段才可以计算相关系数
    corr = filtered_df[numeric_cols].corr()
    fig_corr,ax_corr = plt.subplots(figsize=(8,6))
    cax = ax_corr.matshow(corr,cmap='coolwarm',vmin=-1,vmax=1)
    plt.colorbar(cax)
    ax_corr.set_xticks(range(len(corr.columns)))
    ax_corr.set_yticks(range(len(corr.columns)))
    ax_corr.set_xticklabels(corr.columns,rotation=30)
    ax_corr.set_yticklabels(corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax_corr.text(j,i,f"{corr.iloc[i,j]:.2f}",ha='center',va='center',color='black')
    ax_corr.set_title("相关系数热力图")
    st.pyplot(fig_corr)
else:
    st.warning("可用数值特征不足，无法绘制相关性热力图")


# 3. 面积 VS 价格 （散点 + 趋势）
st.subheader('面积和价格关系')
fig_area,ax_area = plt.subplots(figsize=(10,5))
ax_area.scatter(filtered_df['area'],filtered_df['price'],alpha=0.5,s=20,c='steelblue')
z = np.polyfit(filtered_df['area'],filtered_df['price'], deg=1)
p = np.poly1d(z)

x_line = np.linspace(filtered_df['area'].min(),filtered_df['area'].max(), num=100)
ax_area.plot(x_line, p(x_line),color='red',linewidth=2,label=f'趋势:price={z[0]:.2f}*area+{z[1]:.2f}')
ax_area.set_xlabel('建筑面积(m²)')
ax_area.set_ylabel('总价（万元）')
ax_area.legend()
ax_area.grid(True,linestyle='--',alpha=0.5)

st.pyplot(fig_area)

# 4. 各城区房价箱线图
# 专门适配 Streamlit 云！本地也照样能用，两边都不会崩
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei"]
plt.rcParams["axes.unicode_minus"] = False # 解决负号变方框
st.subheader("各城区房价对比")
districts_order = ['东城','北城','西城','南城']
# 检查筛选以后是否有数据
if len(filtered_df) > 0:
    valid_districts = [d for d in districts_order if d in filtered_df['district'].unique()]
    if valid_districts:
        data_to_plot = [filtered_df[filtered_df['district'] == d]['price'] for d in valid_districts]
        fig_dist,ax_dist = plt.subplots(figsize=(10,5))
        bp = ax_dist.boxplot(data_to_plot,tick_labels=valid_districts,patch_artist=True)
        colors = ['#ff9999','#66B2ff','#99ff99','#ffcc99']
        for patch,color in zip(bp['boxes'],colors[:len(data_to_plot)]):
            patch.set_facecolor(color)

        ax_dist.set_ylabel('总价（万元）')
        ax_dist.grid(axis='y',linestyle='--',alpha=0.5)
        st.pyplot(fig_dist)
    else:
        st.warning('当前筛选条件没有数据，无法绘制城区箱线图')
else:
    st.warning('当前筛选条件没有房源数据，请放宽条件')

# 5. 装修程度 VS 房价
st.subheader("装修程度与房价")
renovation_order = ['毛坯','简装','精装','豪装']
renovation_states = (filtered_df.groupby('renovation')['price']
                    .agg(['mean','std'])
                    .reindex(renovation_order)
                    .dropna()
                    )

fig_reno,ax_reno = plt.subplots(figsize=(10,5))
bars = ax_reno.bar(renovation_states.index,renovation_states['mean'],
                   yerr=renovation_states['std'],
                   capsize=5,
                   color=['#8b8682','#cdbe70','#ffd700','#8860b0'],
                   edgecolor='black')
ax_reno.set_ylabel('平均总价（万元）')
ax_reno.grid(axis='y',linestyle='--',alpha=0.5)
for bar in bars:
    height = bar.get_height()
    ax_reno.text(bar.get_x() + bar.get_width() /2.,
                 height + 5 ,
                 f'{height:.0f}',ha='center',va='bottom')

st.pyplot(fig_reno)

# 6. 房龄 VS 价格
if "age" in filtered_df.columns and "price" in filtered_df.columns:
    st.subheader("房龄和价格关系")
    fig_age,ax_age = plt.subplots(figsize=(10,5))
    ax_age.scatter(filtered_df['age'],filtered_df['price'],alpha=0.4,s=15,c='darkgreen')
    z2 = np.polyfit(filtered_df['age'],filtered_df['price'], deg=2)
    p2 = np.poly1d(z2)

    x_line2 = np.linspace(filtered_df['age'].min(),filtered_df['age'].max(), num=100)
    ax_age.plot(x_line2, p2(x_line2),color='red',linewidth=2,label='二次拟合')
    ax_age.set_xlabel('房龄（年）')
    ax_age.set_ylabel('总价（万元）')
    ax_age.legend()
    ax_age.grid(True,linestyle='--',alpha=0.5)
    st.pyplot(fig_age)
else:
    st.info("缺少age或price字段，跳过【房龄‑价格】图")

# 7. 距离市中心 VS 价格
if "distance_to_center" in filtered_df.columns and "price" in filtered_df.columns:
    st.subheader("距离市中心与价格关系")
    fig_dist2,ax_dist2 = plt.subplots(figsize=(10,5))
    ax_dist2.scatter(filtered_df['distance_to_center'],filtered_df['price'],alpha=0.4,s=15,c='purple')
    z3 = np.polyfit(filtered_df['distance_to_center'],filtered_df['price'], deg=1)
    p3 = np.poly1d(z3)

    x_line3 = np.linspace(filtered_df['distance_to_center'].min(),filtered_df['distance_to_center'].max(), num=100)
    ax_dist2.plot(x_line3, p3(x_line3),color='red',linewidth=2,
                   label=f'trend:price={z3[0]:.2f}*dist+{z3[1]:.2f}')
    ax_dist2.set_xlabel('距离（公里）')
    ax_dist2.set_ylabel('总价（万元）')
    ax_dist2.legend()
    ax_dist2.grid(True,linestyle='--',alpha=0.5)
    st.pyplot(fig_dist2)
else:
    st.info("缺少distance_to_center或price字段，跳过【距离‑价格】图")


# 8. 地段 × 装修热力图
# 专门适配 Streamlit 云！本地也照样能用，两边都不会崩
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei"]
plt.rcParams["axes.unicode_minus"] = False # 解决负号变方框
if {"district","renovation","price"}.issubset(filtered_df.columns):
    st.subheader("地段与装修程度交互分析")
    pivot = filtered_df.pivot_table(values='price',index='district',columns='renovation',aggfunc='mean',fill_value=0)
    fig_pivot = px.imshow(pivot,text_auto='.0f',
                          color_continuous_scale='YlOrRd',
                          labels=dict(color="平均房价（万元）"),
                          title='各地段×装修程度平均房价热力图')
    st.plotly_chart(fig_pivot,use_container_width=True)
else:
    st.info("缺少district / renovation / price字段，跳过地段装修热力图")

st.divider()


# ---------------- 房价预测工具（兼容列缺失安全版） ----------------
st.subheader("房价预测工具（基于线性回归）")

# 期望特征列表
expect_features = ['area', 'bedrooms', 'bathrooms', 'age', 'distance_to_center']
# 只保留真实存在的特征
valid_features = [f for f in expect_features if f in filtered_df.columns]

y_col = "price"
if len(valid_features)>=2 and y_col in filtered_df.columns and len(filtered_df)>1:
    X = filtered_df[valid_features]
    y = filtered_df[y_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    col1, col2 = st.columns(2)
    col1.metric("模型 MAE", f"{mae:.2f} 万元")
    col2.metric("模型 R²", f"{r2:.3f}")

    # 真实 vs 预测散点图
    fig_pred, ax_pred = plt.subplots(figsize=(8, 6))
    ax_pred.scatter(y_test, y_pred, alpha=0.5, s=20)
    ax_pred.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
    ax_pred.set_xlabel("真实房价（万元）")
    ax_pred.set_ylabel("预测房价（万元）")
    ax_pred.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig_pred)

    # 特征系数展示
    st.write("**模型系数**：")
    coef_df = pd.DataFrame({
        '特征': X.columns,
        '系数': model.coef_
    })
    st.dataframe(coef_df)

    # 自定义预测表单：动态生成输入框（只对有效特征显示）
    st.write("**自定义预测**：输入房屋特征，预测价格")
    with st.form("predict_form"):
        input_dict = {}
        cols = st.columns(3)
        idx=0
        if 'area' in valid_features:
            with cols[idx%3]:
                input_dict['area'] = st.number_input(label="面积（m²）", min_value=60, max_value=250, value=120)
            idx+=1
        if 'bedrooms' in valid_features:
            with cols[idx%3]:
                input_dict['bedrooms'] = st.number_input(label="卧室数", min_value=1, max_value=5, value=3)
            idx+=1
        if 'bathrooms' in valid_features:
            with cols[idx%3]:
                input_dict['bathrooms'] = st.number_input(label="浴室数", min_value=1, max_value=3, value=2)
            idx+=1
        if 'age' in valid_features:
            with cols[idx%3]:
                input_dict['age'] = st.number_input(label="房龄（年）", min_value=0, max_value=40, value=10)
            idx+=1
        if 'distance_to_center' in valid_features:
            with cols[idx%3]:
                input_dict['distance_to_center'] = st.number_input(label="距市中心 (km)", min_value=0.5, max_value=20.0, value=5.0, step=0.1)
            idx+=1

        submitted = st.form_submit_button("预测房价")
        if submitted:
            # 按照模型训练时的特征顺序组装数据
            input_arr = np.array([[input_dict[f] for f in valid_features]])
            pred_price = model.predict(input_arr)[0]
            st.success(f"预测总价：**{pred_price:.0f} 万元**")

else:
    st.warning(f"有效特征不足(当前可用:{valid_features})，或数据量不足，无法训练模型。请放宽筛选条件。")


st.divider()

st.subheader("📋 数据明细")
# column_config也要过滤掉不存在的列
all_config = {
    'price': st.column_config.NumberColumn(label='总价（万元）', format="%.0f"),
    'area': '面积（m²）',
    'bedrooms': '卧室',
    'bathrooms': '浴室',
    'age': '房龄（年）',
    'distance_to_center': '距市中心（km）',
    'district': '城区',
    'floor': '楼层',
    'renovation': '装修程度'
}
# 只保留df真实存在的列配置
safe_col_config = {k:v for k,v in all_config.items() if k in filtered_df.columns}

st.dataframe(
    filtered_df,
    use_container_width=True,
    column_config=safe_col_config
)

st.caption("💡 提示：使用左侧侧边栏筛选数据，所有图表和模型将自动更新。")
