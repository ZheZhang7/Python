import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
import seaborn as sns;


# calendar数据集分析
calendar = pd.read_csv('madrid-airbnb-data/calendar.csv');
# print(calendar.head())

# 修改其中price的类型
calendar['price'] = calendar['price'].str.replace(r'[$,]', "", regex = True).astype(np.float32);
calendar['adjusted_price'] = calendar['adjusted_price'].str.replace(r'[$,]', "", regex = True).astype(np.float32);

# 修改date的类型（date）
calendar['date'] = pd.to_datetime(calendar['date']);
# print(calendar['date'].head())

# 星期从0开始表示

# 添加星期几 和 月份的属性
calendar['weekday'] = calendar['date'].dt.weekday;
calendar['month'] = calendar['date'].dt.month;
# print(calendar['month'].head())

# 使用柱状图表示月份和价格关系
# 将每个月的价格分为一组，然后求平均值
# month_price = calendar.groupby('month')['price'].mean();
# 按照月份进行绘图
# sns.barplot(x = month_price.index, y = month_price.values);
# plt.show()

# 使用柱状图表示星期几和价格关系
weekday_price = calendar.groupby('weekday')['price'].mean();
sns.barplot(x = weekday_price.index, y = weekday_price.values);
# plt.show();

# 查看1000以下的价格分布直方图
sns.displot(calendar[calendar['price'] < 300]['price'], kde = True);
# plt.show();



# listings数据集分析
listings_detailed = pd.read_csv('madrid-airbnb-data/listings_detailed.csv');
#将所有列名称，转换成列表进行保存
# print(listings_detailed.columns.values.tolist());

# 修改其中price的类型
listings_detailed['price'] = listings_detailed['price'].str.replace(r'[$,]', "", regex = True).astype(np.float32);
listings_detailed['cleaning_fee'] = listings_detailed['cleaning_fee'].str.replace(r'[$,]', "", regex = True).astype(np.float32);
# print(listings_detailed['price'], listings_detailed['cleaning_fee'])
# listings_detailed['cleaning_fee']中存在nan数值，对nan用0进行填充
listings_detailed['cleaning_fee'].fillna(0, inplace=True);
# 添加最低消费字段
listings_detailed['minimum_cost'] = (listings_detailed['price'] + listings_detailed['cleaning_fee']) * listings_detailed['minimum_nights'];
# print(listings_detailed['minimum_cost'].head())

# 添加设施数量{Wifi,"Air conditioning",Kitchen,Elevator,Heat...
listings_detailed['n_amenities'] = listings_detailed['amenities'].str[1:-1].str.split(",").apply(len);
# 根据房间容纳人数，添加一个新的列，用来表示类型：Signal(1)、Couple(2)、Family(5)、Group(100)
listings_detailed['accommodates_type'] = pd.cut(listings_detailed['accommodates'], bins = [1,2,3,5,100], right=False, include_lowest=True, labels=['Signal', 'Couple', 'Family', 'Group']);
# 属于哪个社区
listings_detailed['neighbourhood_group_cleansed'];
# 房间评分
listings_detailed['review_scores_rating'].head();
# 整理需要的数据

listings_detailed_df = listings_detailed[['id','host_id', 'listing_url', 'room_type', 'neighbourhood_group_cleansed','price', 'cleaning_fee', 'n_amenities', 'amenities','accommodates_type', 'minimum_cost', 'minimum_nights']]

# 房间类型和社区对比分析
# 房间类型对比
# room_type_counts = listings_detailed_df['room_type'].value_counts();
# fig, axes = plt.subplots(1,2, figsize= (10,5));
# # 饼图
# axes[0].pie(room_type_counts.values, autopct = '%.2f%%', labels = room_type_counts.index);
# # 柱状图
# sns.barplot(x = room_type_counts.index, y = room_type_counts.values);
# # 让两个图别有重叠处，进行调整
# plt.tight_layout();
# plt.show();

# 社区对比
plt.figure(figsize=(20,10))
neighbourhood_counts = listings_detailed_df['neighbourhood_group_cleansed'].value_counts();
sns.barplot(y = neighbourhood_counts.index, x = neighbourhood_counts.values, orient = 'h');
# plt.show();

# 在某一个社区，各种房屋类型占比
# 按照neighbourhood_group_cleansed和room_type进行分组
# unstack 按照room_type不进行堆叠
# fillna(0) 使用0进行替换Nan
# 计算比例，row是series类型，series的/是每个value单独计算的
# 按照Entire home/apt进行排序
neighbour_room_type = listings_detailed_df.groupby(['neighbourhood_group_cleansed', 'room_type']) \
    .size() \
    .unstack('room_type') \
    .fillna(0) \
    .apply(lambda row: row / row.sum(), axis=1) \
    .sort_values('Entire home/apt', ascending=True);

# print(neighbour_room_type.head())

# 绘制条形图
# left进行起始位置确定
columns = neighbour_room_type.columns;
index = neighbour_room_type.index;
plt.figure(figsize=(20,10))
plt.barh(index, neighbour_room_type[columns[0]]);
left = neighbour_room_type[columns[0]];
plt.barh(index, neighbour_room_type[columns[1]], left = left);
left += neighbour_room_type[columns[1]];
plt.barh(index, neighbour_room_type[columns[2]], left = left);
left += neighbour_room_type[columns[2]];
plt.barh(index, neighbour_room_type[columns[3]], left = left);
plt.legend(columns);
# plt.show();

# 房东房源数量分析
host_number = listings_detailed_df.groupby('host_id').size();
sns.displot(data = host_number[host_number < 10], kde = True);
# plt.show();
# 查看比例 1,2,3,5+
# [1,2),[2,3),[3,4),5+
host_number_bins = pd.cut(host_number,bins=[1,2,3,5,100],right=False, include_lowest=True, labels=['1', '2', '3-4', '5+']).value_counts();
plt.pie(host_number_bins,autopct='%.2f%%', labels=host_number_bins.index);
# plt.show();



# 评论数量与时间分析,对reviews_detailed进行数据分析
# 获取数据集，将date转换为日期类型
reviews = pd.read_csv('madrid-airbnb-data/reviews_detailed.csv', parse_dates=['date']);
# 添加年月
reviews['year'] = reviews['date'].dt.year;
reviews['month'] = reviews['date'].dt.month;
# 按照年月对数据进行分组，查看哪一年/月的数据有多少
plt.figure(figsize=(20,10));
n_reviews_year = reviews.groupby('year').size();
sns.barplot(x = n_reviews_year.index, y = n_reviews_year.values);
plt.show();
n_reviews_month = reviews.groupby('month').size();
sns.barplot(x = n_reviews_month.index, y = n_reviews_month.values);
# plt.show();

# 评论数量与时间综合分析
year_month_reviews = reviews.groupby(['year', 'month']).size().unstack('month').fillna(0);
# 根据月份绘制（月份-评论）折线图
fig, ax = plt.subplots(figsize=(20,10));
for index in year_month_reviews.index:
    series = year_month_reviews.loc[index];
    sns.lineplot(x = series.index, y = series.values, ax = ax);
ax.legend(labels = year_month_reviews.index);
ax.grid();
# 显示横轴所有月份
_ = ax.set_xticks(list(range(1,13)))
plt.show();

# 房屋价格预测
# 使用listing数据集对房屋价格进行预测
# 提取价格有关的字段
from sklearn.preprocessing import StandardScaler;
ml_listings = listings_detailed[listings_detailed['price'] < 300][[
    'host_is_superhost',
    'host_identity_verified',
    'neighbourhood_group_cleansed',
    'latitude',
    'longitude',
    'property_type',
    'room_type',
    'accommodates',
    'bathrooms',
    'bedrooms',
    'cleaning_fee',
    'minimum_nights',
    'maximum_nights',
    'availability_90',
    'number_of_reviews',
    # 'review_scores_rating',
    'is_business_travel_ready',
    'n_amenities',
    'price'
]]

# 删除异常值
ml_listings.dropna(axis=0, inplace=True);

# 提取特征值和目标值
features = ml_listings.drop(columns=['price']);
targets = ml_listings['price'];

# 对于离散值进行one-hot编码, 统一特征值数据类型,进行目标值预测
disperse_columns = [
    'host_is_superhost',
    'host_identity_verified',
    'neighbourhood_group_cleansed',
    'property_type',
    'room_type',
    'is_business_travel_ready'
]
disperse_features = features[disperse_columns];
disperse_features = pd.get_dummies(disperse_features);
# 对连续值进行标准化,因为数值相差不大,对于结果影响不大
continuouse_features = features.drop(columns = disperse_columns);
scaler = StandardScaler();
continuouse_features = scaler.fit_transform(continuouse_features);

# 对特征值进行组合
feature_array = np.hstack([disperse_features, continuouse_features]);


from sklearn.model_selection import train_test_split;
from sklearn.linear_model import LinearRegression;
from sklearn.metrics import mean_absolute_error, r2_score;
# r2评分：r2的值越接近1越好，提升r2评分,让他拟合的更好,使用随机森林的回归
from sklearn.ensemble import RandomForestRegressor;

# 分割训练集和测试集
X_train, X_test,y_train,y_test = train_test_split(feature_array, targets,test_size=0.25);
regression = RandomForestRegressor();
# 预测
regression.fit(X_train, y_train);
y_predict = regression.predict(X_test);

# 查看平均误差和r2评分
print("平均误差:",mean_absolute_error(y_test, y_predict));
print("r2评分:" , r2_score(y_test, y_predict));

# 评论数量预测
ym_reviews = reviews.groupby(['year', 'month']).size().reset_index().rename(columns={0:'count'});
# 获取特征和目标值
features = ym_reviews[['year', 'month']];
targets = ym_reviews['count'];

# 分割训练集和测试集, 查看模型训练的泛化性
# X_train, X_test,y_train,y_test = train_test_split(features, targets,test_size=0.3);
# regression = RandomForestRegressor(n_estimators=100);
# regression.fit(X_train, y_train);
# y_predict = regression.predict(X_test);
# print("平均误差:",mean_absolute_error(y_test, y_predict));
# print("r2评分:" , r2_score(y_test, y_predict));

regression = RandomForestRegressor(n_estimators=100);
regression.fit(features,targets);

# 预测后结果
y_predict = regression.predict([
    [2019,10],
    [2019,11],
    [2019,12],
])

# 预测可视化
predict_reviews = pd.DataFrame([[2019, 10 + index, x] for index, x in enumerate(y_predict)], columns=['year', 'month', 'count']);
final_reviews = pd.concat([ym_reviews, predict_reviews]).reset_index();
years = final_reviews['year'].unique();
fig, ax = plt.subplots(figsize=(10,5));
for year in years:
    df = final_reviews[final_reviews['year'] == year];
    sns.lineplot(x = 'month', y = 'count', data = df);

ax.legend(labels = year_month_reviews.index);
ax.grid();
_ = ax.set_xticks(list(range(1,13)))
plt.show();