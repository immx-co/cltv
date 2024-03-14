import numpy as np
import pandas as pd
import os
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


train_df = pd.read_parquet("train_data.pqt")
test_df = pd.read_parquet("test_data.pqt")

cat_cols = [
    "channel_code", "city", "city_type",
    "okved", "segment", "start_cluster",
    "index_city_code", "ogrn_month", "ogrn_year",
]

train_df[cat_cols] = train_df[cat_cols].astype("category")
test_df[cat_cols] = test_df[cat_cols].astype("category")

X = train_df.drop(["id", "date", "end_cluster"], axis=1, )
y = train_df["end_cluster"]

x_train, x_val, y_train, y_val = train_test_split(X, y,
                                                  test_size=0.1,
                                                  random_state=42)

for i in x_train.columns:
    if x_train[i].dtype == 'float64':
        x_train[i] = x_train[i].fillna(x_train[i].mean())

drop_columns_to = ['city', 'city_type', 'index_city_code', 'ogrn_month', 'ogrn_year', 'start_cluster'] # 0.81655
# drop_columns_to = ['city', 'city_type', 'index_city_code', 'ogrn_year', 'start_cluster'] # 0.812
# drop_columns_to = ['city', 'city_type', 'index_city_code', 'ogrn_month', 'ogrn_year', 'start_cluster', 'ft_registration_date', 'max_founderpres', 'min_founderpres', 'ogrn_days_end_month', 'ogrn_days_end_quarter']

# for column in x_train.columns:
#     if 'cnt' in column and 'oper' in column and '1m' in column:
#         drop_columns_to.append(column)

x_train.drop(drop_columns_to, axis=1, inplace=True)
x_val.drop(drop_columns_to, axis=1, inplace=True)

# -------------------------------------------------------------
# from random import choice, seed  # noqa: E402

# seed(42)

# for i in x_train.columns:
#     if x_train[i].dtype != 'float64' and x_train[i].name != 'start_cluster':
#         col_unique = [i for i in x_train[i].unique() if type(i) != float]  # noqa: E721
#         for idx_j, j in enumerate(x_train[i]):
#             if pd.isnull(j):
#                 try:
#                     x_train[i].iloc[idx_j] = x_train[i].iloc[idx_j - 1]
#                 except:  # noqa: E722
#                     x_train[i].iloc[idx_j] = choice(col_unique)

# -------------------------------------------------------------
        
# init_x_train = x_train[:len(x_train) // 2]
# init_y_train = y_train[:len(y_train) // 2]
# init_x_val = x_val[:len(x_val) // 2]
# init_y_val = y_val[:len(y_val) // 2]

# cont_x_train = x_train[len(x_train) // 2:]
# cont_y_train = y_train[len(y_train) // 2:]
# cont_x_val = x_val[len(x_val) // 2:]
# cont_y_val = y_val[len(y_val) // 2:]

model = LGBMClassifier(verbosity=-1, random_state=100, n_jobs=-1, boosting_type='dart', n_estimators=100)
model.fit(x_train, y_train)

def weighted_roc_auc(y_true, y_pred, labels, weights_dict):
    unnorm_weights = np.array([weights_dict[label] for label in labels])
    weights = unnorm_weights / unnorm_weights.sum()
    classes_roc_auc = roc_auc_score(y_true, y_pred, labels=labels,
                                    multi_class="ovr", average=None)
    return sum(weights * classes_roc_auc)

cluster_weights = pd.read_excel(os.path.join(r"C:\Users\immx\official\respositories\hakaton\content", r"cluster_weights.xlsx")).set_index("cluster")
weights_dict = cluster_weights["unnorm_weight"].to_dict()

y_pred_proba = model.predict_proba(x_val)

res_roc_auc = weighted_roc_auc(y_val, y_pred_proba, model.classes_, weights_dict)

print('initial:', res_roc_auc)

# cont_model = LGBMClassifier(verbosity=-1, random_state=100, n_jobs=-1, boosting_type='dart', n_estimators=100)
# cont_model.fit(cont_x_train, cont_y_train, init_model=model.booster_)

# y_pred_proba_cont = cont_model.predict_proba(x_val)
# res_roc_auc_cont = weighted_roc_auc(y_val, y_pred_proba_cont, cont_model.classes_, weights_dict)

# print('cont:', res_roc_auc_cont)


test_df.pivot(index="id", columns="date", values="start_cluster").head(3)


test_df["start_cluster"] = train_df["start_cluster"].mode()[0]
test_df["start_cluster"] = test_df["start_cluster"].astype("category")

sample_submission_df = pd.read_csv("content/sample_submission.csv")

last_m_test_df = test_df[test_df["date"] == "month_6"]
last_m_test_df = last_m_test_df.drop(["id", "date"], axis=1)
last_m_test_df.drop(drop_columns_to, axis=1, inplace=True)

test_pred_proba = model.predict_proba(last_m_test_df)
test_pred_proba_df = pd.DataFrame(test_pred_proba, columns=model.classes_)
sorted_classes = sorted(test_pred_proba_df.columns.to_list())
test_pred_proba_df = test_pred_proba_df[sorted_classes]

sample_submission_df[sorted_classes] = test_pred_proba_df
sample_submission_df.to_csv("submission.csv", index=False)


