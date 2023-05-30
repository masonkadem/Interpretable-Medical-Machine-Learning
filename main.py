import pandas as pd
import numpy as np
import os
import warnings
from collections import Counter
import xgboost as xgb
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from xgboost import plot_tree
from sklearn import tree
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV,
    KFold,
    RepeatedStratifiedKFold,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    r2_score,
    mean_absolute_error,
    roc_curve,
    auc,
    average_precision_score,
    mean_absolute_percentage_error,
    mean_squared_error,
)


def col_miss(train_df):
    col_missing_df = train_df.isnull().sum(axis=0).reset_index()
    col_missing_df.columns = ["col", "missing_count"]
    col_missing_df = col_missing_df.sort_values(by="missing_count")
    return col_missing_df


def get_better_variable_names(X):
    X.rename(
        columns={
        },
        inplace=True
    )
    return X


def read_data_Exp1(remove_collinear=False, dropnan=False, selected_features=None):
    # load data
    data = pd.read_csv("./data/raw/ADNIMERGE.csv", na_values=["-1", "-4"])
    target = "DX"

    # index conditions
    idx = []
    for index, row in data.iterrows():
        if (
            (row["COLPROT"] == "ADNI3")
            and (row["VISCODE"] == "bl")
            and ((row["DX"] == "CN") or (row["DX"] == "Dementia"))
        ):
            idx.append(index)
    X = pd.DataFrame(data.iloc[idx], columns=data.columns)

    # Create new processed dataframe
    X = X.reset_index(drop=True)  # reset index
    # select variables with bl in name
    X = X.iloc[:, X.columns.str.contains("bl|BL|APOE4|PT|AGE|ABETA|PTAU|DX")]

    col_miss_data = col_miss(X)
    col_miss_data["Missing_part"] = col_miss_data["missing_count"] / len(X)
    sel_cols = col_miss_data[col_miss_data["Missing_part"] <= 0.25]["col"]

    X_sel = X[sel_cols].copy()
    X_sel2 = X_sel.drop(
        [
            "EXAMDATE_bl",
            "Years_bl",
            "Month_bl",
            "PTID",
            "FAQ_bl",
            "mPACCdigit_bl",
            "mPACCtrailsB_bl",
            "EcogSPTotal",
            "CDRSB_bl",
            "IMAGEUID_bl",
            "FSVERSION_bl",
            "IMAGEUID",
            "DX_bl",
        ],
        axis=1,
    )
    X = pd.DataFrame()
    X = X_sel2
    X = get_better_variable_names(X)

    print(
        "Checking total participants with target score:",
        X[target].notnull().values.sum(),
        "/",
        len(X),
    )
    X = X[X[target].notnull()].reset_index(drop=True)

    print("original target %s" % Counter(X[target]))
    X[target] = X[target].map({"CN": 0, "Dementia": 1})
    X["Gender"] = X["Gender"].map({"Female": 0, "Male": 1})
    X["Ethnicity"] = X["Ethnicity"].map({"Not Hisp/Latino": 0, "Hisp/Latino": 1})
    X["Race"] = X["Race"].map(
        {"White": 0, "Asian": 0, "Unknown": np.nan, "Am Indian/Alaskan": np.nan, "More than one": np.nan, "Black": 1}
    )
    X["Marital Status"] = X["Marital Status"].map(
        {"Never married": 0, "Married": 1, "Divorced": 0, "Widowed": 0, "Unknown": np.nan}
    )

    if remove_collinear == True:
        X = remove_collinear_features(X, "DX", 0.95, verbose=True)

    if dropnan == True:
        X = X.dropna(subset=selected_features, how="any")
        X = X.fillna(-1)

    cols = list(X.columns)
    cols.remove(target)
    cols.append(target)

    x_col = cols[:-1]
    y_col = cols[-1]
    X_data = X[x_col]
    y_data = X[y_col]

    return X, X_data, y_data, x_col, y_col, cat_var, num_var


def read_data_exp2(remove_collinear=False, dropnan=False, selected_features=None, ADNI="ADNI1"):
    adnimerge = pd.read_csv("./data/raw/ADNIMERGE.csv", na_values=["-1", "-4"])

    conv = adnimerge.loc[
        (adnimerge["VISCODE"] == "bl")
        & (adnimerge["DX_bl"].isin(["MCI", "EMCI", "LMCI", ]))
        & (
            adnimerge["RID"].isin(
                adnimerge.loc[
                    (
                        adnimerge["VISCODE"].isin(
                            ["m03", "m06", "m12", "m18", "m24", "m30", "m36"]
                        )
                    )
                    & (adnimerge["DX"].isin(["AD", "Dementia"]))
                ]["RID"].unique()
            )
        )
    ]
    conv["Conversion"] = 1

    stable = adnimerge.loc[
        (adnimerge["VISCODE"] == "m36")
        & (adnimerge["DX_bl"].isin(["MCI", "EMCI", "LMCI"]))
        & (~adnimerge["RID"].isin(conv["RID"]))
    ]
    stable["Conversion"] = 0

    X = pd.concat([stable, conv])

    target = "Conversion"

    X = X.reset_index(drop=True)
    X = X.iloc[:, X.columns.str.contains("bl|BL|APOE4|PT|AGE|ABETA|PTAU|Conversion")]

    X["ABETA_bl"] = X["ABETA_bl"].astype(str)
    X["ABETA_bl"] = X["ABETA_bl"].str.replace(">", "")

    col_miss_data = col_miss(X)
    col_miss_data["Missing_part"] = col_miss_data["missing_count"] / len(X)
    sel_cols = col_miss_data[col_miss_data["Missing_part"] <= 0.25]["col"]

    X_sel = X[sel_cols].copy()

    X_sel2 = X_sel.drop(
        [
            "EXAMDATE_bl",
            "Years_bl",
            "Month_bl",
            "PTID",
            "FAQ_bl",
            "mPACCdigit_bl",
            "mPACCtrailsB_bl",
            "CDRSB_bl",
            "IMAGEUID_bl",
            "FSVERSION_bl",
            "DX_bl",
        ],
        axis=1,
    )

    X_sel2.drop("FLDSTRENG_bl", axis=1, inplace=True, errors="ignore")
    X_sel2.drop("ABETA", axis=1, inplace=True, errors="ignore")
    X_sel2.drop("FLDSTRENG_bl", axis=1, inplace=True, errors="ignore")
    X_sel2.drop("IMAGEUID", axis=1, inplace=True, errors="ignore")
    X_sel2.drop("FDG_bl", axis=1, inplace=True, errors="ignore")

    X = pd.DataFrame()
    X = X_sel2

    X["Gender"] = X["Gender"].map({"Female": 0, "Male": 1})
    X["Ethnicity"] = X["Ethnicity"].map({"Not Hisp/Latino": 0, "Hisp/Latino": 1})
    X["Race"] = X["Race"].map(
        {"White": 0, "Asian": 0, "Unknown": np.nan, "Am Indian/Alaskan": np.nan, "More than one": np.nan, "Black": 1}
    )
    X["Marital Status"] = X["Marital Status"].map(
        {"Never married": 0, "Married": 1, "Divorced": 0, "Widowed": 0, "Unknown": np.nan}
    )

    print("original target %s" % Counter(X[target]))

    if remove_collinear == True:
        X = remove_collinear_features(X, "Conversion", 0.90, verbose=True)

    if dropnan == True:
        X = X.dropna(how="any")

    if [col for col in X.columns if (X[col].nunique()) >= 2 and (X[col].nunique()) < 3]:
        cat_var = [col for col in X.columns if X[col].nunique() >= 2 and X[col].nunique() < 3]
    if [col for col in X.columns if (X[col].nunique()) > 7]:
        num_var = [col for col in X.columns if X[col].nunique() > 7]

    X[cat_var] = X[cat_var].astype("category")
    X[num_var] = X[num_var].astype(float)

    cat_var.remove("Conversion")

    if dropnan == True:
        X = X.dropna(how="any")

    cols = list(X.columns)
    cols.remove(target)
    cols.append(target)

    x_col = cols[:-1]
    y_col = cols[-1]
    X_data = X[x_col]
    y_data = X[y_col]

    return X, X_data, y_data, x_col, y_col, cat_var, num_var


def feature_selection_xgboost(X, y, x_col, y_col, model=None):
    feature_importance = pd.DataFrame()
    feature_importance["feature"] = x_col
    feature_importance["importance"] = 0

    for i in range(100):
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
        model.fit(x_train, y_train)
        feature_importance["importance"] = feature_importance["importance"] + model.feature_importances_ / 100

    feature_importance = feature_importance.sort_values(axis=0, ascending=False, by="importance")
    print("Top features:")
    print(feature_importance.head(10))
    indices = np.argsort(feature_importance["importance"].values)[::-1]
    Top_f = 10
    Top_f_indices = indices[:Top_f]
    feature_importance_cols = feature_importance["feature"].values[:10]

    plt.subplots(figsize=(14, 10))
    g = sns.barplot(
        y=feature_importance.iloc[:Top_f]["feature"].values[Top_f_indices],
        x=feature_importance.iloc[:Top_f]["importance"].values[Top_f_indices],
        orient="h",
        palette="colorblind",
    )
    g.set_xlabel("XGBoost feature importance", fontsize=18)
    g.set_ylabel("Clinical features", fontsize=18)
    g.tick_params(labelsize=14)
    sns.despine()
    plt.savefig("Figures/feature_importances.png")
    plt.show()

    sel_feat = 1
    val_score_old = 0
    val_score_new = 0
    while val_score_new >= val_score_old:
        val_score_old = val_score_new
        selected_features = feature_importance_cols[:sel_feat]
        print("selected feature(s):", selected_features)
        X_selected_features = X[selected_features]

        print("100-round fivefold stratified cross-validation:")
        acc_train, acc_val, acc_train_std, acc_val_std = SKFlabels(X_selected_features, y, model=model)
        print("Training: %.3f ± %.3f; Validation: %.3f ± %.3f" % (acc_train, acc_train_std, acc_val, acc_val_std))

        val_score_new = acc_val
        sel_feat += 1

    print("Features selected:", selected_features[:-1])

    return list(selected_features[:-1]), list(feature_importance_cols)


def feature_selection_greedy(X, y, xgb_model, score, CV, k=15):
    from sklearn.feature_selection import SequentialFeatureSelector
    from sklearn.model_selection import RepeatedKFold
    from mlxtend.feature_selection import SequentialFeatureSelector as SFS
    from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

    sfs1 = SFS(
        xgb_model,
        k_features=k,
        forward=True,
        floating=False,
        verbose=2,
        scoring=score,
        cv=CV,
        n_jobs=-1,
    )

    sfs1 = sfs1.fit(X, y)

    results = pd.DataFrame.from_dict(sfs1.get_metric_dict()).T

    print(sfs1.subsets_)

    sfs1.k_score_

    fig1 = plot_sfs(sfs1.get_metric_dict(), kind="std_dev")

    return results, sfs1.subsets_


def SKFlabels(x, y, model=None, score_type='auc'):
    # 100-round K fold CV
    acc_v = []
    acc_t = []
    # K folds * 100
    for i in range(100):
        # random folds
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
        for train_idx, test_idx in skf.split(x, y):

            x_train = x.iloc[train_idx]
            y_train = y.iloc[train_idx]
            x_test = x.iloc[test_idx]
            y_test = y.iloc[test_idx]

            # fit model

            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            train_pred = model.predict(x_train)
            # using predicted label value instead of predicted probability to find the AUC to better distinguish features
            if score_type == 'auc':
                acc_v.append(roc_auc_score(y_test, pred))
                acc_t.append(roc_auc_score(y_train, train_pred))
            if score_type == 'f1':
                acc_v.append(f1_score(y_test, pred))
                acc_t.append(f1_score(y_train, train_pred))
            if score_type == 'recall':
                acc_v.append(recall_score(y_test, pred, pos_label=1))
                acc_t.append(recall_score(y_train, train_pred, pos_label=1))
            if score_type == 'aucpr':
                acc_v.append(average_precision_score(y_test, pred))
                acc_t.append(average_precision_score(y_train, train_pred))

    # return average
    return [np.mean(acc_t), np.mean(acc_v), np.std(acc_t), np.std(acc_v)]


def SKF(x, y, model=None, score_type='auc'):
    acc_v = []
    acc_t = []

    for i in range(100):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
        for train_idx, test_idx in skf.split(x, y):

            x_train = x.iloc[train_idx]
            y_train = y.iloc[train_idx]
            x_test = x.iloc[test_idx]
            y_test = y.iloc[test_idx]

            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            train_pred = model.predict(x_train)

            pred_proba = model.predict_proba(x_test)[:, 1]
            train_pred_proba = model.predict_proba(x_train)[:, 1]

            if score_type == 'auc':
                acc_v.append(roc_auc_score(y_test, pred_proba))
                acc_t.append(roc_auc_score(y_train, train_pred_proba))
            if score_type == 'f1':
                acc_v.append(f1_score(y_test, pred))
                acc_t.append(f1_score(y_train, train_pred))
            if score_type == 'recall':
                acc_v.append(recall_score(y_test, pred, pos_label=1))
                acc_t.append(recall_score(y_train, train_pred, pos_label=1))
            if score_type == 'aucpr':
                acc_v.append(average_precision_score(y_test, pred))
                acc_t.append(average_precision_score(y_train, train_pred))

    return [np.mean(acc_t), np.mean(acc_v), np.std(acc_t), np.std(acc_v)]


def get_tableone(data, cat_var, num_var):
    groupby = ['Conversion']
    columns = list(data.columns)
    nonnormal = list(data[num_var].columns)
    mytable = TableOne(data, columns=columns, groupby=groupby, categorical=cat_var, nonnormal=nonnormal, pval=True)
    mytable.to_latex('./data/tableone_EXPTEST_latex.tex')
    mytable.to_excel('./data/tableone_EXPTEST_excel.xlsx')


def plot_feature_distribution(df, label1, label2, features):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(10, 10, figsize=(18, 22))
    t0 = df.loc[df['DX_bl'] == 0]
    t1 = df.loc[df['DX_bl'] == 1]
    # features = X_all_features.columns.values

    for feature in features:
        i += 1
        plt.subplot(6, 6, i)
        sns.distplot(t0[feature], hist=False, label=label1)
        sns.distplot(t1[feature], hist=False, label=label2)
        plt.xlabel(feature, fontsize=12)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)
        plt.tick_params(axis='y', which='major', labelsize=6)
    plt.legend()
    plt.show()


def checknorm(df):
    from scipy import stats
    alpha = 0.05
    nonnormal = []
    normal = []

    df = df.dropna(how='any')
    for i in df.columns:
        print([i])
        a, b = stats.kstest(df[[i]], 'norm')
        print("stats", a, "p-value", b)
        if b < alpha:
            nonnormal.append([i])
            print("The null hypothesis can be rejected")
        else:
            normal.append([i])
            print("The null hypothesis cannot be rejected")

    print("nonnormal:", nonnormal)
    print("normal:", normal)


def remove_collinear_features(df_model, target_var, threshold, verbose):
    '''
    '''
    print("removing collinear features > ", threshold)
    # Calculate the correlation matrix
    corr_matrix = df_model.drop(target_var, 1).corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []
    dropped_feature = ""

    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i + 1):
            item = corr_matrix.iloc[j:(j + 1), (i + 1):(i + 2)]
            col = item.columns
            row = item.index
            val = abs(item.values)

            # If correlation exceeds the threshold
            if val >= threshold:
                # Print the correlated features and the correlation value
                if verbose:
                    print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                col_value_corr = df_model[col.values[0]].corr(df_model[target_var])
                row_value_corr = df_model[row.values[0]].corr(df_model[target_var])
                if verbose:
                    print("{}: {}".format(col.values[0], np.round(col_value_corr, 3)))
                    print("{}: {}".format(row.values[0], np.round(row_value_corr, 3)))
                if col_value_corr < row_value_corr:
                    drop_cols.append(col.values[0])
                    dropped_feature = "dropped: " + col.values[0]
                else:
                    drop_cols.append(row.values[0])
                    dropped_feature = "dropped: " + row.values[0]
                if verbose:
                    print(dropped_feature)
                    print("-----------------------------------------------------------------------------")

    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    df_model = df_model.drop(columns=drops)

    print("dropped columns: ")
    print(list(drops))
    print(len(list(drops)))
    print("-----------------------------------------------------------------------------")
    print("used columns: ")
    print(df_model.columns.tolist())
    print(len(df_model.columns.tolist()))

    return df_model


def plot_roc(labels, predict_prob, Moodel_name_i, fig, labels_name, k):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, predict_prob)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    # plt.figure()
    # plt.rc('axes', linewidth=2)
    line_list = ['--', '-']
    # ax = fig.add_subplot(111)
    plt.title('ROC', fontsize=20)
    plt.plot(false_positive_rate, true_positive_rate, line_list[k % 2], linewidth=1 + (1 - k / 5),
             label=Moodel_name_i + ' ROC-AUC = %0.4f' % roc_auc)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel('TPR', fontsize=20)
    plt.xlabel('FPR', fontsize=20)
    labels_name.append(Moodel_name_i + 'AUC = %0.4f' % roc_auc)
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # for axis in ['bottom','left']:
    #     ax.spines[axis].set_linewidth(2)
    # ax.tick_params(width=2)
    # plt.tight_layout()
    # plt.show()
    return labels_name


def plot_decision_boundary(model, x_tr, y_tr):
    """
    :param model:  XGBoost
    :param x_tr:
    :param y_tr:
    :return: None
    """
    # x_ss = StandardScaler().fit_transform(x_tr)
    # x_2d = PCA(n_components=2).fit_transform(x_ss)

    coord1_min = x_tr[:, 0].min() - 1
    coord1_max = x_tr[:, 0].max() + 1
    coord2_min = x_tr[:, 1].min() - 1
    coord2_max = x_tr[:, 1].max() + 1

    coord1, coord2 = np.meshgrid(
        np.linspace(coord1_min, coord1_max, int((coord1_max - coord1_min) * 30)).reshape(-1, 1),
        np.linspace(coord2_min, coord2_max, int((coord2_max - coord2_min) * 30)).reshape(-1, 1),
    )
    coord = np.c_[coord1.ravel(), coord2.ravel()]

    category = model.predict(coord).reshape(coord1.shape)
    # prob = model.predict_proba(coord)[:, 1]
    # category = (prob > 0.99).astype(int).reshape(coord1.shape)

    dir_save = './decision_boundary'
    os.makedirs(dir_save, exist_ok=True)

    # Figure
    plt.close('all')
    plt.figure(figsize=(7, 7))
    custom_cmap = ListedColormap(['#EF9A9A', '#90CAF9'])
    plt.contourf(coord1, coord2, category, cmap=custom_cmap)
    plt.savefig(pjoin(dir_save, 'decision_boundary1.png'), bbox_inches='tight')
    plt.scatter(x_tr[y_tr == 0, 0], x_tr[y_tr == 0, 1], c='yellow', label='Survival', s=30, alpha=1, edgecolor='k')
    plt.scatter(x_tr[y_tr == 1, 0], x_tr[y_tr == 1, 1], c='palegreen', label='Death', s=30, alpha=1, edgecolor='k')
    plt.ylabel('Var1')
    plt.xlabel('Var2')
    plt.legend()
    # plt.savefig(pjoin(dir_save, 'decision_boundary2.png'), dpi=500, bbox_inches='tight')
    plt.show()


def plot_3D_fig(X_data, cols):
    X_data = X_data.dropna(subset=cols, how='all')
    col = 'Diagnosis'
    data_df_sel2_0 = X_data[X_data[col] == 0]
    data_df_sel2_1 = X_data[X_data[col] == 1]

    # fig = plt.figure(dpi=400,figsize=(10, 4))
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111, projection='3d')
    i = 2;
    j = 0;
    k = 1;  # 120 201
    ax.scatter(data_df_sel2_0[cols[i]], data_df_sel2_0[cols[j]], data_df_sel2_0[cols[k]], c=data_df_sel2_0[col],
               cmap='Blues_r', label='Survived', linewidth=0.5)
    ax.scatter(data_df_sel2_1[cols[i]], data_df_sel2_1[cols[j]], data_df_sel2_1[cols[k]], c=data_df_sel2_1[col],
               cmap='gist_rainbow_r', label='Death', marker='x', linewidth=0.5)

    cols_en = ['']
    ax.set_zlabel(cols_en[k])
    ax.set_ylabel(cols_en[j])
    ax.set_xlabel(cols_en[i])
    fig.legend(['Survival', 'Death'], loc='upper center')
    # plt.savefig('./picture_2class/3D_data_'+str(i)+str(j)+str(k)+'_v6.png')
    plt.show()


def plot_prcurve(labels, predict_prob, Moodel_name_i, fig, labels_name, k):
    precision, recall, thresholds = precision_recall_curve(labels, predict_prob)
    # false_positive_rate,true_positive_rate,thresholds=roc_curve(labels, predict_prob)
    pr_auc = auc(recall, precision)
    # plt.figure()
    line_list = ['--', '-']
    # ax = fig.add_subplot(111)
    plt.title('PR', fontsize=20)
    plt.plot(recall, precision, line_list[k % 2], linewidth=1 + (1 - k / 5),
             label=Moodel_name_i + ' PR-AUC = %0.4f' % pr_auc)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel('Precision', fontsize=20)
    plt.xlabel('Recall', fontsize=20)
    labels_name.append(Moodel_name_i + 'AUC = %0.4f' % pr_auc)
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # for axis in ['bottom','left']:
    #     ax.spines[axis].set_linewidth(2)
    # plt.show()
    return labels_name


def compare(selected_features, X, y, val=False, pr=False, st_model=None, xgb_model=None, loop=100, seed=None):
    if val == True and pr == False:
        print("Comparing rocauc (validation)")
    if val == True and pr == True:
        print("Comparing prcauc (validation)")
    if val == False and pr == True:
        print("Comparing prauc (training)")
    if val == False and pr == False:
        print("Comparing rocauc (training)")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=seed)

    # tree_clf = tree.DecisionTreeClassifier(random_state=1,max_depth=4)
    RF_clf = RandomForestClassifier(random_state=1, n_estimators=100, max_depth=4)
    LR_clf = linear_model.LogisticRegression(random_state=1, C=1, solver='lbfgs')
    # SVC_clf = SVC(random_state=1,C=1, probability=True)

    # st_model = xgb.XGBClassifier(enable_categorical= True, objective='binary:logistic', eval_metric = 'auc', tree_method='hist', n_estimators=1, random_state =1)

    # xgb_model = xgb.XGBClassifier(enable_categorical= True, objective='binary:logistic', eval_metric = 'auc', tree_method='hist', subsample=0.9, colsample_bytree=0.9, random_state =1)

    tree_clf = tree.DecisionTreeClassifier(random_state=1, max_depth=4)

    LR_clf = linear_model.LogisticRegression(random_state=1, C=1, solver='lbfgs')
    # SVC_clf = SVC(random_state=1,C=1, probability=True)

    fig = plt.figure(dpi=400, figsize=(8, 8))

    # loop = 20

    i = 0
    labels_names = []
    Moodel_name = ['XGBoost[all features]',

                   'Random forest[all features]',
                   'Decision tree[all features]',
                   'Logistic regression[all features]'

                   ]

    # [xgb_n_clf,tree_clf,RF_clf1,LR_clf]
    for model in [xgb_model, RF_clf, tree_clf, LR_clf]:
        print('Model:' + Moodel_name[i])

        acc_train, acc_val, acc_train_std, acc_val_std = SKF(X, y, model, score_type='auc')
        print("Training: %.3f ± %.3f; Validation: %.3f ± %.3f" % (
        acc_train, acc_train_std, acc_val, acc_val_std))

        model.fit(X_train, y_train)
        pred_train_probe = model.predict_proba(X_train)[:, 1]
        pred_val_probe = model.predict_proba(X_val)[:, 1]
        # plot_roc(y_val, pred_val_probe,Moodel_name[i],fig,labels_names,i) # for validation roc
        # plot_roc(y_train, pred_train_probe,Moodel_name[i],fig,labels_names,i) ## for training roc
        # plot_prcurve(y_train, pred_train_probe,Moodel_name[i],fig,labels_names,i)# train # not used
        if val == True and pr == False:
            plot_roc(y_val, pred_val_probe, Moodel_name[i], fig, labels_names, i)
        if val == True and pr == True:
            plot_prcurve(y_val, pred_val_probe, Moodel_name[i], fig, labels_names, i)
        if val == False and pr == True:
            plot_prcurve(y_train, pred_train_probe, Moodel_name[i], fig, labels_names, i)
        if val == False and pr == False:
            plot_roc(y_train, pred_train_probe, Moodel_name[i], fig, labels_names, i)

        if val == True:
            print('ROC-AUC score:', roc_auc_score(y_val, pred_val_probe))
        if val == False:
            print('ROC-AUC score:', roc_auc_score(y_train, pred_train_probe))

        i = i + 1

    ## Comparison of three feature single tree model
    X_sel = X[selected_features] #
    # ## same division as before
    X_train, X_val, y_train, y_val = train_test_split(X_sel, y, test_size=0.3, random_state=seed)


    # st_model2 = xgb.XGBClassifier(enable_categorical= True, objective='binary:logistic', eval_metric = 'auc', tree_method='hist', n_estimators=1, random_state =1)
    # xgb_model2 = xgb.XGBClassifier(enable_categorical= True, objective='binary:logistic', eval_metric = 'auc', tree_method='hist', subsample=0.9, colsample_bytree=0.9, random_state =1)
    RF_clf2 = RandomForestClassifier(random_state=seed, n_estimators=1, max_depth=3, )
    tree_clf2 = tree.DecisionTreeClassifier(random_state=seed, max_depth=3)
    LR_clf2 = linear_model.LogisticRegression(random_state=seed, C=1, solver='lbfgs')

    #i = 0
    Moodel_name = ['XGBoost[selected_features]',
                   'Int-XGBoost[selected_features]',
                   'Random forest[selected_features]',
                   'Decision tree[selected_features]',
                   'Logistic regression[selected_features]',
                   ]

    for model in [xgb_model, st_model, RF_clf2, tree_clf2, LR_clf2]:
        print('Model' + Moodel_name[i - len(Moodel_name)])
        # f1
        # acc_train, acc_val, acc_train_std, acc_val_std = StratifiedKFold_func(x_np_sel.values, y_np.values,Num_iter,model, score_type ='f1')
        # #print('F1-score of Train:%.6f with std:%.4f \nF1-score of Validation:%.4f with std:%.6f '%(acc_train,acc_train_std,acc_val,acc_val_std))
        # #auc
        acc_train, acc_val, acc_train_std, acc_val_std = SKF(X_sel, y, model, score_type='auc')
        print("Training: %.3f ± %.3f; Validation: %.3f ± %.3f" % (
        acc_train, acc_train_std, acc_val, acc_val_std))

        model.fit(X_train, y_train)
        pred_train_probe = model.predict_proba(X_train)[:, 1]  # train
        pred_val_probe = model.predict_proba(X_val)[:, 1]  # test
        # plot_roc(y_val, pred_val_probe,Moodel_name[i-5],fig,labels_names,i) # test
        # plot_roc(y_train, pred_train_probe,Moodel_name[i-5],fig,labels_names,i)# train
        # plot_prcurve(y_val, pred_val_probe,Moodel_name[i-len(Moodel_name)],fig,labels_names,i)
        if val == True and pr == False:
            plot_roc(y_val, pred_val_probe, Moodel_name[i - len(Moodel_name)], fig, labels_names, i)
            SS = 'validroc'
        if val == True and pr == True:
            plot_prcurve(y_val, pred_val_probe, Moodel_name[i - len(Moodel_name)], fig, labels_names, i)
            SS = 'validpr'
        if val == False and pr == True:
            plot_prcurve(y_train, pred_train_probe, Moodel_name[i - len(Moodel_name)], fig, labels_names, i)
            SS = 'trainpr'
        if val == False and pr == False:
            plot_roc(y_train, pred_train_probe, Moodel_name[i - len(Moodel_name)], fig, labels_names, i)
            SS = 'trainroc'
        if val == True:
            print('ROC-AUC score:', roc_auc_score(y_val, pred_val_probe))
        if val == False:
            print('ROC-AUC score:', roc_auc_score(y_train, pred_train_probe))
        i = i + 1
    if pr == False:
        plt.plot([0, 1], [0, 1], 'r--')
    if pr == True:
        plt.axhline(y=0.5, linestyle='--', color='black')
    plt.legend(loc='lower right', fontsize=12)
    plt.savefig('Figures/' + SS + '.png')
    plt.show()


    
def single_tree(X, y, model, selected_features, test_set=False, seed=seed):
    print('single_tree:\n')
    import shap
    import string
    import seaborn as sns
    import matplotlib.pyplot as plt

    X = X[selected_features]
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.3, random_state=seed)

    model.fit(X_train, y_train)

    # train
    pred_train = model.predict(X_train)
    show_confusion_matrix(y_train, pred_train, 'train')
    print(classification_report(y_train, pred_train))

    # validation
    pred_val = model.predict(X_val)
    show_confusion_matrix(y_val, pred_val, 'valid')
    print(classification_report(y_val, pred_val))

    if test_set:
        data, X_test, y_test, x_col, y_col, cat_var, num_var = read_data_convert(remove_collinear=False, dropnan=False,
                                                                                 ADNI='ADNI2')

        X_test = X_test[selected_features]

        pred_test = model.predict(X_test)
        show_confusion_matrix(y_test, pred_test, 'test')
        print(classification_report(y_test, pred_test))

    plt.figure(dpi=300, figsize=(150, 100))
    plot_tree(model)
    plt.savefig('Figures/plottree.png')
    plt.show()

    booster = model.get_booster()
    m = xgb.DMatrix(X, enable_categorical=True)
    SHAP = booster.predict(m, pred_contribs=True)
    margin = booster.predict(m, output_margin=True)

    shap.summary_plot(SHAP[:, :-1], X, plot_type="dot", max_display=len(selected_features), auto_size_plot=True,
                      show=False)
    plt.tight_layout()
    plt.show()
    vals = np.abs(SHAP[:, :-1]).mean(0)
    feature_importance = pd.DataFrame(list(zip(X.columns, vals)), columns=['col_name', 'feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
    feature_importance.head()
    feature_importance_cols = feature_importance['col_name'].values[:len(selected_features)]

    row = 3
    col = 2

    fig, axs = plt.subplots(nrows=row, ncols=col, figsize=(24, 24))
    axs = axs.ravel()

    for i, ax in enumerate(axs):
        ax.text(-0.1, 1.1, string.ascii_uppercase[i], fontsize=20, transform=ax.transAxes, fontweight='bold', va='top',
                ha='right')
        shap.dependence_plot(selected_features[i], SHAP[:, :-1], X[selected_features], ax=ax, show=False)

        ax.axhline(y=0, color='b', linestyle='--')
        plt.tight_layout()

    fig, axs = plt.subplots(nrows=row, ncols=col, figsize=(24, 24))
    axs = axs.ravel()

    for i, ax in enumerate(axs):
        ax.text(-0.1, 1.1, string.ascii_uppercase[i], fontsize=20, transform=ax.transAxes, fontweight='bold', va='top',
                ha='right')
        shap.dependence_plot(selected_features[i], SHAP[:, :-1], X[selected_features], ax=ax, show=False)

        ax.axhline(y=0, color='b', linestyle='--')
        plt.tight_layout()


def show_confusion_matrix(validations, predictions, name, save=True):
    LABELS = ['CN', 'ADD']
    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(4.5, 3))
    sns.heatmap(matrix,
                cmap='Blues',
                linecolor='white',
                linewidths=2,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    if save:
        plt.savefig('Figures/' + name + '_xgb_fs.png')
    plt.show()


def plot_distrib_violin(data, x_col, target=None, nrows=7, ncols=5):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(30, 30))
    sns.set_context(context='paper', font_scale=0.5)

    ax = ax.ravel()
    data[''] = 0
    for i, feature in enumerate(x_col):
        sns.violinplot(x='', y=feature, hue=target, data=data, split=True, ax=ax[i])
        if i < len(x_col) - 1:
            ax[i].get_legend().remove()
    plt.tight_layout()


def benchmark(x, y, repeat=100, selected_features=None):
    cv = pd.DataFrame()
    cv['model'] = ['NearestCentroid',
                   'SGDClassifier',
                   'LinearSVC',
                   'LogisticRegression',
                   'BernoulliNB',
                   'GaussianNB',
                   'KNeighborsClassifier',
                   'ExtraTreesClassifier',
                   'Perceptron',
                   'LinearDiscriminantAnalysis',
                   'BaggingClassifier',
                   'SVC',
                   'RandomForestClassifier',
                   'AdaBoostClassifier',
                   'PassiveAggressiveClassifier',
                   'RidgeClassifierCV',
                   'XGBClassifier',
                   'DecisionTreeClassifier',
                   'RidgeClassifier',
                   'ExtraTreeClassifier',
                   'LGBMClassifier',
                   'CalibratedClassifierCV',
                   'LabelSpreading',
                   'LabelPropagation',
                   'QuadraticDiscriminantAnalysis',
                   'DummyClassifier']

    cv['score'] = 0

    if selected_features:
        x = x[selected_features]

    for i in range(repeat):
        X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.3, random_state=i)
        clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
        models, predictions = clf.fit(X_train, X_test, y_train, y_test)
        mod2 = models['ROC AUC']
        mod2 = mod2.reset_index(drop=True)

        cv['score'] = cv['score'] + mod2 / repeat

    cv = cv.sort_values(axis=0, ascending=False, by='score')
    print('Top models:')
    print(cv.head(20))

    return cv.head(20)


def xgb_grid_search(X, y, nfolds):
    param_grid = {'max_depth': [3, 4, 5, 6], 'min_child_weight': [4, 5, 6], 'learning_rate': [0.05, 0.1, 0.5],
                  'n_estimators': [20, 50, 100]}
    xgb_model = XGBClassifier()
    xgb_gscv = GridSearchCV(xgb_model, param_grid, cv=nfolds)
    xgb_gscv.fit(X, y)
    print(xgb_gscv.best_params_)
    print(xgb_gscv.best_estimator_)
    print(xgb_gscv.best_score_)


def tune(params, X, y, model, n_iter=50, scoring='roc_auc', grid=False):
    rskfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)
    if grid:
        model = GridSearchCV(model, param_grid=params, scoring=scoring, n_jobs=-1, cv=rskfold, refit=True, verbose=True)
    else:
        model = RandomizedSearchCV(model, param_distributions=params, scoring=scoring, n_jobs=-1, cv=rskfold,
                                   n_iter=n_iter, refit=True, verbose=True)
    model.fit(X, y)
    best_estimator = model.best_estimator_
    print('Best estimator:', best_estimator)
    print('Best paramters:', model.best_params_)
    print('Best score:', model.best_score_)

    return best_estimator, model.best_params_


if __name__ == '__main__':
    from utils_adni import *
    from sklearn.utils import shuffle
    from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
    from sklearn import metrics
    from sklearn import tree
    from sklearn.metrics import classification_report
    from xgboost import XGBClassifier
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    import pandas as pd

    seed = 1

    # MODELS
    model = XGBClassifier(enable_categorical=True, objective='binary:logistic', eval_metric='logloss',
                          tree_method='hist', subsample=0.9, colsample_bytree=0.9)

    st_model = XGBClassifier(enable_categorical=True, objective='binary:logistic', eval_metric='logloss',
                             tree_method='hist', n_estimators=1)

    # DATA - Exp1
    data, X_data, y_data, x_col, y_col, cat_var, num_var = read_data(remove_collinear=True, dropnan=False)
    # EDA
    plot_distrib_violin(data, num_var, target='Conversion', nrows=4, ncols=4)
    X_data, y_data = shuffle(X_data, y_data)

     # Feature selection method 1
    selected_features, top_10_features = feature_selection(X_data, y_data, x_col, y_col, model=model)

    # benchmarking
    compare(selected_features, X_data, y_data, st_model=st_model, xgb_model=model, val=True, pr=False, loop=5, seed=0)

    # INTERPRETABILITY
    single_tree(X_data, y_data, model=st_model, selected_features=selected_features, test_set=False, seed=1)

    # DATA - Exp2
    data, X_data, y_data, x_col, y_col, cat_var, num_var = read_data_convert(remove_collinear=True, dropnan=False)
    X_data, y_data = shuffle(X_data, y_data)
    # EDA
    plot_distrib_violin(data, num_var, target='Conversion', nrows=4, ncols=4)

     # Feature selection method 1
    selected_features, top_10_features = feature_selection(X_data, y_data, x_col, y_col, model=model)
    
    # Feature selection method 2
    CV = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)
    results, subsets = greedyfs(X_data, y_data, model, 'roc_auc', CV, k=15)

    # benchmarking
    compare(selected_features, X_data, y_data, st_model=st_model, xgb_model=model, val=True, pr=False, loop=5, seed=0)

    #INTPRETABILITY
    single_tree(X_data, y_data, model=st_model, selected_features=selected_features, test_set=False, seed=1)

 
