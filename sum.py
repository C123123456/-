import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
import warnings

warnings.filterwarnings('ignore')

# 配置中文字体和图表显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['figure.figsize'] = (5, 3)  # 设置默认图像大小

# 1. 数据加载与预处理 - 改进随机数据生成
def load_data(random_state=None):  # 添加随机状态参数
    """加载数据并进行预处理，支持真实数据和模拟数据"""
    try:
        train_data = pd.read_csv('titanic_train.csv')
        print("成功加载真实数据集")
    except:
        # 移除固定随机种子，使每次生成的数据不同
        if random_state is not None:
            np.random.seed(random_state)

        n = 891
        train_data = pd.DataFrame({
            'Survived': np.random.binomial(1, 0.384, n),
            'Pclass': np.random.choice([1, 2, 3], n, p=[0.2, 0.25, 0.55]),
            'Sex': np.random.choice(['male', 'female'], n, p=[0.62, 0.38]),
            'Age': np.random.normal(29.7, 14.5, n),
            'SibSp': np.random.randint(0, 5, n),
            'Parch': np.random.randint(0, 4, n),
            'Fare': np.random.lognormal(3, 1, n),
            'Embarked': np.random.choice(['S', 'C', 'Q'], n, p=[0.72, 0.2, 0.08])
        })
        train_data.loc[train_data['Age'] < 0, 'Age'] = np.nan
        train_data.loc[np.random.choice(n, int(n * 0.206)), 'Age'] = np.nan
        train_data.loc[np.random.choice(n, 2), 'Embarked'] = np.nan
        train_data.loc[np.random.choice(n, int(n * 0.0081)), 'Fare'] = np.nan
        print("使用模拟数据集进行演示")

    train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
    train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)
    train_data['Fare'].fillna(train_data['Fare'].median(), inplace=True)

    train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch']
    train_data['IsAlone'] = (train_data['FamilySize'] == 0).astype(int)

    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'IsAlone']
    X = train_data[features]
    y = train_data['Survived']

    return X, y, train_data


# 2. 数据探索可视化
def visualize_data_exploration(train_data):
    plt.figure()
    ax = sns.countplot(x='Survived', data=train_data)
    plt.xlabel('生存状态', fontsize=10)
    plt.ylabel('乘客数量', fontsize=10)
    plt.xticks([0, 1], ['未存活', '存活'], fontsize=8)
    plt.title('生存状态分布', fontsize=12)
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + 0.35, p.get_height() + 5), fontsize=7)
    plt.tight_layout()
    plt.show()

    plt.figure()
    ax = sns.countplot(x='Sex', hue='Survived', data=train_data)
    plt.xlabel('性别', fontsize=10)
    plt.ylabel('乘客数量', fontsize=10)
    plt.legend(['未存活', '存活'], fontsize=8)
    plt.title('性别与生存状态分布', fontsize=12)
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + 0.1, p.get_height() + 5), fontsize=7)
    plt.tight_layout()
    plt.show()

    plt.figure()
    ax = sns.countplot(x='Pclass', hue='Survived', data=train_data)
    plt.xlabel('舱位等级', fontsize=10)
    plt.ylabel('乘客数量', fontsize=10)
    plt.legend(['未存活', '存活'], fontsize=8)
    plt.title('舱位等级与生存状态分布', fontsize=12)
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + 0.1, p.get_height() + 5), fontsize=7)
    plt.tight_layout()
    plt.show()

    plt.figure()
    sns.histplot(train_data, x='Age', hue='Survived', kde=True, bins=20)
    plt.xlabel('年龄', fontsize=10)
    plt.ylabel('乘客数量', fontsize=10)
    plt.legend(['存活', '未存活'], fontsize=8)
    plt.title('年龄分布与生存状态关系', fontsize=12)
    plt.tight_layout()
    plt.show()

    plt.figure()
    sns.histplot(train_data, x='Fare', hue='Survived', kde=True, bins=20)
    plt.xlabel('票价', fontsize=10)
    plt.ylabel('乘客数量', fontsize=10)
    plt.legend(['存活', '未存活'], fontsize=8)
    plt.title('票价分布与生存状态关系', fontsize=12)
    plt.tight_layout()
    plt.show()

    plt.figure()
    train_data['FamilySizeCategory'] = pd.cut(
        train_data['FamilySize'], bins=[0, 1, 4, 10], labels=['独自出行', '小家庭', '大家庭']
    )
    ax = sns.countplot(x='FamilySizeCategory', hue='Survived', data=train_data)
    plt.xlabel('家庭大小', fontsize=10)
    plt.ylabel('乘客数量', fontsize=10)
    plt.legend(['未存活', '存活'], fontsize=8)
    plt.title('家庭大小与生存状态分布', fontsize=12)
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + 0.1, p.get_height() + 5), fontsize=7)
    plt.tight_layout()
    plt.show()

    print("数据探索可视化完成，共6张图表")


# 3. 模型构建与评估
def build_models(X, y):
    # 移除固定随机种子，使数据分割不同
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )
    print(f"数据划分：训练集 {len(X_train)} 条，测试集 {len(X_test)} 条")

    categorical_features = ['Pclass', 'Sex', 'Embarked']
    numeric_features = ['Age', 'Fare', 'FamilySize', 'IsAlone']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features),
            ('num', StandardScaler(), numeric_features)
        ])

    dt_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(max_depth=3))  # 移除固定随机种子
    ])
    dt_model = dt_pipeline.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    y_prob_dt = dt_model.predict_proba(X_test)[:, 1]

    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier())  # 移除固定随机种子
    ])

    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [5, 10, 20],
        'classifier__min_samples_split': [2, 5]
    }

    grid_search = GridSearchCV(rf_pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    rf_model = grid_search.best_estimator_
    y_pred_rf = rf_model.predict(X_test)
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

    metrics = {
        'dt': {
            'accuracy': accuracy_score(y_test, y_pred_dt),
            'precision': precision_score(y_test, y_pred_dt),
            'recall': recall_score(y_test, y_pred_dt),
            'f1': f1_score(y_test, y_pred_dt),
            'roc_auc': roc_auc_score(y_test, y_prob_dt),
            'confusion_matrix': confusion_matrix(y_test, y_pred_dt)
        },
        'rf': {
            'accuracy': accuracy_score(y_test, y_pred_rf),
            'precision': precision_score(y_test, y_pred_rf),
            'recall': recall_score(y_test, y_pred_rf),
            'f1': f1_score(y_test, y_pred_rf),
            'roc_auc': roc_auc_score(y_test, y_prob_rf),
            'confusion_matrix': confusion_matrix(y_test, y_pred_rf)
        }
    }

    print("\n--- 模型评估结果 ---")
    print("决策树:")
    for k, v in metrics['dt'].items():
        if k != 'confusion_matrix':
            print(f"  {k}: {v:.4f}")

    print("\n随机森林:")
    for k, v in metrics['rf'].items():
        if k != 'confusion_matrix':
            print(f"  {k}: {v:.4f}")

    return dt_model, rf_model, X_test, y_test, y_pred_dt, y_pred_rf, y_prob_dt, y_prob_rf, metrics, preprocessor


# 4. 性能对比可视化
def visualize_performance_comparison(metrics, y_test, y_prob_dt, y_prob_rf):
    plt.figure()
    metrics_names = ['准确率', '精确率', '召回率', 'F1分数', 'ROC-AUC']
    dt_values = [metrics['dt']['accuracy'], metrics['dt']['precision'],
                 metrics['dt']['recall'], metrics['dt']['f1'], metrics['dt']['roc_auc']]
    rf_values = [metrics['rf']['accuracy'], metrics['rf']['precision'],
                 metrics['rf']['recall'], metrics['rf']['f1'], metrics['rf']['roc_auc']]

    x = np.arange(len(metrics_names))
    width = 0.3

    plt.bar(x - width / 2, dt_values, width, label='决策树', color='#5DA5DA', alpha=0.8)
    plt.bar(x + width / 2, rf_values, width, label='随机森林', color='#FAA43A', alpha=0.8)

    plt.xlabel('评估指标', fontsize=10)
    plt.ylabel('分数', fontsize=10)
    plt.title('决策树与随机森林性能指标对比', fontsize=12)
    plt.xticks(x, metrics_names, fontsize=8)
    plt.legend(fontsize=8)

    for i, v in enumerate(dt_values):
        plt.text(i - width / 2, v + 0.01, f'{v:.4f}', ha='center', fontsize=7)
    for i, v in enumerate(rf_values):
        plt.text(i + width / 2, v + 0.01, f'{v:.4f}', ha='center', fontsize=7)

    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)

    fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

    plt.plot(fpr_dt, tpr_dt, label=f'决策树 (AUC = {metrics["dt"]["roc_auc"]:.4f})',
             color='#5DA5DA', lw=1.5)
    plt.plot(fpr_rf, tpr_rf, label=f'随机森林 (AUC = {metrics["rf"]["roc_auc"]:.4f})',
             color='#FAA43A', lw=1.5)
    plt.xlabel('假阳性率', fontsize=10)
    plt.ylabel('真阳性率', fontsize=10)
    plt.title('ROC曲线对比', fontsize=12)
    plt.legend(fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.subplot(1, 2, 1)
    cm_dt = metrics['dt']['confusion_matrix']
    sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 8})
    plt.xlabel('预测标签', fontsize=10)
    plt.ylabel('真实标签', fontsize=10)
    plt.title('决策树混淆矩阵', fontsize=12)

    plt.subplot(1, 2, 2)
    cm_rf = metrics['rf']['confusion_matrix']
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 8})
    plt.xlabel('预测标签', fontsize=10)
    plt.ylabel('真实标签', fontsize=10)
    plt.title('随机森林混淆矩阵', fontsize=12)

    plt.tight_layout()
    plt.show()

    print("模型性能对比可视化完成，共3张图表")


# 5. 特征重要性对比可视化
def visualize_feature_importance(dt_model, rf_model, preprocessor):
    onehot = preprocessor.named_transformers_['cat']
    cat_features = onehot.get_feature_names_out(['Pclass', 'Sex', 'Embarked'])
    all_features = list(cat_features) + ['Age', 'Fare', 'FamilySize', 'IsAlone']

    dt_importance = pd.DataFrame({
        '特征': all_features,
        '重要性': dt_model.named_steps['classifier'].feature_importances_
    }).sort_values('重要性', ascending=False)

    rf_importance = pd.DataFrame({
        '特征': all_features,
        '重要性': rf_model.named_steps['classifier'].feature_importances_
    }).sort_values('重要性', ascending=False)

    plt.figure()
    plt.subplot(1, 2, 1)
    sns.barplot(x='重要性', y='特征', data=dt_importance, color='#5DA5DA', alpha=0.8)
    plt.xlabel('重要性', fontsize=10)
    plt.ylabel('特征', fontsize=10)
    plt.title('决策树特征重要性排序', fontsize=12)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)

    plt.subplot(1, 2, 2)
    sns.barplot(x='重要性', y='特征', data=rf_importance, color='#FAA43A', alpha=0.8)
    plt.xlabel('重要性', fontsize=10)
    plt.ylabel('特征', fontsize=10)
    plt.title('随机森林特征重要性排序', fontsize=12)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)

    plt.tight_layout()
    plt.show()

    top_features = dt_importance.head(5)['特征'].tolist()
    dt_top = dt_importance[dt_importance['特征'].isin(top_features)]
    rf_top = rf_importance[rf_importance['特征'].isin(top_features)]

    plt.figure()
    plt.barh(top_features, dt_top['重要性'], 0.3, label='决策树', color='#5DA5DA', alpha=0.8)
    plt.barh([f + ' ' for f in top_features], rf_top['重要性'], 0.3, label='随机森林', color='#FAA43A', alpha=0.8)
    plt.xlabel('重要性', fontsize=10)
    plt.ylabel('特征', fontsize=10)
    plt.title('Top 5特征重要性对比', fontsize=12)
    plt.legend(fontsize=8)
    plt.yticks(fontsize=8)

    plt.tight_layout()
    plt.show()

    print("特征重要性可视化完成，共2张图表")


# 6. 决策树与随机森林结构可视化
def visualize_tree_structure(dt_model, rf_model, preprocessor):
    onehot = preprocessor.named_transformers_['cat']
    cat_features = onehot.get_feature_names_out(['Pclass', 'Sex', 'Embarked'])
    all_features = list(cat_features) + ['Age', 'Fare', 'FamilySize', 'IsAlone']

    plt.figure(figsize=(8, 5))  # 适当调整大小以适应决策树结构
    plot_tree(dt_model.named_steps['classifier'],
              feature_names=all_features,
              class_names=['未存活', '存活'],
              filled=True,
              rounded=True,
              proportion=True,
              impurity=False,
              fontsize=8)
    plt.title('三层决策树结构可视化', fontsize=7)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))  # 适当调整大小以适应随机森林中的决策树
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plot_tree(rf_model.named_steps['classifier'].estimators_[i],
                  feature_names=all_features,
                  class_names=['未存活', '存活'],
                  filled=True,
                  rounded=True,
                  proportion=True,
                  impurity=False,
                  fontsize=6,
                  max_depth=3)
        plt.title(f'随机森林 - 第{i + 1}棵决策树结构', fontsize=7)
    plt.tight_layout()
    plt.show()

    print("决策树结构可视化完成，共2张图表")


# 7. 预测结果对比可视化
def visualize_prediction_comparison(X_test, y_test, y_pred_dt, y_pred_rf, dt_model, rf_model):
    sample_indices = np.random.choice(len(X_test), min(50, len(X_test)), replace=False)
    X_sample = X_test.iloc[sample_indices].copy()
    y_sample = y_test.iloc[sample_indices]
    y_pred_dt_sample = y_pred_dt[sample_indices]
    y_pred_rf_sample = y_pred_rf[sample_indices]

    X_sample['真实标签'] = y_sample
    X_sample['决策树预测'] = y_pred_dt_sample
    X_sample['随机森林预测'] = y_pred_rf_sample
    X_sample['决策树预测正确'] = (y_sample == y_pred_dt_sample).astype(int)
    X_sample['随机森林预测正确'] = (y_sample == y_pred_rf_sample).astype(int)

    plt.figure()
    models = ['决策树', '随机森林']
    accuracy = [
        X_sample['决策树预测正确'].mean(),
        X_sample['随机森林预测正确'].mean()
    ]

    plt.bar(models, accuracy, color=['#5DA5DA', '#FAA43A'], alpha=0.8)
    plt.ylabel('预测正确率', fontsize=7)
    plt.title('模型整体预测正确率对比', fontsize=7)
    plt.ylim(0, 1)
    plt.yticks(fontsize=8)

    for i, v in enumerate(accuracy):
        plt.text(i, v + 0.01, f'{v * 100:.2f}%', ha='center', fontsize=7)

    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.subplot(1, 2, 1)
    sex_correct_dt = X_sample.groupby('Sex')['决策树预测正确'].mean()
    sns.barplot(x=sex_correct_dt.index, y=sex_correct_dt.values, color='#5DA5DA', alpha=0.8)
    plt.xlabel('性别', fontsize=7)
    plt.ylabel('预测正确率', fontsize=7)
    plt.title('决策树按性别预测正确率', fontsize=7)
    plt.ylim(0, 1)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)

    for i, v in enumerate(sex_correct_dt.values):
        plt.text(i, v + 0.01, f'{v * 100:.2f}%', ha='center', fontsize=7)

    plt.subplot(1, 2, 2)
    sex_correct_rf = X_sample.groupby('Sex')['随机森林预测正确'].mean()
    sns.barplot(x=sex_correct_rf.index, y=sex_correct_rf.values, color='#FAA43A', alpha=0.8)
    plt.xlabel('性别', fontsize=7)
    plt.ylabel('预测正确率', fontsize=7)
    plt.title('随机森林按性别预测正确率', fontsize=7)
    plt.ylim(0, 1)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)

    for i, v in enumerate(sex_correct_rf.values):
        plt.text(i, v + 0.01, f'{v * 100:.2f}%', ha='center', fontsize=7)

    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.subplot(1, 2, 1)
    pclass_correct_dt = X_sample.groupby('Pclass')['决策树预测正确'].mean()
    sns.barplot(x=pclass_correct_dt.index, y=pclass_correct_dt.values, color='#5DA5DA', alpha=0.8)
    plt.xlabel('舱位等级', fontsize=7)
    plt.ylabel('预测正确率', fontsize=7)
    plt.title('决策树按舱位等级预测正确率', fontsize=7)
    plt.ylim(0, 1)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)

    for i, v in enumerate(pclass_correct_dt.values):
        plt.text(i, v + 0.01, f'{v * 100:.2f}%', ha='center', fontsize=7)

    plt.subplot(1, 2, 2)
    pclass_correct_rf = X_sample.groupby('Pclass')['随机森林预测正确'].mean()
    sns.barplot(x=pclass_correct_rf.index, y=pclass_correct_rf.values, color='#FAA43A', alpha=0.8)
    plt.xlabel('舱位等级', fontsize=7)
    plt.ylabel('预测正确率', fontsize=7)
    plt.title('随机森林按舱位等级预测正确率', fontsize=7)
    plt.ylim(0, 1)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)

    for i, v in enumerate(pclass_correct_rf.values):
        plt.text(i, v + 0.01, f'{v * 100:.2f}%', ha='center', fontsize=7)

    plt.tight_layout()
    plt.show()

    print("预测结果对比可视化完成，共3张图表")


# 主函数
def main():
    print("开始执行泰坦尼克号数据分析与可视化...")

    # 使用当前时间作为随机种子，确保每次运行数据不同
    random_seed = int(pd.Timestamp.now().timestamp())
    print(f"使用随机种子: {random_seed}")

    X, y, train_data = load_data(random_seed)
    visualize_data_exploration(train_data)

    dt_model, rf_model, X_test, y_test, y_pred_dt, y_pred_rf, y_prob_dt, y_prob_rf, metrics, preprocessor = build_models(
        X, y)

    visualize_performance_comparison(metrics, y_test, y_prob_dt, y_prob_rf)
    visualize_feature_importance(dt_model, rf_model, preprocessor)
    visualize_tree_structure(dt_model, rf_model, preprocessor)
    visualize_prediction_comparison(X_test, y_test, y_pred_dt, y_pred_rf, dt_model, rf_model)

    print("\n可视化完成！已移除固定随机种子，每次运行结果将不同")


if __name__ == "__main__":
    main()