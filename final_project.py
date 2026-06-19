"""
============================================================
Credit Card Fraud Detection — Unified ML Pipeline
============================================================
[구성 개요]
  Part 1 : 데이터 로드 & 탐색 (EDA)
  Part 2 : 차원 축소 (PCA 2D 시각화)
  Part 3 : 분류 모델 학습 & 평가
             ├─ Baseline  : AdalineSGD (선형 퍼셉트론)
             ├─ Logistic Regression  (L2 + GridSearch)
             ├─ SVM                  (rbf + GridSearch)  ← 추가
             └─ Random Forest        (앙상블 + GridSearch)
  Part 4 : 회귀 모델로 거래 금액(Amount) 예측
             ├─ Linear Regression (OLS baseline)
             ├─ RANSAC             (이상치 탐지 도구)
             ├─ Ridge / Lasso      (정규화 + GridSearch)
             └─ Random Forest Regressor
  Part 5 : 비지도 이상치 탐지 (DBSCAN)
  Part 6 : 잔차 기반 이상 거래 탐지 분석
  Part 7 : 전체 모델 성능 종합 비교 대시보드
============================================================
필요 패키지:
  pip install numpy pandas matplotlib scikit-learn
데이터셋:
  Kaggle "Credit Card Fraud Detection" → creditcard.csv
  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings

warnings.filterwarnings('ignore')

# ── 한글 폰트 (OS별 자동 감지) ──────────────────────────────
import platform
if platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
elif platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:                          # Linux (Ubuntu 등)
    plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 11

# ── Scikit-learn ─────────────────────────────────────────────
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

# 분류 모델
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# 회귀 모델
from sklearn.linear_model import LinearRegression, RANSACRegressor, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

# ── [추가] 피드백 반영을 위한 추가 라이브러리 ────────────────
#   - precision_score / recall_score / confusion_matrix : Recall·Precision 평가
#   - IsolationForest / LocalOutlierFactor / OneClassSVM : 비지도 이상 탐지 확장
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


# ╔══════════════════════════════════════════════════════════╗
# ║  PART 1 · 데이터 로드 & EDA                              ║
# ╚══════════════════════════════════════════════════════════╝
print("=" * 60)
print("PART 1 · 데이터 로드 & 탐색적 데이터 분석 (EDA)")
print("=" * 60)

df = pd.read_csv('creditcard.csv')
print("\n[원본 데이터셋 통계]")
print(df.describe())

# ── 20,000건 샘플링 (메모리·시간 효율) ──────────────────────
df_s = df.sample(20_000, random_state=1)
feature_names = list(df_s.drop(columns=['Class', 'Amount']).columns)

# 분류용 타겟
X_clf = df_s.drop(columns=['Class']).values          # Amount 포함
y_clf = df_s['Class'].values

# 회귀용 타겟 (Amount 예측, Class 제외)
X_reg = df_s.drop(columns=['Class', 'Amount']).values
y_reg = df_s['Amount'].values
fraud_labels = df_s['Class'].values

print(f"\n[샘플 정보] 총 {len(df_s):,}건")
print(f"  정상: {np.sum(y_clf==0):,}건 | 사기: {np.sum(y_clf==1):,}건 "
      f"| 사기 비율: {np.sum(y_clf==1)/len(y_clf)*100:.3f}%")
print(f"  거래 금액 — 평균 ${np.mean(y_reg):.2f} "
      f"| 중앙값 ${np.median(y_reg):.2f} | 최대 ${np.max(y_reg):.2f}")

# ── EDA 시각화 ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('EDA — Credit Card Fraud Dataset', fontsize=14, fontweight='bold')

# 클래스 불균형
axes[0].bar(['Normal', 'Fraud'],
            [np.sum(y_clf==0), np.sum(y_clf==1)],
            color=['steelblue', 'tomato'], edgecolor='black', alpha=0.85)
axes[0].set_title('Class Distribution (Imbalanced)')
axes[0].set_ylabel('Count')
for bar, cnt in zip(axes[0].patches, [np.sum(y_clf==0), np.sum(y_clf==1)]):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+30,
                 f'{cnt:,}', ha='center', fontsize=10)

# Amount 전체 분포
axes[1].hist(y_reg, bins=60, color='steelblue', edgecolor='black', alpha=0.7)
axes[1].set_title('Transaction Amount — Full Range')
axes[1].set_xlabel('Amount ($)')
axes[1].set_ylabel('Frequency')

# Amount 줌인 (< $200)
axes[2].hist(y_reg[y_reg < 200], bins=60, color='coral', edgecolor='black', alpha=0.7)
axes[2].set_title('Transaction Amount — Zoom (< $200)')
axes[2].set_xlabel('Amount ($)')
axes[2].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

print(f"\n  $50 미만 거래:  {np.sum(y_reg<50)/len(y_reg)*100:.1f}%")
print(f"  $200 미만 거래: {np.sum(y_reg<200)/len(y_reg)*100:.1f}%")


# ── 데이터 분할 ──────────────────────────────────────────────
# 분류: Stratified 분할로 사기 비율 유지
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clf, y_clf, test_size=0.3, random_state=1, stratify=y_clf
)

# 회귀: fraud_labels도 함께 분할하여 잔차 분석에 활용
X_train_r, X_test_r, y_train_r, y_test_r, fraud_train, fraud_test = train_test_split(
    X_reg, y_reg, fraud_labels, test_size=0.3, random_state=1
)

print(f"\n[데이터 분할]")
print(f"  분류 — 학습: {len(X_train_c):,}건 | 테스트: {len(X_test_c):,}건 (Stratified)")
print(f"  회귀 — 학습: {len(X_train_r):,}건 | 테스트: {len(X_test_r):,}건")


# ╔══════════════════════════════════════════════════════════╗
# ║  PART 2 · PCA 2D 차원 축소 시각화                        ║
# ╚══════════════════════════════════════════════════════════╝
print("\n" + "=" * 60)
print("PART 2 · PCA 2D 차원 축소 — 선형 분리 가능성 확인")
print("=" * 60)

pca_pipe = make_pipeline(StandardScaler(), PCA(n_components=2))
X_train_pca = pca_pipe.fit_transform(X_train_c)

# 분류용 피처에 Amount가 포함되어 있어, PCA 후 두 클래스 겹침 정도 확인
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('PCA 2D Projection — Credit Card Transactions', fontsize=13, fontweight='bold')

# 전체
axes[0].scatter(X_train_pca[y_train_c==0, 0], X_train_pca[y_train_c==0, 1],
                label='Normal', alpha=0.3, c='lightgray', s=8)
axes[0].scatter(X_train_pca[y_train_c==1, 0], X_train_pca[y_train_c==1, 1],
                label='Fraud', alpha=0.8, c='red', s=20)
axes[0].set_title('Full Range')
axes[0].set_xlabel('PC 1')
axes[0].set_ylabel('PC 2')
axes[0].legend()

# 사기 거래 밀집 구간 줌인 (PC1: -10~10, PC2: -10~10)
mask_zoom = (X_train_pca[:, 0] > -10) & (X_train_pca[:, 0] < 10) & \
            (X_train_pca[:, 1] > -10) & (X_train_pca[:, 1] < 10)
axes[1].scatter(X_train_pca[mask_zoom & (y_train_c==0), 0],
                X_train_pca[mask_zoom & (y_train_c==0), 1],
                label='Normal', alpha=0.3, c='lightgray', s=8)
axes[1].scatter(X_train_pca[mask_zoom & (y_train_c==1), 0],
                X_train_pca[mask_zoom & (y_train_c==1), 1],
                label='Fraud', alpha=0.9, c='red', s=25)
axes[1].set_title('Zoom-in (PC1/PC2: −10 ~ 10)')
axes[1].set_xlabel('PC 1')
axes[1].set_ylabel('PC 2')
axes[1].legend()

plt.tight_layout()
plt.show()

print("  → PCA 2D에서 사기 거래가 일부 분리되는 구간이 존재하나,")
print("    완전한 선형 분리는 어려움 → 비선형 모델의 필요성 시사")


# ╔══════════════════════════════════════════════════════════╗
# ║  PART 3 · 분류 모델 학습 & 평가                          ║
# ╚══════════════════════════════════════════════════════════╝
print("\n" + "=" * 60)
print("PART 3 · 분류 모델 학습 & 평가")
print("=" * 60)

# ── 3-1. Baseline : AdalineSGD ───────────────────────────────
print("\n[3-1] Baseline — AdalineSGD (수학적으로 Adaline과 동일)")
adaline = make_pipeline(
    StandardScaler(),
    SGDClassifier(loss='squared_error', max_iter=20, random_state=1)
)
adaline.fit(X_train_c, y_train_c)
y_pred_ada = adaline.predict(X_test_c)
f1_ada = f1_score(y_test_c, y_pred_ada, zero_division=0)
auc_ada = roc_auc_score(y_test_c, y_pred_ada)
print(f"  F1: {f1_ada:.4f}  |  ROC-AUC: {auc_ada:.4f}")
print(classification_report(y_test_c, y_pred_ada, zero_division=0))

# ── 3-2. Logistic Regression + GridSearch ───────────────────
print("[3-2] Logistic Regression — L2 Regularization + GridSearch")
pipe_lr = make_pipeline(
    StandardScaler(),
    LogisticRegression(penalty='l2', random_state=1, solver='liblinear')
)
gs_lr = GridSearchCV(
    pipe_lr,
    [{'logisticregression__C': [0.01, 0.1, 1.0, 10.0]}],
    scoring='f1', cv=5, n_jobs=-1
)
gs_lr.fit(X_train_c, y_train_c)
y_pred_lr = gs_lr.best_estimator_.predict(X_test_c)
f1_lr  = f1_score(y_test_c, y_pred_lr)
auc_lr = roc_auc_score(y_test_c,
    gs_lr.best_estimator_.predict_proba(X_test_c)[:, 1])
print(f"  Best C={gs_lr.best_params_['logisticregression__C']} "
      f"| F1: {f1_lr:.4f}  |  ROC-AUC: {auc_lr:.4f}")

# ── 3-3. SVM + GridSearch (추가) ────────────────────────────
print("\n[3-3] SVM (rbf kernel) + GridSearch  ← 추가 모델")
# SVM은 불균형 데이터에 class_weight='balanced'를 적용하는 것이 중요
pipe_svm = make_pipeline(
    StandardScaler(),
    SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=1)
)
gs_svm = GridSearchCV(
    pipe_svm,
    [{'svc__C': [0.1, 1.0, 10.0], 'svc__gamma': ['scale', 'auto']}],
    scoring='f1', cv=3, n_jobs=-1
)
gs_svm.fit(X_train_c, y_train_c)
y_pred_svm = gs_svm.best_estimator_.predict(X_test_c)
f1_svm  = f1_score(y_test_c, y_pred_svm)
auc_svm = roc_auc_score(y_test_c,
    gs_svm.best_estimator_.predict_proba(X_test_c)[:, 1])
print(f"  Best C={gs_svm.best_params_['svc__C']}, "
      f"gamma={gs_svm.best_params_['svc__gamma']} "
      f"| F1: {f1_svm:.4f}  |  ROC-AUC: {auc_svm:.4f}")

# ── 3-4. Random Forest + GridSearch ─────────────────────────
print("\n[3-4] Random Forest — 앙상블 + GridSearch")
pipe_rf_c = make_pipeline(
    RandomForestClassifier(random_state=1, class_weight='balanced')
)
gs_rf_c = GridSearchCV(
    pipe_rf_c,
    [{'randomforestclassifier__n_estimators': [50, 100, 150],
      'randomforestclassifier__max_depth':    [5, 10, None]}],
    scoring='f1', cv=3, n_jobs=-1
)
gs_rf_c.fit(X_train_c, y_train_c)
y_pred_rf_c = gs_rf_c.best_estimator_.predict(X_test_c)
f1_rf_c  = f1_score(y_test_c, y_pred_rf_c)
auc_rf_c = roc_auc_score(y_test_c,
    gs_rf_c.best_estimator_.predict_proba(X_test_c)[:, 1])
print(f"  Best params: {gs_rf_c.best_params_}")
print(f"  F1: {f1_rf_c:.4f}  |  ROC-AUC: {auc_rf_c:.4f}")
print("\n  --- Final Classification Report (Random Forest) ---")
print(classification_report(y_test_c, y_pred_rf_c))

# ╔══════════════════════════════════════════════════════════╗
# ║  [추가] PART 3-2 · Precision / Recall 평가                ║
# ║   [피드백 반영] "사기 탐지를 했는가/안했는가"가 중요 →     ║
# ║   Recall(실제 사기 중 탐지 비율)을 Precision과 함께 명시  ║
# ╚══════════════════════════════════════════════════════════╝
print("\n" + "=" * 60)
print("[추가] PART 3-2 · Precision / Recall 평가 (Recall 중심)")
print("=" * 60)

precision_ada = precision_score(y_test_c, y_pred_ada, zero_division=0)
recall_ada    = recall_score(y_test_c, y_pred_ada, zero_division=0)

precision_lr  = precision_score(y_test_c, y_pred_lr,  zero_division=0)
recall_lr     = recall_score(y_test_c, y_pred_lr,  zero_division=0)

precision_svm = precision_score(y_test_c, y_pred_svm, zero_division=0)
recall_svm    = recall_score(y_test_c, y_pred_svm, zero_division=0)

precision_rf_c = precision_score(y_test_c, y_pred_rf_c, zero_division=0)
recall_rf_c    = recall_score(y_test_c, y_pred_rf_c, zero_division=0)

clf_precisions = [precision_ada, precision_lr, precision_svm, precision_rf_c]
clf_recalls    = [recall_ada,    recall_lr,    recall_svm,    recall_rf_c]

n_total_fraud_test = int(np.sum(y_test_c == 1))

for name_m, y_pred_m, prec_m, rec_m in zip(
        ['AdalineSGD', 'Logistic Regression', 'SVM', 'Random Forest'],
        [y_pred_ada, y_pred_lr, y_pred_svm, y_pred_rf_c],
        clf_precisions, clf_recalls):
    tn_m, fp_m, fn_m, tp_m = confusion_matrix(y_test_c, y_pred_m).ravel()
    print(f"  [{name_m:>20}] Precision: {prec_m:.4f} | Recall: {rec_m:.4f}  "
          f"→ 실제 사기 {n_total_fraud_test}건 중 {tp_m}건 탐지, {fn_m}건 미탐지(놓침)")

print("\n  ※ Recall이 낮으면 실제 사기 거래를 놓치는(False Negative) 경우가 많다는 의미이며,")
print("    금융 사기 탐지에서는 Precision 못지않게 Recall이 핵심 지표가 됨.")

# ── 분류 Feature Importance ─────────────────────────────────
importances_c = (gs_rf_c.best_estimator_
                 .named_steps['randomforestclassifier']
                 .feature_importances_)
indices_c = np.argsort(importances_c)[::-1]
clf_feat_names = list(df_s.drop(columns=['Class']).columns)  # Amount 포함

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle('Classification — Feature Importance & Model Comparison',
             fontsize=13, fontweight='bold')

axes[0].bar(range(10), importances_c[indices_c[:10]],
            color='steelblue', edgecolor='black', alpha=0.85)
axes[0].set_xticks(range(10))
axes[0].set_xticklabels([clf_feat_names[i] for i in indices_c[:10]], rotation=45)
axes[0].set_title('Top-10 Feature Importances (RF Classifier)')
axes[0].set_ylabel('Importance')

# 모델별 F1 / AUC 비교
clf_models  = ['AdalineSGD', 'Logistic\nRegression', 'SVM', 'Random\nForest']
clf_f1s     = [f1_ada,  f1_lr,  f1_svm,  f1_rf_c]
clf_aucs    = [auc_ada, auc_lr, auc_svm, auc_rf_c]
x_c = np.arange(len(clf_models))
w   = 0.35
bars1 = axes[1].bar(x_c - w/2, clf_f1s,  w, label='F1',      color='#3498DB', alpha=0.85, edgecolor='black')
bars2 = axes[1].bar(x_c + w/2, clf_aucs, w, label='ROC-AUC', color='#E74C3C', alpha=0.85, edgecolor='black')
axes[1].set_xticks(x_c)
axes[1].set_xticklabels(clf_models)
axes[1].set_ylim(0, 1.15)
axes[1].set_title('Classifier Comparison — F1 & ROC-AUC')
axes[1].legend()
for bar in bars1:
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                 f'{bar.get_height():.3f}', ha='center', fontsize=8)
for bar in bars2:
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                 f'{bar.get_height():.3f}', ha='center', fontsize=8)

plt.tight_layout()
plt.show()

print(f"\n[분류 최고 모델] ROC-AUC 기준:")
best_clf_name = clf_models[np.argmax(clf_aucs)]
print(f"  → {best_clf_name.replace(chr(10),'')}  "
      f"(AUC={max(clf_aucs):.4f})")

# ── [추가] Precision / Recall 비교 시각화 ───────────────────
fig, ax = plt.subplots(figsize=(8, 5))
x_pr = np.arange(len(clf_models))
w_pr = 0.35
bars_p = ax.bar(x_pr - w_pr/2, clf_precisions, w_pr, label='Precision',
                 color='#3498DB', alpha=0.85, edgecolor='black')
bars_r = ax.bar(x_pr + w_pr/2, clf_recalls,    w_pr, label='Recall',
                 color='#E74C3C', alpha=0.85, edgecolor='black')
ax.set_xticks(x_pr)
ax.set_xticklabels(clf_models)
ax.set_ylim(0, 1.15)
ax.set_title('[추가] Classifier Comparison — Precision vs Recall\n(Recall = 실제 사기 탐지율)')
ax.legend()
for bar in bars_p:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
            f'{bar.get_height():.3f}', ha='center', fontsize=8)
for bar in bars_r:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
            f'{bar.get_height():.3f}', ha='center', fontsize=8)
plt.tight_layout()
plt.show()

best_recall_idx = np.argmax(clf_recalls)
print(f"\n[추가] Recall 기준 최고 모델: {clf_models[best_recall_idx].replace(chr(10),'')} "
      f"(Recall={clf_recalls[best_recall_idx]:.4f}) — 사기를 가장 많이 탐지해낸 모델")


# ╔══════════════════════════════════════════════════════════╗
# ║  [추가] PART 3-5 · 주요 변수 분리 기준 명확화             ║
# ║   [피드백 반영] V1~V28 같은 모호한 변수명/중요도 랭킹 대신,║
# ║   실제 값 분포와 구체적 임계값(threshold)으로 분리 기준 제시║
# ╚══════════════════════════════════════════════════════════╝
print("\n" + "=" * 60)
print("[추가] PART 3-5 · 주요 변수의 정상/사기 분리 기준(Threshold) 분석")
print("=" * 60)

top_n_feat = 5
top_feat_idx = indices_c[:top_n_feat]
top_feat_names = [clf_feat_names[i] for i in top_feat_idx]

X_train_c_df = pd.DataFrame(X_train_c, columns=clf_feat_names)
X_train_c_df['Class'] = y_train_c

sep_summary = []
print(f"\n  상위 {top_n_feat}개 변수의 정상 vs 사기 거래 값 분포 및 분리 기준:\n")
for feat in top_feat_names:
    normal_vals = X_train_c_df.loc[X_train_c_df['Class'] == 0, feat]
    fraud_vals  = X_train_c_df.loc[X_train_c_df['Class'] == 1, feat]
    # 사기 거래의 중앙값을 분리 기준(threshold)으로 사용
    threshold = fraud_vals.median()
    is_lower  = fraud_vals.median() < normal_vals.median()
    rule_op   = '<=' if is_lower else '>='
    flagged   = (X_train_c_df[feat] <= threshold) if is_lower else (X_train_c_df[feat] >= threshold)
    flagged_df = X_train_c_df[flagged]
    fraud_rate_in_flag = np.mean(flagged_df['Class']) * 100
    rule_text = f"{feat} {rule_op} {threshold:.2f}"
    print(f"  [{feat}] 정상 평균={normal_vals.mean():.2f}, 사기 평균={fraud_vals.mean():.2f} | "
          f"분리 기준: {rule_text}  → 해당 조건 거래의 사기 비율: {fraud_rate_in_flag:.2f}%")
    sep_summary.append({
        'Feature': feat, 'Normal_Mean': round(normal_vals.mean(), 3),
        'Fraud_Mean': round(fraud_vals.mean(), 3), 'Threshold': round(threshold, 3),
        'Rule': rule_text, 'Fraud_Rate_in_Rule(%)': round(fraud_rate_in_flag, 2)
    })

sep_df = pd.DataFrame(sep_summary)
print("\n--- 변수별 분리 기준 요약표 (V1~V28 단순 중요도 대신 구체적 수치 기준) ---")
print(sep_df.to_string(index=False))

# 박스플롯으로 분리 기준 시각화 (단순 중요도 막대 대신 실제 분포·임계값으로 확인)
fig, axes = plt.subplots(1, top_n_feat, figsize=(4*top_n_feat, 5))
fig.suptitle('[추가] Top Features — Normal vs Fraud Distribution (분리 기준 시각화)',
             fontsize=13, fontweight='bold')
for i, feat in enumerate(top_feat_names):
    data_to_plot = [X_train_c_df.loc[X_train_c_df['Class'] == 0, feat],
                    X_train_c_df.loc[X_train_c_df['Class'] == 1, feat]]
    bp = axes[i].boxplot(data_to_plot, tick_labels=['Normal', 'Fraud'],
                          patch_artist=True, showfliers=False)
    for patch, color in zip(bp['boxes'], ['lightgray', 'tomato']):
        patch.set_facecolor(color)
    axes[i].set_title(feat)
    thr_val = sep_df.loc[sep_df['Feature'] == feat, 'Threshold'].values[0]
    axes[i].axhline(y=thr_val, color='blue', linestyle='--', linewidth=1, label='Threshold')
    axes[i].legend(fontsize=7)
plt.tight_layout()
plt.show()

print("\n  ※ V1~V28은 PCA로 익명화된 변수라 의미 해석은 어렵지만,")
print("    위 표·박스플롯처럼 '구체적인 값 기준'으로 정상/사기 분리 경계를 제시하면")
print("    모호한 중요도 랭킹보다 실무에서 훨씬 활용도가 높음.")


# ╔══════════════════════════════════════════════════════════╗
# ║  PART 4 · 회귀 모델 — 거래 금액(Amount) 예측             ║
# ╚══════════════════════════════════════════════════════════╝
print("\n" + "=" * 60)
print("PART 4 · 회귀 모델 — 거래 금액(Amount) 예측")
print("  [목적] 정상 패턴으로 금액을 예측하고,")
print("         예측 오차(잔차)가 큰 거래 = 사기 가능성 높음을 검증")
print("=" * 60)

# ── 4-1. Linear Regression (OLS baseline) ───────────────────
print("\n[4-1] Baseline — Ordinary Least Squares (OLS)")
pipe_ols = make_pipeline(StandardScaler(), LinearRegression())
pipe_ols.fit(X_train_r, y_train_r)
y_pred_ols = pipe_ols.predict(X_test_r)
mse_ols = mean_squared_error(y_test_r, y_pred_ols)
mae_ols = mean_absolute_error(y_test_r, y_pred_ols)
r2_ols  = r2_score(y_test_r, y_pred_ols)
print(f"  MSE: {mse_ols:.2f} | MAE: {mae_ols:.2f} | R²: {r2_ols:.4f}")

# ── 4-2. RANSAC (이상치 탐지 도구) ──────────────────────────
print("\n[4-2] RANSAC — 이상치 탐지 (예측 성능보다 Outlier 분석이 목적)")
pipe_ransac = make_pipeline(
    StandardScaler(),
    RANSACRegressor(random_state=1, max_trials=100)
)
pipe_ransac.fit(X_train_r, y_train_r)
y_pred_ransac = pipe_ransac.predict(X_test_r)
mse_ransac = mean_squared_error(y_test_r, y_pred_ransac)
mae_ransac = mean_absolute_error(y_test_r, y_pred_ransac)
r2_ransac  = r2_score(y_test_r, y_pred_ransac)
print(f"  MSE: {mse_ransac:.2f} | MAE: {mae_ransac:.2f} | R²: {r2_ransac:.4f}")
print("  ※ RANSAC은 이상치를 제외 후 학습 → R²가 낮은 것은 정상")

# RANSAC Inlier/Outlier 분석
inlier_mask = pipe_ransac.named_steps['ransacregressor'].inlier_mask_
n_in  = np.sum(inlier_mask)
n_out = np.sum(~inlier_mask)
fr_in  = np.mean(fraud_train[inlier_mask])  * 100
fr_out = np.mean(fraud_train[~inlier_mask]) * 100 if n_out > 0 else 0

print(f"\n  RANSAC 학습 데이터 이상치 분석:")
print(f"    Inliers  ({n_in:>5}건): 사기 비율 {fr_in:.2f}%")
print(f"    Outliers ({n_out:>5}건): 사기 비율 {fr_out:.2f}%")
ratio_str = (f"→ Outlier 사기 비율이 Inlier 대비 "
             f"{fr_out/max(fr_in,0.001):.1f}배 높음 ✓"
             if fr_out > fr_in else
             "→ Outlier 사기 비율이 Inlier보다 낮거나 같음")
print(f"    {ratio_str}")

# RANSAC 시각화
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(['Inliers\n(Normal Pattern)', 'Outliers\n(Anomalous Pattern)'],
       [fr_in, fr_out],
       color=['#4CAF50', '#F44336'], edgecolor='black', alpha=0.85, width=0.5)
ax.set_title('RANSAC: Fraud Rate — Inliers vs Outliers')
ax.set_ylabel('Fraud Rate (%)')
for i, (rate, cnt) in enumerate(zip([fr_in, fr_out], [n_in, n_out])):
    ax.text(i, rate + 0.05, f'{rate:.2f}%\n({cnt}건)',
            ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.show()

# ── 4-3. Ridge (L2) + GridSearch ────────────────────────────
print("\n[4-3] Ridge Regression — L2 정규화 + GridSearch")
pipe_ridge = make_pipeline(StandardScaler(), Ridge())
gs_ridge = GridSearchCV(
    pipe_ridge,
    [{'ridge__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}],
    scoring='r2', cv=5, n_jobs=-1
)
gs_ridge.fit(X_train_r, y_train_r)
y_pred_ridge = gs_ridge.best_estimator_.predict(X_test_r)
mse_ridge = mean_squared_error(y_test_r, y_pred_ridge)
mae_ridge = mean_absolute_error(y_test_r, y_pred_ridge)
r2_ridge  = r2_score(y_test_r, y_pred_ridge)
print(f"  Best alpha={gs_ridge.best_params_['ridge__alpha']} "
      f"| MSE: {mse_ridge:.2f} | MAE: {mae_ridge:.2f} | R²: {r2_ridge:.4f}")

# ── 4-4. Lasso (L1) + GridSearch ────────────────────────────
print("\n[4-4] Lasso Regression — L1 정규화 + GridSearch (자동 변수 선택)")
pipe_lasso = make_pipeline(StandardScaler(), Lasso(max_iter=10000))
gs_lasso = GridSearchCV(
    pipe_lasso,
    [{'lasso__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}],
    scoring='r2', cv=5, n_jobs=-1
)
gs_lasso.fit(X_train_r, y_train_r)
y_pred_lasso = gs_lasso.best_estimator_.predict(X_test_r)
mse_lasso = mean_squared_error(y_test_r, y_pred_lasso)
mae_lasso = mean_absolute_error(y_test_r, y_pred_lasso)
r2_lasso  = r2_score(y_test_r, y_pred_lasso)
print(f"  Best alpha={gs_lasso.best_params_['lasso__alpha']} "
      f"| MSE: {mse_lasso:.2f} | MAE: {mae_lasso:.2f} | R²: {r2_lasso:.4f}")

lasso_coefs = gs_lasso.best_estimator_.named_steps['lasso'].coef_
n_nz = np.sum(lasso_coefs != 0)
n_z  = np.sum(lasso_coefs == 0)
print(f"\n  Lasso 변수 선택: {len(lasso_coefs)}개 중 {n_nz}개 사용 | {n_z}개 제거")
selected = [feature_names[i] for i in range(len(lasso_coefs)) if lasso_coefs[i] != 0]
removed  = [feature_names[i] for i in range(len(lasso_coefs)) if lasso_coefs[i] == 0]
if selected: print(f"  선택: {', '.join(selected)}")
if removed:  print(f"  제거: {', '.join(removed)}")

# ── 4-5. Random Forest Regressor + GridSearch ───────────────
print("\n[4-5] Random Forest Regressor — 비선형 패턴 포착 + GridSearch")
pipe_rf_r = make_pipeline(RandomForestRegressor(random_state=1))
gs_rf_r = GridSearchCV(
    pipe_rf_r,
    [{'randomforestregressor__n_estimators': [50, 100, 150],
      'randomforestregressor__max_depth':    [5, 10, None]}],
    scoring='r2', cv=3, n_jobs=-1
)
gs_rf_r.fit(X_train_r, y_train_r)
y_pred_rf_r = gs_rf_r.best_estimator_.predict(X_test_r)
mse_rf_r = mean_squared_error(y_test_r, y_pred_rf_r)
mae_rf_r = mean_absolute_error(y_test_r, y_pred_rf_r)
r2_rf_r  = r2_score(y_test_r, y_pred_rf_r)
print(f"  Best: {gs_rf_r.best_params_}")
print(f"  MSE: {mse_rf_r:.2f} | MAE: {mae_rf_r:.2f} | R²: {r2_rf_r:.4f}")

# RF Regressor Feature Importance
rf_imp = (gs_rf_r.best_estimator_
          .named_steps['randomforestregressor']
          .feature_importances_)
rf_idx = np.argsort(rf_imp)[::-1]
print(f"  상위 5 변수: "
      + ", ".join([f"{feature_names[i]} ({rf_imp[i]:.3f})" for i in rf_idx[:5]]))

# ── 4-6. 회귀 모델 성능 비교 ────────────────────────────────
reg_results = pd.DataFrame({
    'Model': ['OLS', 'Ridge', 'Lasso', 'RF Regressor'],
    'MSE':   [mse_ols, mse_ridge, mse_lasso, mse_rf_r],
    'MAE':   [mae_ols, mae_ridge, mae_lasso, mae_rf_r],
    'R²':    [r2_ols,  r2_ridge,  r2_lasso,  r2_rf_r]
})
print("\n--- Regression Model Comparison ---")
print("  ※ RANSAC은 이상치 탐지 목적 → 예측 비교에서 제외")
print(reg_results.to_string(index=False))
best_reg_idx  = reg_results['R²'].idxmax()
best_reg_name = reg_results.loc[best_reg_idx, 'Model']
best_reg_r2   = reg_results.loc[best_reg_idx, 'R²']
print(f"\n  최고 성능 모델: {best_reg_name} (R²={best_reg_r2:.4f})")

best_reg_preds = {'OLS': y_pred_ols, 'Ridge': y_pred_ridge,
                  'Lasso': y_pred_lasso, 'RF Regressor': y_pred_rf_r}[best_reg_name]

# 회귀 비교 시각화
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Regression Model Analysis', fontsize=13, fontweight='bold')
colors_reg = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12']

axes[0].barh(reg_results['Model'], reg_results['R²'],
             color=colors_reg, edgecolor='black', alpha=0.85)
axes[0].set_xlabel('R²')
axes[0].set_title('R² Score (higher is better)')
for i, v in enumerate(reg_results['R²']):
    axes[0].text(v+0.001, i, f'{v:.4f}', va='center', fontsize=9)

axes[1].barh(reg_results['Model'], reg_results['MAE'],
             color=colors_reg, edgecolor='black', alpha=0.85)
axes[1].set_xlabel('MAE ($)')
axes[1].set_title('MAE (lower is better)')
for i, v in enumerate(reg_results['MAE']):
    axes[1].text(v+0.2, i, f'${v:.2f}', va='center', fontsize=9)

axes[2].scatter(y_test_r, best_reg_preds,
                alpha=0.3, c='steelblue', edgecolors='none', s=10)
axes[2].plot([y_test_r.min(), y_test_r.max()],
             [y_test_r.min(), y_test_r.max()],
             'r--', linewidth=2, label='Perfect Prediction')
axes[2].set_xlabel('Actual Amount ($)')
axes[2].set_ylabel('Predicted Amount ($)')
axes[2].set_title(f'Actual vs Predicted — {best_reg_name} (R²={best_reg_r2:.4f})')
axes[2].legend()

plt.tight_layout()
plt.show()

# RF Regressor Feature Importance 시각화
plt.figure(figsize=(10, 5))
plt.bar(range(10), rf_imp[rf_idx[:10]],
        color='teal', edgecolor='black', alpha=0.8)
plt.xticks(range(10), [feature_names[i] for i in rf_idx[:10]], rotation=45)
plt.ylabel('Importance')
plt.title('Feature Importances — Amount Prediction (RF Regressor)')
plt.tight_layout()
plt.show()


# ╔══════════════════════════════════════════════════════════╗
# ║  PART 5 · DBSCAN 비지도 이상치 탐지                      ║
# ╚══════════════════════════════════════════════════════════╝
print("\n" + "=" * 60)
print("PART 5 · DBSCAN — 비지도 밀도 기반 이상치 탐지")
print("=" * 60)

# PCA 2D 공간에서 DBSCAN 수행 (전처리된 저차원 공간 활용)
# 메모리 한계로 테스트 데이터의 PCA 변환 결과를 사용
X_test_pca = pca_pipe.transform(X_test_c)

db = DBSCAN(eps=0.5, min_samples=5, metric='euclidean')
db_labels = db.fit_predict(X_test_pca)

n_noise    = np.sum(db_labels == -1)
n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
print(f"  클러스터 수: {n_clusters}개 | 노이즈(이상치)로 분류된 포인트: {n_noise}개")

noise_mask = db_labels == -1
fr_noise   = np.mean(y_test_c[noise_mask]) * 100 if n_noise > 0 else 0
fr_cluster = np.mean(y_test_c[~noise_mask]) * 100

print(f"  클러스터 내 사기 비율:  {fr_cluster:.2f}%")
print(f"  노이즈 포인트 사기 비율: {fr_noise:.2f}%")
ratio_db = fr_noise / max(fr_cluster, 0.001)
print(f"  → 노이즈 포인트의 사기 비율이 클러스터 대비 {ratio_db:.1f}배 "
      f"({'높음 ✓' if fr_noise > fr_cluster else '낮거나 같음'})")

# DBSCAN 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('DBSCAN — Unsupervised Anomaly Detection (PCA 2D Space)',
             fontsize=13, fontweight='bold')

axes[0].scatter(X_test_pca[~noise_mask, 0], X_test_pca[~noise_mask, 1],
                c='lightblue', alpha=0.4, s=8, label=f'Cluster ({np.sum(~noise_mask)})')
axes[0].scatter(X_test_pca[noise_mask, 0], X_test_pca[noise_mask, 1],
                c='orange', alpha=0.8, s=20, label=f'Noise ({n_noise})')
axes[0].set_title('DBSCAN: Cluster vs Noise')
axes[0].set_xlabel('PC 1')
axes[0].set_ylabel('PC 2')
axes[0].legend()

axes[1].bar(['Cluster Points', 'Noise (Outliers)'],
            [fr_cluster, fr_noise],
            color=['#4CAF50', '#F44336'], edgecolor='black', alpha=0.85)
axes[1].set_title('DBSCAN: Fraud Rate in Cluster vs Noise')
axes[1].set_ylabel('Fraud Rate (%)')
for i, (rate, cnt) in enumerate(zip([fr_cluster, fr_noise],
                                    [np.sum(~noise_mask), n_noise])):
    axes[1].text(i, rate + 0.02, f'{rate:.2f}%\n({cnt}건)',
                 ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.show()


# ╔══════════════════════════════════════════════════════════╗
# ║  [추가] PART 5-1 · Isolation Forest / LOF / One-Class SVM ║
# ║   [피드백 반영] 사기 탐지(Anomaly Detection)를 위한        ║
# ║   비지도 학습 기법을 DBSCAN 외에 추가로 확장              ║
# ╚══════════════════════════════════════════════════════════╝
print("\n" + "=" * 60)
print("[추가] PART 5-1 · Isolation Forest / LOF / One-Class SVM")
print("  [피드백 반영] 비지도 Anomaly Detection 기법 확장 + Recall 평가")
print("=" * 60)

scaler_unsup = StandardScaler()
X_train_c_scaled = scaler_unsup.fit_transform(X_train_c)
X_test_c_scaled  = scaler_unsup.transform(X_test_c)

# contamination(이상치 비율)은 학습 데이터의 실제 사기 비율로 설정
contamination_rate = float(np.mean(y_train_c))
print(f"\n  (참고) Contamination 비율 = 학습 데이터 사기 비율 = {contamination_rate*100:.3f}%\n")

# 1) Isolation Forest — 트리 기반 고립도로 이상치 탐지
iso = IsolationForest(contamination=contamination_rate, random_state=1, n_estimators=100)
iso.fit(X_train_c_scaled)
iso_pred_bin = (iso.predict(X_test_c_scaled) == -1).astype(int)   # 1 = 이상치(사기 추정)

# 2) Local Outlier Factor — 국소 밀도 기반 이상치 탐지 (novelty=True → 신규 데이터 예측 가능)
lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination_rate, novelty=True)
lof.fit(X_train_c_scaled)
lof_pred_bin = (lof.predict(X_test_c_scaled) == -1).astype(int)

# 3) One-Class SVM — 정상 데이터의 경계(boundary)를 학습해 이상치 탐지
ocsvm = OneClassSVM(kernel='rbf', nu=max(contamination_rate, 0.01), gamma='scale')
ocsvm.fit(X_train_c_scaled)
ocsvm_pred_bin = (ocsvm.predict(X_test_c_scaled) == -1).astype(int)

unsup_models = {'Isolation Forest': iso_pred_bin,
                'LOF': lof_pred_bin,
                'One-Class SVM': ocsvm_pred_bin}

unsup_results = []
n_total_fraud_test_u = int(np.sum(y_test_c == 1))
for name_u, pred_u in unsup_models.items():
    prec_u = precision_score(y_test_c, pred_u, zero_division=0)
    rec_u  = recall_score(y_test_c, pred_u, zero_division=0)
    f1_u   = f1_score(y_test_c, pred_u, zero_division=0)
    n_flagged = int(np.sum(pred_u))
    n_caught  = int(np.sum((pred_u == 1) & (y_test_c == 1)))
    print(f"  [{name_u:>16}] 탐지된 이상치: {n_flagged:>4}건 | "
          f"실제 사기 중 탐지: {n_caught}/{n_total_fraud_test_u}건 "
          f"| Precision: {prec_u:.4f} | Recall: {rec_u:.4f} | F1: {f1_u:.4f}")
    unsup_results.append({'Model': name_u, 'Precision': prec_u, 'Recall': rec_u, 'F1': f1_u})

unsup_df = pd.DataFrame(unsup_results)
print("\n--- 비지도 이상 탐지 모델 비교 (Recall 중심) ---")
print(unsup_df.to_string(index=False))

# 시각화: 비지도 모델별 Precision / Recall / F1
fig, ax = plt.subplots(figsize=(9, 5))
x_u = np.arange(len(unsup_df))
w_u = 0.25
ax.bar(x_u - w_u, unsup_df['Precision'], w_u, label='Precision', color='#3498DB', edgecolor='black', alpha=0.85)
ax.bar(x_u,       unsup_df['Recall'],    w_u, label='Recall',    color='#E74C3C', edgecolor='black', alpha=0.85)
ax.bar(x_u + w_u, unsup_df['F1'],        w_u, label='F1',        color='#2ECC71', edgecolor='black', alpha=0.85)
ax.set_xticks(x_u)
ax.set_xticklabels(unsup_df['Model'])
ax.set_ylim(0, 1.1)
ax.set_title('[추가] Unsupervised Anomaly Detection — Precision / Recall / F1\n(사기 탐지 여부 = Recall 중심 평가)')
ax.legend()
for i in range(len(unsup_df)):
    for j, col in enumerate(['Precision', 'Recall', 'F1']):
        val = unsup_df.iloc[i][col]
        ax.text(x_u[i] + (j-1)*w_u, val + 0.02, f'{val:.3f}', ha='center', fontsize=8)
plt.tight_layout()
plt.show()

print("\n  ※ 비지도 기법은 레이블 없이도 이상 거래를 탐지할 수 있어,")
print("    신규/미지의 사기 패턴(unseen fraud pattern) 대응에 강점이 있음.")
print("    단, Recall이 지도 학습 대비 낮을 수 있어 보조 탐지 신호로 함께 활용하는 것을 권장.")


# PART 6 · 잔차 기반 이상 거래 탐지 분석

print("\n" + "=" * 60)
print("PART 6 · 잔차 기반 이상 거래 탐지 (Residual-based Anomaly Detection)")
print("  [아이디어] 정상 패턴으로 학습한 회귀 모델의 예측 오차가 크면")
print("             → 해당 거래가 일반 패턴에서 벗어남 → 사기 가능성 ↑")
print("=" * 60)

residuals     = np.abs(y_test_r - best_reg_preds)
overall_fr    = np.mean(fraud_test) * 100

print(f"\n분석 모델: {best_reg_name}")
print(f"전체 테스트 사기 비율: {overall_fr:.2f}%\n")
print("  잔차 크기별 사기 비율:")
print("  " + "-" * 65)

for pct in [5, 10, 20]:
    thr     = np.percentile(residuals, 100 - pct)
    hmask   = residuals >= thr
    fr_high = np.mean(fraud_test[hmask]) * 100
    n_flag  = np.sum(hmask)
    n_fraud = np.sum(fraud_test[hmask] == 1)
    ratio   = fr_high / max(overall_fr, 0.001)
    print(f"  상위 {pct:>2d}% ({n_flag:>4d}건): "
          f"사기 {n_fraud:>2d}건 | 사기율 {fr_high:>5.2f}% "
          f"(전체 평균의 {ratio:.1f}배)")

# 잔차 분포 + 구간별 사기율 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Residual-based Anomaly Detection', fontsize=13, fontweight='bold')

res_normal = residuals[fraud_test == 0]
res_fraud  = residuals[fraud_test == 1]
axes[0].hist(res_normal, bins=50, alpha=0.7, color='steelblue',
             label=f'Normal (n={len(res_normal)})', edgecolor='black')
axes[0].hist(res_fraud, bins=50, alpha=0.7, color='red',
             label=f'Fraud (n={len(res_fraud)})', edgecolor='black')
axes[0].set_title(f'Residual Distribution — {best_reg_name}')
axes[0].set_xlabel('|Actual − Predicted| ($)')
axes[0].set_ylabel('Frequency')
axes[0].legend()

pct_edges = [0, 25, 50, 75, 90, 95, 100]
fr_by_res, bar_labels = [], []
for i in range(len(pct_edges) - 1):
    lo = np.percentile(residuals, pct_edges[i])
    hi = np.percentile(residuals, pct_edges[i+1])
    mask = (residuals >= lo) & (residuals < hi) if i < len(pct_edges)-2 else (residuals >= lo)
    if np.sum(mask) > 0:
        fr_by_res.append(np.mean(fraud_test[mask]) * 100)
        bar_labels.append(f'{pct_edges[i]}-{pct_edges[i+1]}%')

bar_colors_r = ['#b0d4b0'] * (len(bar_labels) - 2) + ['#f5a623', '#e74c3c']
axes[1].bar(range(len(bar_labels)), fr_by_res,
            color=bar_colors_r, edgecolor='black', alpha=0.85)
axes[1].set_xticks(range(len(bar_labels)))
axes[1].set_xticklabels(bar_labels, fontsize=9)
axes[1].set_xlabel('Residual Percentile Range')
axes[1].set_ylabel('Fraud Rate (%)')
axes[1].set_title('Fraud Rate by Residual Magnitude')
axes[1].axhline(y=overall_fr, color='blue', linestyle='--', linewidth=1.5,
                label=f'Overall Fraud Rate ({overall_fr:.2f}%)')
axes[1].legend()
plt.tight_layout()
plt.show()


# PART 7 · 전체 모델 성능 종합 대시보드

print("전체 모델 성능 종합 대시보드")

fig = plt.figure(figsize=(18, 10))
fig.suptitle('Credit Card Fraud Detection — Full Pipeline Summary',
             fontsize=15, fontweight='bold', y=1.01)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# (0,0) 분류 F1 비교
ax00 = fig.add_subplot(gs[0, 0])
bar_c = ax00.bar(clf_models, clf_f1s,
                 color=['#95A5A6', '#3498DB', '#9B59B6', '#E74C3C'],
                 edgecolor='black', alpha=0.85)
ax00.set_title('Classification — F1 Score')
ax00.set_ylabel('F1')
ax00.set_ylim(0, 1.1)
for bar in bar_c:
    ax00.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
              f'{bar.get_height():.3f}', ha='center', fontsize=9)

# (0,1) 분류 ROC-AUC 비교
ax01 = fig.add_subplot(gs[0, 1])
bar_a = ax01.bar(clf_models, clf_aucs,
                 color=['#95A5A6', '#3498DB', '#9B59B6', '#E74C3C'],
                 edgecolor='black', alpha=0.85)
ax01.set_title('Classification — ROC-AUC')
ax01.set_ylabel('AUC')
ax01.set_ylim(0, 1.1)
for bar in bar_a:
    ax01.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
              f'{bar.get_height():.3f}', ha='center', fontsize=9)

# (0,2) 비지도 탐지 요약
ax02 = fig.add_subplot(gs[0, 2])
detect_methods = ['RANSAC\n(Outliers)', 'DBSCAN\n(Noise)']
detect_rates   = [fr_out, fr_noise]
base_rate      = overall_fr
colors_d = ['#F44336', '#FF9800']
ax02.bar(detect_methods, detect_rates, color=colors_d, edgecolor='black', alpha=0.85)
ax02.axhline(y=base_rate, color='blue', linestyle='--', linewidth=1.5,
             label=f'Overall ({base_rate:.2f}%)')
ax02.set_title('Unsupervised Anomaly Detection\nFraud Rate in Detected Outliers')
ax02.set_ylabel('Fraud Rate (%)')
ax02.legend(fontsize=8)
for i, (rate, method) in enumerate(zip(detect_rates, detect_methods)):
    ax02.text(i, rate + 0.05, f'{rate:.2f}%', ha='center', fontweight='bold')

# (1,0) 회귀 R² 비교
ax10 = fig.add_subplot(gs[1, 0])
ax10.barh(reg_results['Model'], reg_results['R²'],
          color=colors_reg, edgecolor='black', alpha=0.85)
ax10.set_xlabel('R²')
ax10.set_title('Regression — R² Score')
for i, v in enumerate(reg_results['R²']):
    ax10.text(v+0.001, i, f'{v:.4f}', va='center', fontsize=9)

# (1,1) 잔차 상위 5% 사기율 (잔차 분석 요약)
ax11 = fig.add_subplot(gs[1, 1])
pct_labels = ['전체 평균', '잔차 상위 20%', '잔차 상위 10%', '잔차 상위 5%']
pct_vals   = [overall_fr]
for pct in [20, 10, 5]:
    thr  = np.percentile(residuals, 100 - pct)
    mask = residuals >= thr
    pct_vals.append(np.mean(fraud_test[mask]) * 100)
bar_p = ax11.bar(pct_labels, pct_vals,
                 color=['#4CAF50', '#FFC107', '#FF9800', '#F44336'],
                 edgecolor='black', alpha=0.85)
ax11.set_title(f'Residual Analysis — {best_reg_name}')
ax11.set_ylabel('Fraud Rate (%)')
for bar in bar_p:
    ax11.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
              f'{bar.get_height():.2f}%', ha='center', fontsize=9)
ax11.tick_params(axis='x', labelsize=8)

# (1,2) 파이프라인 구조 텍스트 요약
ax12 = fig.add_subplot(gs[1, 2])
ax12.axis('off')
summary_text = (
    "■ PIPELINE SUMMARY\n\n"
    "PART1 EDA  →  Class Imbalance Check\n"
    "              Amount Distribution\n\n"
    "PART2 PCA  →  2D Projection\n"
    "              Linear Separability\n\n"
    "PART3 CLF  →  AdalineSGD (baseline)\n"
    "              Logistic Regression\n"
    "              SVM (rbf, balanced)\n"
    "              Random Forest ★\n\n"
    "PART4 REG  →  OLS / Ridge / Lasso\n"
    "              RANSAC (outlier tool)\n"
    f"              RF Regressor ★\n\n"
    "PART5 DBSCAN → Density Clustering\n\n"
    "PART6 RESIDUAL → Anomaly Signal\n\n"
    "PART7 DASHBOARD (this page)"
)
ax12.text(0.05, 0.95, summary_text,
          transform=ax12.transAxes,
          fontsize=9, verticalalignment='top',
          fontfamily='monospace',
          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.savefig('fraud_detection_summary.png', dpi=150, bbox_inches='tight')
plt.show()

# 결론

print("\n" + "=" * 60)
print("■ 최종 분석 결론")
print("=" * 60)

best_clf_f1  = max(clf_f1s)
best_clf_auc = max(clf_aucs)
best_clf_idx = np.argmax(clf_aucs)
print(f"\n[분류] 최고 모델: {clf_models[best_clf_idx].replace(chr(10),'')} "
      f"| F1={clf_f1s[best_clf_idx]:.4f} | AUC={best_clf_auc:.4f}")
print(f"[회귀] 최고 모델: {best_reg_name} | R²={best_reg_r2:.4f}")

top5_ratio = (np.mean(fraud_test[residuals >= np.percentile(residuals, 95)]) * 100 / max(overall_fr, 0.001))
print(f"\n[잔차 분석] 잔차 상위 5% 거래의 사기 비율은 전체 평균의 {top5_ratio:.1f}배")
print(f"[RANSAC]   Outlier 그룹의 사기 비율: {fr_out:.2f}% (Inlier: {fr_in:.2f}%)")
print(f"[DBSCAN]   Noise 포인트 사기 비율: {fr_noise:.2f}% (Cluster: {fr_cluster:.2f}%)")
print()
print("► 지도 학습(분류)으로 개별 거래의 사기 여부를 고정밀 탐지하고,")
print("  회귀 잔차·RANSAC·DBSCAN의 비지도 신호를 결합하면")
print("  실제 금융 환경의 '앙상블 이상 탐지 시스템' 구축이 가능합니다.")


# ╔══════════════════════════════════════════════════════════╗
# ║  [추가] PART 8 · 피드백 반영 종합 — Recall 중심 모델 비교 ║
# ╚══════════════════════════════════════════════════════════╝
print("\n" + "=" * 60)
print("[추가] PART 8 · 피드백 반영 종합 비교 (지도 + 비지도, Recall 중심)")
print("=" * 60)

final_compare = pd.DataFrame({
    'Model': [m.replace('\n', ' ') for m in clf_models] + list(unsup_df['Model']),
    'Type':  ['Supervised'] * len(clf_models) + ['Unsupervised'] * len(unsup_df),
    'Precision': clf_precisions + list(unsup_df['Precision']),
    'Recall':    clf_recalls    + list(unsup_df['Recall']),
})
print(final_compare.to_string(index=False))

fig, ax = plt.subplots(figsize=(11, 5))
colors_final = ['#3498DB' if t == 'Supervised' else '#F39C12' for t in final_compare['Type']]
x_f = np.arange(len(final_compare))
ax.bar(x_f, final_compare['Recall'], color=colors_final, edgecolor='black', alpha=0.85)
ax.set_xticks(x_f)
ax.set_xticklabels(final_compare['Model'], rotation=30, ha='right')
ax.set_ylim(0, 1.15)
ax.set_ylabel('Recall')
ax.set_title('[추가] 전체 모델 Recall 비교 — 지도학습(파랑) vs 비지도학습(주황)\n(Recall = 실제 사기를 얼마나 놓치지 않고 탐지했는가)')
mean_recall = float(np.mean(final_compare['Recall']))
ax.axhline(y=mean_recall, color='gray', linestyle='--', linewidth=1,
           label=f'평균 Recall ({mean_recall:.3f})')
ax.legend()
for i, v in enumerate(final_compare['Recall']):
    ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=8)
plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("■ [추가] 피드백 반영 최종 결론")
print("=" * 60)
print("""
1) Recall 중심 평가
   - 사기 탐지 시스템의 핵심은 '실제 사기 거래를 놓치지 않는 것(Recall)'.
   - Precision과 Recall을 함께 제시하여, 모델이 사기를 '탐지했는지 안했는지'를
     F1/AUC 한 줄 요약이 아닌 실제 탐지 건수(TP)·미탐지 건수(FN)로 직접 확인 가능하도록 보완.

2) 분리 기준 명확화
   - 기존 V1~V28 중요도 랭킹은 변수 자체가 PCA로 익명화되어 의미 해석이 어려움.
   - 상위 변수별 정상/사기 그룹의 실제 값 분포(Box Plot)와 구체적 분리 임계값(threshold)을
     제시하여 "어떤 값일 때 사기로 의심되는가"를 명확히 보완 (PART 3-5).

3) 비지도 이상 탐지 기법 확장
   - 기존 DBSCAN에 더해 Isolation Forest, Local Outlier Factor, One-Class SVM을 추가하여
     레이블 없이도 이상 거래를 탐지하는 Anomaly Detection 접근을 보강 (PART 5-1).
   - 비지도 기법은 신규/미지의 사기 패턴에도 대응 가능하다는 장점이 있으며,
     지도 학습 모델과 함께 앙상블 신호로 활용 시 Recall 보완 효과를 기대할 수 있음.
""")