# !pip install pulp
import pandas as pd
import pulp

cust_df = pd.read_csv('customers.csv')
prob_df = pd.read_csv('visit_probability.csv')


# 数理モデルのインスタンス作成
problem = pulp.LpProblem(name='DiscountCouponProblem2', sense=pulp.LpMaximize)


# セグメントのリスト
S = prob_df['segment_id'].to_list()
M = [1, 2, 3]
# （1）各会員に対してどのパターンのダイレクトメールを送付するかを決定
xsm = {}
# ［0,1］の変数を宣言
for s in S:
    for m in M:
        xsm[s,m] = pulp.LpVariable(
            name=f'xsm({s},{m})',
            lowBound=0, upBound=1, cat='Continuous')


# （2）各会員に対して送付するダイレクトメールはいずれか1パターン
for s in S:
    problem += pulp.lpSum(xsm[s,m] for m in M) == 1


prob_ver_df = prob_df.rename(
    columns={'prob_dm1': 1,'prob_dm2': 2, 'prob_dm3': 3}
    ).melt(
        id_vars=['segment_id'],
        value_vars=[1,2,3],
        var_name='dm',
        value_name='prob'
    )
Psm = prob_ver_df.set_index(['segment_id','dm'])['prob'].to_dict()
keys = ['age_cat', 'freq_cat']
cust_prob_df = pd.merge(cust_df, prob_df, on=keys)
# 各セグメントとそのセグメントに属する顧客数を対応させる辞書の作成
Ns = cust_prob_df.groupby('segment_id')['customer_id'].count().to_dict()
# （3）クーポン付与による来客増加数を最大化
problem += pulp.lpSum(
    Ns[s] * (Psm[s,m] - Psm[s,1]) * xsm[s,m]
    for s in S for m in [2,3]
)


# （4）会員の予算消費期待値の合計は100万円以下
Cm = {1:0, 2:1000, 3:2000}
problem += pulp.lpSum(
    Cm[m] * Ns[s] * Psm[s,m] * xsm[s,m]
    for s in S for m in [2,3]
) <= 1000000


# （5）各パターンのダイレクトメールをそれぞれのセグメントに属する会員数の10%以上送付
for s in S:
    for m in M:
        problem += xsm[s,m] >= 0.1
import time
time_start = time.time()
status = problem.solve()
time_stop = time.time()
print(f'ステータス:{pulp.LpStatus[status]}')
print(f'目的関数値:{pulp.value(problem.objective):.4}')
print(f'計算時間:{(time_stop - time_start):.3}(秒)')
