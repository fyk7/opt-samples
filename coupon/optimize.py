# !pip install pulp
import pandas as pd
import pulp

cust_df = pd.read_csv('customers.csv')
prob_df = pd.read_csv('visit_probability.csv')

problem = pulp.LpProblem(name='DiscountCouponProblem1',sense=pulp.LpMaximize)

# 会員IDのリスト
I = cust_df['customer_id'].to_list()
# ダイレクトメールのパターンのリスト
M = [1, 2, 3]
# （1）各会員に対してどのパターンのダイレクトメールを送付するかを決定
xim = {}
for i in I:
    for m in M:
        xim[i,m] = pulp.LpVariable(name=f'xim({i},{m})',cat='Binary')


#　（2）各会員に対して送付するダイレクトメールはいずれか1パターン
for i in I:
    problem += pulp.lpSum(xim[i,m] for m in M) == 1


keys = ['age_cat', 'freq_cat']
cust_prob_df = pd.merge(cust_df, prob_df, on=keys)
cust_prob_ver_df = cust_prob_df.rename(
    columns={'prob_dm1': 1, 'prob_dm2': 2, 'prob_dm3': 3}
    ).melt(
        id_vars=['customer_id'],
        value_vars=[1,2,3],
        var_name='dm',
        value_name='prob'
    )
Pim = cust_prob_ver_df.set_index(['customer_id','dm'])['prob'].to_dict()
# （3）クーポン付与による来客増加数を最大化
problem += pulp.lpSum(
    (Pim[i,m] - Pim[i,1]) * xim[i,m] for i in I for m in [2,3])


# （4）顧客の消費する費用の期待値の合計は100万円以下
Cm = {1:0, 2:1000, 3:2000}
problem += pulp.lpSum(
    Cm[m] * Pim[i,m]* xim[i,m] for i in I for m in [2,3]) <= 1000000


# セグメントのリスト
S = prob_df['segment_id'].to_list()
# 各セグメントとそのセグメントに属する顧客数を対応させる辞書の作成
Ns = cust_prob_df.groupby('segment_id')['customer_id'].count().to_dict()
# 会員をキーとして属するセグメントを返す辞書
Si = cust_prob_df.set_index('customer_id')['segment_id'].to_dict()
# （5）各パターンのダイレクトメールをそれぞれのセグメントに属する会員数の10%以上送付
for s in S:
    for m in M:
        problem += pulp.lpSum(
            xim[i,m] for i in I if Si[i] == s) >= 0.1 * Ns[s]


# 時間を計測
# モデリング1は、一部の環境ではgapRel（計算の終了判定とする上界と下界のギャップのしきい値）を指定しないと停止しない
# solver = pulp.PULP_CBC_CMD(gapRel=10e-4)
import time
time_start = time.time()
status = problem.solve()
# gapRelを指定した場合はsolve関数にて上でパラメータを指定したsolverを引数にとる
# status = problem.solve(solver)
time_stop = time.time()



# 結果表示
print(f'ステータス:{pulp.LpStatus[status]}')
print(f'目的関数値:{pulp.value(problem.objective):.4}')
print(f'計算時間:{(time_stop - time_start):.3}(秒)')

send_dm_df = pd.DataFrame(
    [[xim[i,m].value() for m in M] for i in I],
    columns=['send_dm1', 'send_dm2', 'send_dm3'])
send_dm_df.head()

cust_send_df = pd.concat(
    [cust_df[['customer_id', 'age_cat', 'freq_cat']], send_dm_df], axis=1)
cust_send_df.head()
