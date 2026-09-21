import random
import math

N = 32
d_model = 16


def softmax_stable(S):
    m = max(S)
    exp_S = [math.exp(S[i] - m) for i in range(N)]
    sum_exp_S = sum(exp_S)
    P = [v / sum_exp_S for v in exp_S]
    return P


def softmax_online(S):
    m = -math.inf
    l = 0
    P = []
    for s in S:
        m_old = m
        m = max(m, s)
        l_old = l
        l = l * math.exp(m_old - m) + math.exp(s - m)
        # renormalize
        for i in range(len(P)):
            P[i] = P[i] * l_old * math.exp(m_old - m) / l

        # append new
        P.append(math.exp(s - m) / l)
    return P


def softmaxV_online(S, V):
    m = -math.inf
    l = 0
    O = [0 for _ in range(d_model)]
    for i, s in enumerate(S):
        m_old = m
        m = max(m, s)
        l_old = l
        l = l * math.exp(m_old - m) + math.exp(s - m)

        for j in range(d_model):
            # renormalized value + new value
            O[j] = O[j] * l_old * math.exp(m_old - m) / l + math.exp(s - m) / l * V[i][j]

    return O


def softmaxV2_online(S, V):
    m = -math.inf
    l = 0
    O = [0 for _ in range(d_model)]
    for i, s in enumerate(S):
        m_old = m
        m = max(m, s)
        l = l * math.exp(m_old - m) + math.exp(s - m)

        for j in range(d_model):
            # renormalized value + new value
            O[j] = O[j] * math.exp(m_old - m) + math.exp(s - m) * V[i][j]

    for j in range(d_model):
        O[j] = O[j] / l

    return O


S = [random.normalvariate() for _ in range(N)]

P_stable = softmax_stable(S)
P_online = softmax_online(S)
diff = sum([math.fabs(P_stable[i] - P_online[i]) for i in range(N)])
print(diff)

V = [[random.normalvariate() for _ in range(d_model)] for _ in range(N)]
SV = []
for i in range(d_model):
    SV.append(sum([P_stable[j] * V[j][i] for j in range(N)]))

SV_online = softmaxV_online(S, V)
SV2_online = softmaxV2_online(S, V)
diff = sum([math.fabs(SV[i] - SV_online[i]) for i in range(d_model)])
diff2 = sum([math.fabs(SV[i] - SV2_online[i]) for i in range(d_model)])

print(diff, diff2)
