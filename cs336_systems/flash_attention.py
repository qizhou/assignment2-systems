import random
import math


seq_len = 128
d_model = 32

Q = [[random.normalvariate() for _ in range(d_model)] for _ in range(seq_len)]
K = [[random.normalvariate() for _ in range(d_model)] for _ in range(seq_len)]
V = [[random.normalvariate() for _ in range(d_model)] for _ in range(seq_len)]


def standard_attention(Q, K, V):
    # S = Q * K'
    S = [[] for _ in range(seq_len)] # size of seq_len x seq_len
    for row in range(seq_len):
        for col in range(seq_len):
            # Given query row, obtains all lookups without mask
            v = sum([Q[row][i] * K[col][i] for i in range(d_model)])
            S[row].append(v)

    # softmax in rows
    P = []
    for row in range(seq_len):
        m = max(S[row])
        stable_exp = [math.exp(S[row][i] - m) for i in range(seq_len)]
        sum_stable_exp = sum(stable_exp)
        P.append([v / sum_stable_exp for v in stable_exp])

    # output O = PV
    O = [[] for _ in range(seq_len)]
    for row in range(seq_len):
        for col in range(d_model):
            O[row].append(sum([P[row][i] * V[i][col] for i in range(seq_len)]))

    return O


# def flash_attention(Q, K, V):
#     B_c = 4 # TODO
#     B_r = 4 # TODO
#     T_r = seq_len // B_r # Q in [Q_1, Q_2, .. Q_Tr]
#     T_c = seq_len // T_c # divide K, V

#     O = [[0 for _ in range(d_model)] for _ in range(seq_len)]
#     l = [0 for _ in range(seq_len)]
#     m = [-math.inf for _ in range(seq_len)]

#     for j in range(T_c):
#         # load K_j
#         Kj = [K[j*B_c+k] for k in range(B_c)]
#         Vj = [V[j*B_c+k] for k in range(B_c)]
#         for i in range(T_r):
#             # load Q_i, O_i, l_i, m_i
#             Qi = [Q[i*B_r+k] for k in range(B_r)]
#             Oi = [O[i*B_r+k] for k in range(B_r)]

#             # Evaluate query j, i

def flash_attention_simplified(Q, K, V):
    O = [[0 for _ in range(d_model)] for _ in range(seq_len)]
    l = [0 for _ in range(seq_len)]
    m = [-math.inf for _ in range(seq_len)]

    for j in range(seq_len): # key j
        for i in range(seq_len): # query i
            # Evaluate query i, key j
            s = sum([Q[i][k] * K[j][k] for k in range(d_model)])
            m_old = m[i]
            m[i] = max(m[i], s)
            l_old = l[i]
            l[i] = l[i] * math.exp(m_old - m[i]) + math.exp(s - m[i])

            for k in range(d_model):
                O[i][k] = O[i][k] * l_old * math.exp(m_old - m[i]) / l[i] + math.exp(s - m[i]) / l[i] * V[j][k]
    return O

def flash_attention2_simplified(Q, K, V):
    O = [[0 for _ in range(d_model)] for _ in range(seq_len)]

    for i in range(seq_len): # query i
        m = -math.inf
        l = 0
        for j in range(seq_len): # key j
            # Evaluate query i, key j
            s = sum([Q[i][k] * K[j][k] for k in range(d_model)])
            m_old = m
            m = max(m, s)
            l = l * math.exp(m_old - m) + math.exp(s - m)

            for k in range(d_model):
                O[i][k] = O[i][k] * math.exp(m_old - m) + math.exp(s - m) * V[j][k]

        for k in range(d_model):
            O[i][k] = O[i][k]  / l

    return O


O_standard = standard_attention(Q, K, V)
O_flash = flash_attention_simplified(Q, K, V)
O_flash2 = flash_attention2_simplified(Q, K, V)

diff = sum([sum([math.fabs(O_standard[i][j] - O_flash[i][j]) for j in range(d_model)]) for i in range(seq_len)])
diff2 = sum([sum([math.fabs(O_standard[i][j] - O_flash2[i][j]) for j in range(d_model)]) for i in range(seq_len)])

print(diff, diff2)