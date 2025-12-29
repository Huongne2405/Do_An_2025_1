import numpy as np
import math

# Cấu hình in ấn
np.set_printoptions(suppress=True, linewidth=200, precision=4, edgeitems=5)

def Matrix_Companion(Data):
    n = len(Data)
    A = np.zeros((n, n))
    A[0, :] = -np.array(Data)
    for i in range(1, n):
        A[i, i - 1] = 1
    return A

def Chuan_Hoa_L2(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-12 else v

def Giai_Phuong_Trinh_Bac_2(a, b, c):
    delta = b**2 - 4*a*c
    if delta >= 0:
        z1 = (-b + math.sqrt(delta)) / (2*a)
        z2 = (-b - math.sqrt(delta)) / (2*a)
        return z1, z2
    else:
        real = -b / (2*a)
        return real, real

def Xu_li(A, Y, E=1e-8):
    B_list = [Chuan_Hoa_L2(Y)]
    m = 1
    TH = 3 # Mặc định TH3 nếu quá 500 vòng
    max_iter = 500
    while m < max_iter:
        vec_next = A.dot(B_list[m-1])
        B_list.append(Chuan_Hoa_L2(vec_next))
        
        if m > 1 and np.max(np.abs(B_list[m] - B_list[m-1])) <= E:
            TH = 1
            break
        elif m > 2 and np.max(np.abs(B_list[m] - B_list[m-2])) <= E:
            TH = 2
            break
        m += 1
    
    for _ in range(2):
        B_list.append(Chuan_Hoa_L2(A.dot(B_list[-1])))
        
    return m, TH, B_list

def Xuong_thang(A, lamda, v):
    v = v.reshape(-1, 1)
    y = v / np.dot(v.T, v).item()
    B = A - lamda * np.outer(v, y.T)
    return B

def thuc_hien(Initial_Data):
    n = len(Initial_Data)
    A_curr = Matrix_Companion(Initial_Data)
    Q = []
    k = 0
    
    while k < n:
        np.random.seed(k)
        Y = np.random.rand(n, 1)
        m, TH, B_all = Xu_li(A_curr, Y)
        
        current_step_results = []
        if TH == 1:
            v_f = B_all[m]
            idx = np.argmax(np.abs(v_f))
            lamda = (A_curr.dot(v_f)[idx] / v_f[idx]).item()
            current_step_results.append((lamda, v_f))
        elif TH == 2:
            AmY = B_all[m]
            Am_2Y = B_all[m-2]
            i_idx = np.argmax(np.abs(Am_2Y))
            ti_so = AmY[i_idx] / Am_2Y[i_idx]
            l1 = math.sqrt(abs(ti_so))
            l2 = -l1
            v_f_1 = (AmY + l1 * Am_2Y) / np.max(np.abs(AmY + l1 * Am_2Y))
            v_f_2 = (AmY + l2 * Am_2Y) / np.max(np.abs(AmY + l2 * Am_2Y))
            current_step_results.append((l1, v_f_1))
            current_step_results.append((l2, v_f_2))
        elif TH == 3:
            v_k_2, v_k_1, v_k = B_all[m-2], B_all[m-1], B_all[m]
            v_kp1 = A_curr.dot(v_k)
            i = 0
            a = v_k_2[i]*v_k_1[i+1] - v_k_2[i+1]*v_k_1[i]
            b = v_k_2[i]*v_k[i+1] - v_k_2[i+1]*v_k[i]
            c = v_k_1[i]*v_k[i+1] - v_kp1[i+1]*v_k[i]
            l1, l2 = Giai_Phuong_Trinh_Bac_2(a, b, c)
            v_f_1 = Chuan_Hoa_L2(A_curr.dot(v_k) - l2 * v_k)
            v_f_2 = Chuan_Hoa_L2(A_curr.dot(v_k) - l1 * v_k)
            current_step_results.append((l1, v_f_1))
            current_step_results.append((l2, v_f_2))
        print("-"*100)
        print(f"\n5 vector quanh mốc hội tụ  (m={m})")
        for i in range(max(0, m-2), min(len(B_all), m+3)):
            tag = " <--- MỐC HỘI TỤ" if i == m else ""
            print(f"Bước {i:3}{tag}: {B_all[i].flatten()}")

        for lam_val, vec_val in current_step_results:
            if k < n:
                Q.append(lam_val)
                print(f"Trị riêng: {lam_val:.8f}")
                print(f"Vector riêng tương ứng:\n{vec_val.flatten()}")
                print(f"Trạng thái hội tụ: TH{TH}")
                print(f"Số lần lặp: {m}")
                if k < n - 1:
                    A_curr = Xuong_thang(A_curr, lam_val, vec_val)
                    print(f"\n Ma trận xuống thang lần {k+1}:")
                    print(A_curr)
                k += 1

    print("Danh sách các nghiệm:", np.sort(np.round(Q, 4)))


Data = [-36, 546, -4536, 22449, -67284, 118124, -109584, 40320]
#Data = [-6,11,-6]
#Data = [-18,99,-162]
thuc_hien(Data)