import os
import numpy as np
import argparse

def extract_pssm(filepath):
    vectors = []
    with open(filepath, "r") as f:
        for line in f:
            if line.strip() == "":
                continue
            parts = line.strip().split()
            if len(parts) < 22:
                continue
            if parts[0].isdigit():
                vectors.append([int(x) for x in parts[2:22]])  # 20维
    return np.array(vectors)  # shape: (L, 20)

def extract_hhm(filepath):
    vectors = []
    with open(filepath, "r") as f:
        lines = f.readlines()

    p = 0
    while lines[p][0] != "#":
        p += 1
    p += 5

    for i in range(p, len(lines), 3):
        if lines[i].strip() == "//":
            continue
        record = lines[i].strip().split()[2:-1]
        row = []
        for x in record:
            if x == "*":
                row.append(9999)
            else:
                row.append(int(x))
        vectors.append(row)
    return np.array(vectors)

def main(pssm_dir, hhm_dir):
    all_pssm = []
    all_hhm = []

    for fname in os.listdir(pssm_dir):
        if fname.endswith(".pssm"):
            fpath = os.path.join(pssm_dir, fname)
            vec = extract_pssm(fpath)
            all_pssm.append(vec)

    for fname in os.listdir(hhm_dir):
        if fname.endswith(".hhm"):
            fpath = os.path.join(hhm_dir, fname)
            vec = extract_hhm(fpath)
            all_hhm.append(vec)

    all_pssm = np.vstack(all_pssm)  # shape: (N, 20)
    all_hhm = np.vstack(all_hhm)

    print("\n=== PSSM统计 ===")
    print("Max_pssm = np.array({})".format(np.max(all_pssm, axis=0).tolist()))
    print("Min_pssm = np.array({})".format(np.min(all_pssm, axis=0).tolist()))

    print("\n=== HHM统计 ===")
    print("Max_hhm = np.array({})".format(np.max(all_hhm, axis=0).tolist()))
    print("Min_hhm = np.array({})".format(np.min(all_hhm, axis=0).tolist()))

    # 可选保存为 npy
    np.save("Max_pssm.npy", np.max(all_pssm, axis=0))
    np.save("Min_pssm.npy", np.min(all_pssm, axis=0))
    np.save("Max_hhm.npy", np.max(all_hhm, axis=0))
    np.save("Min_hhm.npy", np.min(all_hhm, axis=0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pssm_dir", type=str, required=True, help="Directory containing .pssm files")
    parser.add_argument("--hhm_dir", type=str, required=True, help="Directory containing .hhm files")
    args = parser.parse_args()

    main(args.pssm_dir, args.hhm_dir)

#Train_set
# === PSSM统计 ===
# Max_pssm = np.array([8, 9, 9, 9, 12, 10, 8, 8, 12, 8, 7, 9, 11, 10, 9, 8, 8, 13, 10, 8])
# Min_pssm = np.array([-11, -11, -12, -13, -11, -11, -11, -11, -11, -11, -11, -11, -10, -11, -12, -11, -10, -11, -11, -11])
#
# === HHM统计 ===
# Max_hhm = np.array([11442, 12364, 11979, 12226, 12412, 11829, 12342, 11964, 11834, 12074, 12525, 12637, 12164, 12709, 11997, 11616, 11411, 11396, 12915, 12129])
# Min_hhm = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

#Test_set
# === PSSM统计 ===
# Max_pssm = np.array([7, 9, 9, 9, 12, 9, 8, 8, 12, 8, 7, 8, 11, 9, 9, 7, 8, 13, 10, 8])
# Min_pssm = np.array([-12, -13, -13, -14, -12, -12, -13, -13, -11, -12, -12, -13, -11, -11, -13, -13, -12, -12, -11, -13])
#
# === HHM统计 ===
# Max_hhm = np.array([10470, 12693, 11997, 11561, 11666, 11543, 11990, 11993, 11821, 11583, 11633, 11948, 12136, 11944, 12088, 11484, 11274, 11584, 12559, 12494])
# Min_hhm = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

