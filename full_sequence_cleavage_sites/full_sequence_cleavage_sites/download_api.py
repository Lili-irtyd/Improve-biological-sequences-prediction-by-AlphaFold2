###download tab file for the sepreate process of intfold.py

import requests

# 读取蛋白质名称
with open('../data analysis/2019_cleavage_protein_list.txt', 'r') as f:
    query = ' OR '.join([line.strip() for line in f])

# 设置 UniProt 查询 URL
url = f"https://rest.uniprot.org/uniprotkb/search?query=({query})&format=tsv"

# 下载结果
response = requests.get(url)

# 保存到文件
with open("../data analysis/2019_cleavage_protein_list.tab", "wb") as f:
    f.write(response.content)
