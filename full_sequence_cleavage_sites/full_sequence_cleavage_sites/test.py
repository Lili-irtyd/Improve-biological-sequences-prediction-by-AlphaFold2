import gzip
import shutil

# 输入和输出文件路径
input_file = "/media/gpux1/Pro1/app/intfold/2019_ccd_filtered.tab"
output_file = "/media/gpux1/Pro1/app/intfold/2019_ccd_filtered.tab.gz"

# 压缩文件
with open(input_file, 'rb') as f_in:
    with gzip.open(output_file, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

print(f"文件已成功压缩为 {output_file}")
