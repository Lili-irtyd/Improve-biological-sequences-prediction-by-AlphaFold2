# import os
#
# def parse_args():
#
#     import argparse
#
#     parser = argparse.ArgumentParser(
#         description='batch generate intermediate representations'
#     )
#     parser.add_argument('--data-dir', default='/media/gpux1/Pro1/DOWNLOAD_DIR', type=str, help='absolute path to alphafold data')
#     #peptides
#     # parser.add_argument('--fasta-path', default='/media/gpux1/Pro1/negative_fasta/B0JZF8', type=str, help='absolute path to folder include fasta files')
#     # parser.add_argument('--out-dir', default='/media/gpux1/Pro1/negative_file/B0JZF8', type=str, help='absolute output path')
#
#     #full sequences
#     parser.add_argument('--fasta-path', default='/media/gpux1/Pro1/RNA-495_Train', type=str, help='absolute path to folder include fasta files')
#     parser.add_argument('--out-dir', default='/media/gpux1/Pro1/output_full_sequences', type=str, help='absolute output path')
#
#     args = parser.parse_args()
#
#     return args
#
# def main(args):
#
#     print(args)
#
#     fasta_list = [os.path.join(args.fasta_path, fasta) for fasta in os.listdir(args.fasta_path)]
#     fasta_paths = ','.join(fasta_list)
#     if not os.path.exists(args.out_dir):
#         os.makedirs(args.out_dir)
#
#     cmd = 'python docker/run_docker.py \
#     --fasta_paths={} \
#     --max_template_date=2020-05-14 \
#     --db_preset=reduced_dbs \
#     --data_dir={} \
#     --output_dir={} \
#     --docker_image_name=alphafold-test'.format(fasta_paths, args.data_dir, args.out_dir)
#
#
# ##原本是--max_template_date=2021-11-01 \
#     os.system(cmd)
#
# if __name__ == '__main__':
#
#     args = parse_args()
#     main(args)



# def parse_args():
#     parser = argparse.ArgumentParser(
#         description='batch generate intermediate representations for two datasets'
#     )
#     parser.add_argument('--data-dir', default='/media/gpux1/Pro1/DOWNLOAD_DIR', type=str, help='absolute path to alphafold data')
#     parser.add_argument('--train-fasta-path', default='/media/gpux1/Pro1/RNA-495_Train_fasta_new_1_196', type=str, help='absolute path to train fasta folder')
#     parser.add_argument('--train-out-dir', default='/media/gpux1/Pro1/new_output_sequence_train', type=str, help='absolute output path for train set')
#     # parser.add_argument('--test-fasta-path', default='/media/gpux1/Pro1/RNA-117_Test_fasta', type=str, help='absolute path to test fasta folder')
#     # parser.add_argument('--test-out-dir', default='/media/gpux1/Pro1/output_full_sequences_test', type=str, help='absolute output path for test set')
#     args = parser.parse_args()
#     return args
#
#
# def run_alphafold(fasta_dir, output_dir, data_dir):
#     fasta_list = [os.path.join(fasta_dir, fasta) for fasta in os.listdir(fasta_dir)]
#     fasta_paths = ','.join(fasta_list)
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     cmd = 'python docker/run_docker.py \
#     --fasta_paths="{}" \
#     --max_template_date=2020-05-14 \
#     --db_preset=reduced_dbs \
#     --data_dir="{}" \
#     --output_dir="{}" \
#     --docker_image_name=alphafold-test'.format(fasta_paths, data_dir, output_dir)
#
#     print(f"⚡ Running AlphaFold for {fasta_dir}...")
#     os.system(cmd)
#
#
# def main(args):
#     print("🚀 Starting processing...")
#
#     # 处理训练集
#     run_alphafold(args.train_fasta_path, args.train_out_dir, args.data_dir)
#
#     # 处理测试集
#     #run_alphafold(args.test_fasta_path, args.test_out_dir, args.data_dir)
#
#     print("✅ All processing finished!")
#
#
# if __name__ == '__main__':
#     args = parse_args()
#     main(args)

import os
import argparse
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(
        description='batch generate intermediate representations for fasta folder'
    )
    parser.add_argument('--data-dir', default='/media/gpux1/Pro1/DOWNLOAD_DIR', type=str, help='absolute path to alphafold data')
    # parser.add_argument('--data-dir', default='/usr/appli/freeware/AlphaFold/2.3.2/data', type=str,
    #                     help='absolute path to alphafold data')
    parser.add_argument('--train-fasta-path', default='/media/gpux1/Pro1/GraphBind/Datasets/PHEM/fasta_files_train', type=str,
                        help='absolute path to train fasta folder')
    parser.add_argument('--train-out-dir', default='/media/gpux1/Pro1/GraphBind/Datasets/PHEM/AF2_train', type=str,
                        help='absolute output path for train set')
    return parser.parse_args()


def run_alphafold_single(fasta_file, output_dir, data_dir):
    base_name = os.path.splitext(os.path.basename(fasta_file))[0]
    output_subdir = os.path.join(output_dir, base_name)
    os.makedirs(output_subdir, exist_ok=True)

    cmd = [
        "python", "docker/run_docker.py",
        f"--fasta_paths={fasta_file}",
        "--max_template_date=2020-05-14",
        "--db_preset=reduced_dbs",
        f"--data_dir={data_dir}",
        f"--output_dir={output_subdir}",
        "--docker_image_name=alphafold-test"
    ]

    print(f"⚡ Running AlphaFold for {fasta_file}...")
    try:
        result = subprocess.run(cmd, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error processing {fasta_file}:\n{e.stderr}")
        print("➡️ Skipping to next file...\n")


def run_alphafold_batch(fasta_dir, output_dir, data_dir):
    fasta_list = sorted([os.path.join(fasta_dir, f) for f in os.listdir(fasta_dir) if f.endswith(".fasta")])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for fasta_file in fasta_list:
        run_alphafold_single(fasta_file, output_dir, data_dir)


def main(args):
    print("🚀 Starting AlphaFold batch prediction...")
    run_alphafold_batch(args.train_fasta_path, args.train_out_dir, args.data_dir)
    print("✅ All processing finished!")


if __name__ == '__main__':
    args = parse_args()
    main(args)
