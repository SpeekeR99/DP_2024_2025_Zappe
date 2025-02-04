import os
import sys
import time
import gzip
import bz2
import lzma
import zstandard as zstd


def compress_gzip(input_file, output_file):
    start_time = time.time()
    with open(input_file, "rb") as f_in, gzip.open(output_file, "wb") as f_out:
        f_out.writelines(f_in)
    end_time = time.time()
    return end_time - start_time, os.path.getsize(output_file)


def decompress_gzip(input_file, output_file):
    start_time = time.time()
    with gzip.open(input_file, "rb") as f_in, open(output_file, "wb") as f_out:
        f_out.writelines(f_in)
    end_time = time.time()
    return end_time - start_time


def compress_bzip2(input_file, output_file):
    start_time = time.time()
    with open(input_file, "rb") as f_in, bz2.open(output_file, "wb") as f_out:
        f_out.writelines(f_in)
    end_time = time.time()
    return end_time - start_time, os.path.getsize(output_file)


def decompress_bzip2(input_file, output_file):
    start_time = time.time()
    with bz2.open(input_file, "rb") as f_in, open(output_file, "wb") as f_out:
        f_out.writelines(f_in)
    end_time = time.time()
    return end_time - start_time


def compress_xz(input_file, output_file):
    start_time = time.time()
    with open(input_file, "rb") as f_in, lzma.open(output_file, "wb") as f_out:
        f_out.writelines(f_in)
    end_time = time.time()
    return end_time - start_time, os.path.getsize(output_file)


def compress_xz_1(input_file, output_file, preset=1):
    start_time = time.time()
    with open(input_file, "rb") as f_in, lzma.open(output_file, "wb", preset=preset) as f_out:
        f_out.writelines(f_in)
    end_time = time.time()
    return end_time - start_time, os.path.getsize(output_file)


def compress_xz_9(input_file, output_file, preset=9):
    start_time = time.time()
    with open(input_file, "rb") as f_in, lzma.open(output_file, "wb", preset=preset) as f_out:
        f_out.writelines(f_in)
    end_time = time.time()
    return end_time - start_time, os.path.getsize(output_file)


def decompress_xz(input_file, output_file):
    start_time = time.time()
    with lzma.open(input_file, "rb") as f_in, open(output_file, "wb") as f_out:
        f_out.writelines(f_in)
    end_time = time.time()
    return end_time - start_time


def compress_zstd(input_file, output_file):
    start_time = time.time()
    cctx = zstd.ZstdCompressor()
    with open(input_file, "rb") as f_in, open(output_file, "wb") as f_out:
        f_out.write(cctx.compress(f_in.read()))
    end_time = time.time()
    return end_time - start_time, os.path.getsize(output_file)


def decompress_zstd(input_file, output_file):
    start_time = time.time()
    dctx = zstd.ZstdDecompressor()
    with open(input_file, "rb") as f_in, open(output_file, "wb") as f_out:
        f_out.write(dctx.decompress(f_in.read()))
    end_time = time.time()
    return end_time - start_time


def compress_zstd_19_T0(input_file, output_file, compression_level=19, num_threads=-1):
    start_time = time.time()
    cctx = zstd.ZstdCompressor(level=compression_level, threads=num_threads)
    with open(input_file, "rb") as f_in, open(output_file, "wb") as f_out:
        f_out.write(cctx.compress(f_in.read()))
    end_time = time.time()
    return end_time - start_time, os.path.getsize(output_file)


def compare_compression_methods(input_file):
    original_size = os.path.getsize(input_file)
    methods_compression = {
        "Gzip": compress_gzip,
        "Bzip2": compress_bzip2,
        "XZ": compress_xz,
        "XZ_1": compress_xz_1,
        "XZ_9": compress_xz_9,
        "Zstandard": compress_zstd,
        "Zstandard_19_T0": compress_zstd_19_T0,
    }
    methods_decompression = {
        "Gzip": decompress_gzip,
        "Bzip2": decompress_bzip2,
        "XZ": decompress_xz,
        "XZ_1": decompress_xz,
        "XZ_9": decompress_xz,
        "Zstandard": decompress_zstd,
        "Zstandard_19_T0": decompress_zstd,
    }

    temp_files = []
    results = {}

    for method, func in methods_compression.items():
        output_file = f"{input_file}.{method.lower()}"
        time_taken, compressed_size = func(input_file, output_file)
        compression_ratio = original_size / compressed_size
        results[method] = [time_taken, compression_ratio]
        temp_files.append(output_file)

    for method, func in methods_decompression.items():
        input_file_method = f"{input_file}.{method.lower()}"
        output_file = f"{input_file}_{method.lower()}_decompressed.csv"
        time_taken = func(input_file_method, output_file)
        results[method].append(time_taken)
        temp_files.append(output_file)

    # Cleanup
    for file in temp_files:
        os.remove(file)

    return results


if __name__ == "__main__":
    input_file = sys.argv[1]
    results = compare_compression_methods(input_file)
    print("Method, Compression Time (s), Compression Ratio, Decompression Time (s)")
    for method, result in results.items():
        print(f"{method}, {result[0]}, {result[1]}, {result[2]}")
