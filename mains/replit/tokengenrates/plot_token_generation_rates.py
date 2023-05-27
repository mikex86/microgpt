import re
import matplotlib.pyplot as plt

if __name__ == '__main__':
    file_path = "token_generation_rates.txt"
    gpu_throughputs = {}

    with open(file_path, "r") as file:
        file_content = file.read()

    gpu_blocks = re.findall(r"# (.*?)\n(.*?)\n\n", file_content, re.DOTALL)

    for gpu_block in gpu_blocks:
        gpu_name = gpu_block[0]
        gpu_data = gpu_block[1]

        gpu_throughputs[gpu_name] = {"batch_sizes": [], "throughputs": []}

        batch_sizes = re.findall(r"Batch size: (\d+)", gpu_data)
        throughputs = re.findall(r"throughput: ([\d.]+) tokens/s", gpu_data)

        gpu_throughputs[gpu_name]["batch_sizes"] = [int(bs) for bs in batch_sizes]
        gpu_throughputs[gpu_name]["throughputs"] = [float(tp) for tp in throughputs]

    for gpu_name, gpu_data in gpu_throughputs.items():
        batch_sizes = gpu_data["batch_sizes"]
        throughputs = gpu_data["throughputs"]

        plt.plot(batch_sizes, throughputs, marker="o", label=gpu_name)

        max_throughput = max(throughputs)
        max_throughput_batch_size = batch_sizes[throughputs.index(max_throughput)]
        print(f"Highest throughput for {gpu_name}: {max_throughput} tokens/s at batch size {max_throughput_batch_size}")

    plt.xlabel("Batch Size")
    plt.ylabel("Throughput (tokens/s)")
    plt.legend()

    plt.grid(True)
    plt.show()
