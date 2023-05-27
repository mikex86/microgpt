import re
import matplotlib.pyplot as plt

if __name__ == '__main__':
    file_path = "token_generation_rates.txt"
    pricing_file_path = "prices.txt"

    gpu_throughputs = {}
    gpu_pricings = {}

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

    # Parse pricing data
    with open(pricing_file_path, "r") as pricing_file:
        pricing_content = pricing_file.read()

    gpu_prices = re.findall(r"(.*?): \$(.*?)\/hr", pricing_content)

    for gpu_price in gpu_prices:
        gpu_name = gpu_price[0]
        gpu_hourly_price = float(gpu_price[1])

        gpu_pricings[gpu_name] = gpu_hourly_price

    token_to_dollar = []
    # Plotting
    for gpu_name, gpu_data in gpu_throughputs.items():
        batch_sizes = gpu_data["batch_sizes"]
        throughputs = gpu_data["throughputs"]

        max_throughput = max(throughputs)
        max_throughput_batch_size = batch_sizes[throughputs.index(max_throughput)]
        print(f"Highest throughput for {gpu_name}: {max_throughput} tokens/s at batch size {max_throughput_batch_size}")

        token_to_dollar.append((gpu_name, (max_throughput * 60 * 60) / gpu_pricings[gpu_name]))

    token_to_dollar.sort(key=lambda x: x[1], reverse=True)

    plt.figure(figsize=(10, 5))
    plt.bar([x[0] for x in token_to_dollar], [x[1] for x in token_to_dollar])
    plt.xlabel("GPU")
    plt.ylabel("Tokens/$")
    plt.grid(True)
    plt.show()
