import time
import threading
import pynvml
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Global variables
monitor_threads = []
x_data_all = []
y_data_all = []
fig, ax = plt.subplots(figsize=(10, 5))
ani = None


def monitor_gpu(gpu_index, duration):
    global x_data_all, y_data_all

    # Initialize NVML
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

    x_data, y_data = [], []
    x_data_all.append(x_data)
    y_data_all.append(y_data)

    start_time = time.time()
    while not done:
        # Query GPU utilization
        info = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_usage = info.gpu

        # Calculate time elapsed
        elapsed_time = time.time() - start_time

        # Append data to lists
        x_data.append(elapsed_time)
        y_data.append(gpu_usage)

        # Pause to control the refresh rate
        time.sleep(0.1)  # Adjust the refresh rate as needed

    # Shutdown NVML
    pynvml.nvmlShutdown()


def monitor_start(duration=120):
    global done
    global monitor_threads
    global x_data_all
    global y_data_all
    global ani

    # Reset global variables
    done = False
    monitor_threads = []
    x_data_all = []
    y_data_all = []

    # Initialize NVML
    pynvml.nvmlInit()
    num_gpus = pynvml.nvmlDeviceGetCount()

    # Start monitoring threads for each GPU
    for i in range(num_gpus):
        t = threading.Thread(target=monitor_gpu, args=(i, duration))
        t.start()
        monitor_threads.append(t)

    ani = FuncAnimation(fig, update, interval=1000)  # Update plot every second


def monitor_end():
    global done
    global monitor_threads

    # Set the flag to stop monitoring
    done = True

    # Wait for all monitoring threads to finish
    for t in monitor_threads:
        t.join()

    # Close the NVML handle
    pynvml.nvmlShutdown()


def update(frame, duration=120):
    global x_data_all, y_data_all, ax

    # Clear the previous plot
    ax.clear()

    # Plot the monitoring data from all GPUs
    for i in range(len(x_data_all)):
        ax.plot(x_data_all[i], y_data_all[i], label=f"GPU {i}")

    ax.legend()
    ax.set_title('GPU Usage')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Usage (%)')

    # Adjust x-axis limits dynamically based on the duration of monitoring
    if x_data_all:
        max_elapsed_time = max(max(x_data) for x_data in x_data_all)
        ax.set_xlim(max(0, max_elapsed_time - duration), max(duration, max_elapsed_time))


# # Example usage:
# monitor_start(duration=120)  # Start monitoring for 120 seconds and Start the animation to continuously update the plot
#
# # Simulate running the model for 30 seconds (replace this with your actual code)
# Model()
#
# monitor_end()  # End monitoring
