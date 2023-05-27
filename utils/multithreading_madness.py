import multiprocessing
import threading
from dataclasses import dataclass
from queue import Queue
from typing import Tuple, Dict


@dataclass
class ProcessExecutionResult:
    is_finished: bool


class ProcessPoolExecutor:

    def __init__(self, num_parallel_processes: int):
        self.num_parallel_processes = num_parallel_processes
        self.running_processes = []
        self.task_queue = Queue()
        self.stop_monitor_event = threading.Event()
        self.process_to_function_arg_bundle_map: Dict[multiprocessing.Process, Tuple[callable, tuple]] = dict()
        self.function_arg_bundle_result_map: Dict[Tuple[callable, tuple], ProcessExecutionResult] = dict()
        self.monitor_thread = threading.Thread(target=self.__monitor, daemon=True, name="ProcessPoolExecutor.monitor")

    def __monitor(self):
        while not self.stop_monitor_event.is_set():
            n_running_processes = 0
            for process in self.running_processes:
                if process.is_alive():
                    n_running_processes += 1
                else:
                    self.running_processes.remove(process)

                    function_arg_bundle = self.process_to_function_arg_bundle_map[process]
                    self.function_arg_bundle_result_map[function_arg_bundle].is_finished = True
                    del self.process_to_function_arg_bundle_map[process]
                    del self.function_arg_bundle_result_map[function_arg_bundle]

            if n_running_processes < self.num_parallel_processes:
                task = self.task_queue.get()
                func, args = task
                process = multiprocessing.Process(target=func, args=args)
                process.start()
                self.running_processes.append(process)
                self.process_to_function_arg_bundle_map[process] = task

    def submit(self, function: callable, args: tuple = None) -> ProcessExecutionResult:
        function_arg_bundle = (function, args)
        self.task_queue.put(function_arg_bundle)
        result = ProcessExecutionResult(is_finished=False)
        self.function_arg_bundle_result_map[function_arg_bundle] = result
        return result

    def __enter__(self):
        self.monitor_thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_monitor_event.set()
        self.monitor_thread.join()
