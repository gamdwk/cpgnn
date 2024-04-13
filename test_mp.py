import torch
from torch.multiprocessing import Process

# 定义一个函数作为进程的任务
def run_task():
    print("Child process is running")

if __name__ =="__main__":
# 创建一个进程对象
    process = Process(target=run_task)

# 启动进程
    process.start()

# 等待进程执行完毕
    process.join()

# 继续执行后续代码
    print("Process finished!")
