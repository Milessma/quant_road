import datetime
import sys
from typing import Any

def log(*args: Any, **kwargs: Any) -> None:
    """
    增强的日志函数，显示时间并打印语句
    
    参数:
        *args: 要打印的内容，多个参数会自动用空格分隔
        **kwargs: 额外的打印选项，如sep、end等
    """
    # 获取当前时间
    current_time = datetime.datetime.now()
    
    # 格式化时间字符串
    # 可以调整格式：%Y-%m-%d %H:%M:%S.%f 包含毫秒
    # %Y-%m-%d %H:%M:%S 标准格式
    time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    
    # 构建日志前缀
    log_prefix = f"[{time_str}]"
    
    # 处理额外参数
    sep = kwargs.get('sep', ' ')
    end = kwargs.get('end', '\n')
    file = kwargs.get('file', sys.stdout)
    
    # 打印时间和内容
    print(log_prefix, *args, sep=sep, end=end, file=file)

# ========== 使用示例 ==========

if __name__ == "__main__":
    # 基本使用
    log("程序开始运行")
    log("用户登录", "用户ID: 12345")
    
    # 带有变量的日志
    user_name = "张三"
    age = 25
    log(f"用户信息: 姓名={user_name}, 年龄={age}")
    
    # 错误日志示例
    try:
        result = 10 / 0
    except Exception as e:
        log(f"发生错误: {e}", file=sys.stderr)
    
    log("程序结束")
    
# def example_func(a, b, *args, **kwargs):
#     """
#     函数参数顺序必须是：
#     1. 普通参数 (a, b)
#     2. *args
#     3. **kwargs
#     """
#     print(f"a = {a}")
#     print(f"b = {b}")
#     print(f"args = {args}")
#     print(f"kwargs = {kwargs}")
#     print("-" * 30)

# # 使用示例
# example_func(1, 2)
# # 输出:
# # a = 1
# # b = 2
# # args = ()
# # kwargs = {}

# example_func(1, 2, 3, 4, 5)
# # 输出:
# # a = 1
# # b = 2
# # args = (3, 4, 5)
# # kwargs = {}

# example_func(1, 2, 3, 4, name="张三", age=25)
# # 输出:
# # a = 1
# # b = 2
# # args = (3, 4)
# # kwargs = {'name': '张三', 'age': 25}