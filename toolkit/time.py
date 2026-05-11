from datetime import datetime

def print_with_time(*args, **kwargs):
    """带时间戳的print函数，不替换内置print"""
    # 获取当前时间并格式化为[YYYY-MM-DD HH:MM:SS]形式
    current_time = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    # 调用内置print函数，并在前面添加时间戳
    print(current_time, *args, **kwargs,flush=True)
    
    
def return_second_timestamp_str():


    # 1. 获取当前日期时间对象（包含月、日、时、分、秒）
    now = datetime.now()  # 或 datetime.today()，效果一致

    # 2. 格式化为10位月日时分秒字符串（MMDDHHMMSS）
    # %m=2位月 | %d=2位日 | %H=2位小时(24小时制) | %M=2位分钟 | %S=2位秒
    thirteen_digit_datetime_str = now.strftime("%m%d%H%M%S%f")[:-3]
    
    return thirteen_digit_datetime_str