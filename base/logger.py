import logging
import os.path
import time
from pathlib import Path

def init_logger( log_path = "./Logs" ):
    # 第一步，创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关
    # 第二步，创建一个handler，用于写入日志文件
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))

    if not os.path.exists( log_path ):
        os.mkdir( log_path )

    logfile = os.path.join(log_path,rq +'.log')
    fh = logging.FileHandler(logfile, mode='w')
    # fh.setLevel(logging.WARNING)  # 输出到file的log等级的开关
    # 第三步，定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    # 第四步，将logger添加到handler里面
    logger.addHandler(fh)
    return logger

if __name__ == '__main__':
    ROOT = Path(__file__).resolve().parents[0]

    print("ROOT: ",ROOT)
    log_path = os.path.join(ROOT,'Logs')
    print("log_path: ",log_path)
    
    
    logger = init_logger(log_path)

    # 将信息打印到控制台上
    logger.debug(u"苍井空")
    logger.info(u"麻生希")
    logger.warning(u"小泽玛利亚")
    logger.error(u"桃谷绘里香")
    logger.critical(u"泷泽萝拉")