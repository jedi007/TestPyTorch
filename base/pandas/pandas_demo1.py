import pandas as pd


if __name__ == '__main__':
    print("version: ",pd.__version__)  # 查看版本



    mydataset = {
    'sites': ["Google", "Runoob", "Wiki"],
    'number': [1, 2, 3]
    }

    myvar = pd.DataFrame(mydataset)

    print(myvar)