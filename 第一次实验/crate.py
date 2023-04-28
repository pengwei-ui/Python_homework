import matplotlib.pyplot as plt
import pandas as pd
import random, re
import numpy as np
import matplotlib.ticker as mtick

# 设置全局字体及大小，设置公式字体
config = {
    "font.family": 'serif',  # 衬线字体
    "font.size": 20,  # 相当于小四大小
    "mathtext.fontset": 'stix',  # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
    "font.serif": ['SimSun'],  # 宋体SimSun
    "axes.unicode_minus": False,  # 用来正常显示负号
    "xtick.direction": 'in',  # 横坐标轴的刻度设置向内(in)或向外(out)
    "ytick.direction": 'in',  # 纵坐标轴的刻度设置向内(in)或向外(out)
}
plt.rcParams.update(config)


# 清洗数据
def modify(x):
    if re.findall("(\d*)-\d*-\d*", x):
        return re.findall("(\d*)-\d*-\d*", x)[0]
    else:
        return x


class Patting(object):
    def __init__(self, data_path=None):
        super(Patting, self).__init__()
        # 读取数据
        data = pd.read_csv(data_path)
        df = data[(data['sellPrice'] < 1300000) & (data['sellPrice'] > 400000) & (
                    data['sellPrice'] % 500 == 0)]  # 和符号用的是&，用的不是and
        df_ = df.copy()
        #将日期转化如2019-6-28--->2019
        df_['Date'] = df_['Date'].apply(lambda r: modify(r))
        self.df = df_.dropna()

    def create_bar(self):
        # 设置画布的大小
        plt.figure(figsize=(8, 10))
        # 这是我选的类别，两个类别
        suburbs = ('Baulkham Hills', 'Cherrybrook')
        # 这是我的横坐标标签
        years = ["2017", "2018", "2019"]
        house_count = []
        for suburb in suburbs:
            for year in years:
                house_count.append(len(self.df[(self.df['suburb'] == suburb) & (self.df['Date'] == year)]))
        # 这是我的Y轴数据
        house1_count = house_count[0:3]
        house2_count = house_count[3:6]
        # 制作横坐标，使其横坐标标签居中
        x = np.arange(len(years))
        width = 0.8 / len(suburbs)
        x = x - (0.8 - width) / 2
        # 误差列表，为了画标准偏差，随机生成度数
        std_err1 = [random.randint(8, 20) for _ in house1_count]
        std_err2 = [random.randint(4, 10) for _ in house2_count]
        # 设置误差标记参数
        error_params = dict(elinewidth=2, ecolor='black', capsize=4)  # 设置误差标记参数

        plt.bar(x, house1_count, lw=0.5, fc="r", yerr=std_err1, error_kw=error_params, width=0.4, label=suburbs[0])
        plt.bar(x + width, house2_count, lw=0.5, fc="b", yerr=std_err2, error_kw=error_params, width=0.4,
                label=suburbs[1])
        # 纵坐标标签
        plt.ylabel("count(individual)")
        # 横坐标标签
        plt.xlabel("year")
        plt.title("number of home sales")
        # 内标签显示位置，可以选左上upper left
        plt.legend(loc="upper right")
        # 显示格子线
        plt.grid()
        # 横坐标值
        plt.xticks(range(0, len(years)), years)
        plt.savefig('./bar.png', dpi=100)
        plt.show()
        pass

    def crate_scatter(self):
        # 设置画布的大小
        plt.figure(figsize=(10, 10))
        # 生成伪随机数
        self.df["random_w"] = [random.random() for _ in range(len(self.df))]
        self.df["random_b"] = [random.random() for _ in range(len(self.df))]
        self.df["random_x"] = self.df["random_w"] * self.df["bed"] + self.df["random_b"]
        self.df["random_y"] = self.df["random_w"] * self.df["bath"] + self.df["random_b"]
        #
        propTypes = ["house", "townhouse", "duplex/semi-detached"]
        df3 = self.df[["propType", "random_x", "random_y"]]

        x1 = df3[(df3['propType'] == "house")]['random_x']
        x1 = np.array(x1[0:200]) + np.array([round(random.randint(3, 4), 1) for _ in range(200)])
        y1 = df3[(df3['propType'] == "house")]['random_y']
        y1 = np.array(y1[0:200])

        x2 = df3[(df3['propType'] == "townhouse")]['random_x']
        x2 = np.array(x2[0:200])
        y2 = df3[(df3['propType'] == "townhouse")]['random_y']
        y2 = np.array(y2[0:200]) + np.array([round(random.randint(3, 4), 1) for _ in range(200)])

        x3 = df3[(df3['propType'] == "duplex/semi-detached")]['random_x']
        x3 = np.array(x3[0:200])
        y3 = df3[(df3['propType'] == "duplex/semi-detached")]['random_y']
        y3 = np.array(y3[0:200])

        plt.scatter(x1, y1, s=100, alpha=0.5, cmap='afmhot_r',
                    )  # s 点的大小  c 点的颜色 alpha 透明度 cmap 颜色条(color需要设置为数组)
        plt.scatter(x2, y2, s=100, alpha=0.5, cmap='afmhot_r',
                    )  # s 点的大小  c 点的颜色 alpha 透明度 cmap 颜色条(color需要设置为数组)
        plt.scatter(x3, y3, s=100, alpha=0.5, cmap='afmhot_r')
        # 纵坐标标签
        plt.ylabel("bed")
        # 横坐标标签
        plt.xlabel("bath")
        # 表头
        plt.title("three house-style distributed")
        plt.legend(labels=propTypes, title="species", loc="upper right")
        plt.savefig('./scatter.png', dpi=100)
        plt.show()
        pass

    def create_line(self):
        years = [str(i) for i in range(2008, 2020)]
        mean_prices, max_prices, min_prices = [], [], []
        Mean_Accelerations = [0]
        for i in range(2008, 2020):
            df_line_data_Greenacre_sellPrice = self.df[
                (self.df['suburb'] == "Greenacre") & ((self.df['Date'] == str(i)))]
            mean_prices.append(df_line_data_Greenacre_sellPrice.mean()['sellPrice'])
            max_prices.append(max(df_line_data_Greenacre_sellPrice['sellPrice']))
            min_prices.append(min(df_line_data_Greenacre_sellPrice['sellPrice']))
        for i in range(1, len(mean_prices)):
            Mean_Accelerations.append(100 * (mean_prices[i] - mean_prices[i - 1]) / mean_prices[i - 1])
        # 绘图
        # #设置画布的大小
        fig, ax1 = plt.subplots(figsize=(20, 10))
        ax1.plot(years, min_prices, label='min_prices', c='Red')
        ax1.plot(years, max_prices, label='max_prices', c='black')
        # 变动折线图
        ax2 = ax1.twinx()
        ax2.plot(years, Mean_Accelerations, 'o-', c='green', label=u"Mean_price_accelerations")
        max_proges = max(Mean_Accelerations)
        # 设置纵轴格式
        fmt = '%.1f%%'
        yticks = mtick.FormatStrFormatter(fmt)
        ax2.yaxis.set_major_formatter(yticks)
        ax2.set_ylim(min(Mean_Accelerations), int(max_proges * 1.2))
        ax2.set_ylabel(u"Mean_price_accelerations")

        # x轴标签
        ax1.set_xlabel('year')
        # y轴标签
        ax1.set_ylabel('price')
        # 图表标题
        plt.title('Greenacre house price/year')
        # 显示图例
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        # 保存图形
        plt.savefig('./line.png', dpi=100)
        plt.show()
        pass

    def create_box(self):
        # 设置画布的大小
        plt.figure(figsize=(10, 10))
        suburbs = ["Greenacre", "Auburn", "Merrylands"]
        prices = []
        indexs = random.sample(range(0, 1000), 8)
        for suburb in suburbs:
            prices.append(np.array(self.df[self.df["suburb"] == suburb]["sellPrice"]))
        for i in range(len(suburbs)):
            for index in range(len(indexs)):
                if index % 8 == 0:
                    prices[i][index] = prices[i][index] * 0.1
                else:
                    prices[i][index] = prices[i][index] * 1.5

        box_plot = plt.boxplot(prices, sym='o', patch_artist=True, vert=True)
        medians = [median.get_ydata()[0] for median in box_plot["medians"]]
        outliers = [outlier.get_ydata() for outlier in box_plot["fliers"]]
        # 在异常值位置添加数值标签和离群值标记
        for i in range(len(medians)):
            plt.text(i + 1, medians[i], round(medians[i], 2),
                     horizontalalignment='center', fontsize=10, color="black")
            for j in range(len(outliers[i])):
                plt.text(i + 1, outliers[i][j], outliers[i][j],
                         horizontalalignment='center', fontsize=10,
                         color='red')
        plt.xticks(np.arange(1, 4), suburbs)
        plt.xlabel('suburb')
        plt.ylabel('price')
        plt.title('suburb house price')
        plt.savefig('./box.png', dpi=100)
        plt.show()
        pass


if __name__ == '__main__':
    data_path = r".\SydneyHousePrices.csv"
    patting = Patting(data_path=data_path)
    # patting.create_bar()
    # patting.create_line()
    patting.create_box()
    # patting.crate_scatter()
