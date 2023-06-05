from dbutils.pooled_db import PooledDB
from pyecharts.charts import Line, Pie, Liquid
from pyecharts import options as opts
from pyecharts.globals import ThemeType
import pymysql

POOL = PooledDB(
    creator=pymysql,
    maxcached=6,  # 最大连接数
    mincached=4,  # 初始化时最小连接数
    blocking=True,  # 链接池中若没有创建爱的链接则等待
    host="127.0.0.1",
    port=3306,
    user="root",
    password="pengwei",
    database="db01",
    charset="utf8"
)
conn = POOL.connection()
cursor = conn.cursor()

def create_line(table):
    sql = f'''select * from data_day_download where data_name= "{table}"'''
    cursor.execute(sql)
    datas = cursor.fetchall()
    time_list = [data[2] for data in datas]
    downloads = [data[3] for data in datas]
    line_chart = (
        Line(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width='600px', height='450px'))
        .add_xaxis(time_list)
        .add_yaxis("", downloads)
        .set_series_opts(label_opts=opts.LabelOpts(is_show=True))
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(splitline_opts=opts.SplitLineOpts(is_show=True)),
            title_opts=opts.TitleOpts(title="数据集每日下载量")
        )
    )
    line_chart.render('../Web/templates/line.html')

def create_pie(table):
    sql = f'''select * from dataset where data_name= "{table}"'''
    cursor.execute(sql)
    datas = cursor.fetchall()
    starts = [data[-6] for data in datas]
    un_starts = [data[-3] for data in datas]
    X = ['like', 'dislike']
    Y = [starts, un_starts]
    print(Y)
    pie_data = [(x, y) for x, y in zip(X, Y)]
    pie_chart = (
        Pie(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width='600px', height='500px'))
        .add(
            series_name="",
            data_pair=pie_data,
            radius=["15%", "35%"],
            itemstyle_opts=opts.ItemStyleOpts(
                border_width=1, border_color="rgba(0,0,0,0.3)"
            ),
        )
        .set_global_opts(
            tooltip_opts=opts.TooltipOpts(is_show=True, formatter="{b} {d}%"),
            legend_opts=opts.LegendOpts(is_show=True),
        )
    )
    pie_chart.render('../Web/templates/pie.html')


def create_liquid(starts, un_starts):
    liquid_chart = (
        Liquid(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width='600px', height='450px'))
        .add("好评率", [starts/float(starts+un_starts), starts/float(starts+un_starts), starts/float(starts+un_starts)], is_outline_show=False)
        .set_global_opts(title_opts=opts.TitleOpts(title="好评率"))
    )
    liquid_chart.render('../Web/templates/liquid.html')


if __name__ == '__main__':
    table = "2023_Countries_by_Population"
    create_line(table=table)
    create_pie(table=table)
    # create_liquid(table=table)
    cursor.close()
    conn.close()
