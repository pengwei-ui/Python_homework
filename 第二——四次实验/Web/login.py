import os
import re
from animals_classfication import predict
from flask import Flask, render_template, redirect, request, url_for
import pymysql
from dbutils.pooled_db import PooledDB
from create_picture import create_pie, create_line, create_liquid

app = Flask(__name__)

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
animals1 = ["狗", "马", "大象", "蝴蝶", "鸡"]
image_paths1 = [
    "../static/images/animals/dog.png",
    "../static/images/animals/horse.png",
    "../static/images/animals/elephant.png",
    "../static/images/animals/butterfly.png",
    "../static/images/animals/rooster.png",

]
animals2 = ["猫", "牛", "绵阳", "蜘蛛", "松鼠"]
image_paths2 = [
    "../static/images/animals/cat.png",
    "../static/images/animals/cow.png",
    "../static/images/animals/sheep.png",
    "../static/images/animals/spider.png",
    "../static/images/animals/squirrel.png",
]
datas1 = []
for data in zip(animals1, image_paths1):
    datas1.append(data)

datas2 = []
for data in zip(animals2, image_paths2):
    datas2.append(data)


def fetch_all(sql):
    # 去连接池获取一个链接
    conn = POOL.connection()
    cursor = conn.cursor()
    cursor.execute(sql)
    des = cursor.description
    datas = cursor.fetchall()
    title = [item[0] for item in des]
    cursor.close()
    # 将连接放回连接池
    conn.close()
    return title, datas


def fetch(sql):
    conn = POOL.connection()
    cursor = conn.cursor()
    cursor.execute(sql)
    datas = cursor.fetchall()
    cursor.close()
    conn.close()
    return datas


def fetch_dataset(sql):
    # 去连接池获取一个链接
    conn = POOL.connection()
    cursor = conn.cursor()
    cursor.execute(sql)
    datas = cursor.fetchall()
    down_link = datas[0][3]
    downloads = datas[0][4]
    starts = datas[0][5]
    see = datas[0][6]
    start_time = datas[0][7]
    un_starts = datas[0][8]
    # print(datas)
    cursor.close()
    # 将连接放回连接池
    conn.close()
    return down_link, downloads, starts, see, start_time, un_starts


def update(sql):
    # 去连接池获取一个链接
    conn = POOL.connection()
    cursor = conn.cursor()
    cursor.execute(sql)
    conn.commit()
    cursor.close()
    # 将连接放回连接池
    conn.close()


@app.route('/', methods=["GET", "POST"])
def login():
    return render_template("login.html")


@app.route('/vertify', methods=["GET", "POST"])
def vertify():
    user = request.args.get('user')
    pwd = request.args.get('pwd')
    sql = f'''select * from users where name = "{user}" and password = "{pwd}"'''
    title, datas = fetch_all(sql)
    error = "用户名或密码错误！"
    if len(datas) != 0:
        user_image = datas[0][-1]
        print(f"user_image:{user_image}")
        sql = "select * from dataset limit 0,4"
        title, datas = fetch_all(sql)
        infos = []
        for data in datas:
            infos.append((data[-1], data[1], data[2], data[-2]))
        # return render_template("index.html", infos=infos, user_image=user_image)
        return redirect(url_for('index', user_image=user_image, user=user))
    else:
        # return redirect("/")
        return render_template("login.html", error=error)


@app.route('/index')
def index():
    sql = "select * from dataset limit 0,4"
    # user_image = request.args.get("user_image")
    user = request.args.get("user")
    sql1 = f'''select image from users where name = "{user}"'''
    user_image = fetch(sql1)[0][0]
    print(f"index_user:{user}")
    title, datas = fetch_all(sql)
    infos = []
    print(f"index_user_image:{user_image}")
    for data in datas:
        infos.append((data[-1], data[1], data[2], data[-2]))
    return render_template("index.html", infos=infos, user_image=user_image, user=user)


@app.route("/data", methods=["GET", "POST"])
def data():
    table = request.args.get("table")
    user = request.args.get("user")
    user_image = request.args.get("user_image")
    print(f"table:{table}")
    print(f"user:{user}")
    sql1 = f'''select * from `{table}` limit 0,5 '''
    sql2 = f'''select * from dataset where data_name = "{table}"'''
    sql3 = f'''select * from comment'''
    create_line(table=table)
    titles, datas = fetch_all(sql1)
    comments_titles, comments = fetch_all(sql3)
    down_link, downloads, starts, see, start_time, un_starts = fetch_dataset(sql2)
    create_liquid(starts=starts, un_starts=un_starts)
    return render_template("data.html", titles=titles, datas=datas, table=table, down_link=down_link,
                           downloads=downloads, starts=starts, see=see, start_time=start_time, user=user
                           , un_starts=un_starts, comments_titles=comments_titles, comments=comments,
                           user_image=user_image)


@app.route("/update", methods=["GET", "POST"])
def update_data():
    table = request.args.get("table")
    user = request.args.get("user")
    sql = ""
    try:
        starts = request.args.get("starts")
        sql = f'''update dataset set starts = {int(starts) + 1} where data_name = "{table}" '''
    except:
        pass
    try:
        downloads = request.args.get("downloads")
        sql = f'''update dataset set downloads = {int(downloads) + 1} where data_name = "{table}" '''
    except:
        pass
    try:
        un_starts = request.args.get("un_starts")
        sql = f'''update dataset set un_starts = {int(un_starts) + 1} where data_name = "{table}" '''
    except:
        pass
    if sql:
        update(sql)
        sql_update_liquid = f'''select starts, un_starts from dataset where data_name = "{table}"'''
        title, datas = fetch_all(sql_update_liquid)
        starts, un_starts = datas[0][0], datas[0][1]
        print(datas)
        create_liquid(starts=starts, un_starts=un_starts)
    return redirect(f'/data?table={table}&user={user}')


@app.route("/update_see", methods=["GET", "POST"])
def update_see():
    table = request.args.get("table")
    try:
        see = request.args.get("see")
        sql = f'''update dataset set see = {int(see) + 1} where data_name = "{table}" '''
        update(sql)
    except:
        pass
    return ""


@app.route("/delete", methods=["GET", "POST"])
def delete():
    table = request.args.get("table")
    id = request.args.get("id")
    user = request.args.get("user")
    one_title = request.args.get("one_title")
    print(f"tabel:{table}")
    print(f"one_title:{one_title}")
    sql = f''' DELETE from {table} where `{one_title}`={id} '''
    print(f"sql:{sql}")
    print(f"id:{id}")
    update(sql)
    return redirect(f'/data?table={table}&user={user}')


@app.route("/delete_comments", methods=["GET", "POST"])
def delete_comments():
    print("hello")
    table = request.args.get("table")
    id = request.args.get("id")
    user = request.args.get("user")
    print(f"tabel:{table}")
    sql = f''' DELETE from comment where `id`={id} '''
    print(f"sql:{sql}")
    print(f"id:{id}")
    update(sql)
    return redirect(f'/data?table={table}&user={user}')


@app.route("/components")
def Components():
    user = request.args.get("user")
    return render_template("components.html", user=user)


@app.route("/push_comment", methods=['POST', 'GET'])
def push_comment():
    table = request.args.get('table')
    print(f"url:{request.url}")
    comment = request.args.get("comment")
    comment = request.args.get("comment")
    user = request.args.get("user")
    sql1 = f'''select image from users where name = "{user}"'''
    user_image = fetch(sql1)[0][0]
    sql2 = f'''INSERT into `comment`(user_name, comments, user_image) values('{user}','{comment}','{user_image}')'''
    update(sql2)
    return redirect(f'/data?table={table}&user={user}')


@app.route("/read_image", methods=['GET', 'POST'])
def read_image():
    try:
        image_name = request.args.get('image_name')
    except:
        image_name = ""
    try:
        image_result = request.args.get('image_result')
    except:
        image_result = ""
    print(f"this is read_image image_result:{image_name}")
    return render_template("read_image.html", datas1=datas1, datas2=datas2, image_name=image_name,
                           image_result=image_result)


@app.route("/check_image", methods=["GET", 'POST'])
def check_image():
    image_name = request.args.get('image_name')
    try:
        dir_name = re.findall("(.*?)\d", image_name)[0]
    except:
        dir_name = ""
    if dir_name:
        image_path = os.path.join(r"D:\data\animals_224", dir_name, image_name)
    else:
        image_path = os.path.join(r"D:\data\animals_224", image_name)
    print(image_path)
    image_result, image_rate = predict.run(img_path=image_path)
    print(f"this is check_image {image_name}")
    # return render_template("read_image_result.html", image_result=image_result, datas=datas)
    return redirect(f'/read_image?image_name={image_name}&image_result={image_result}')


# @app.route("/check_image", methods=['POST'])
# def check_image():
#     # source_img = request.form['file_source']
#     image_name = request.args.get('image_name')
#     # data = source_img.split(',')[1]
#     # image_data = base64.b64decode(data)
#     # with open('test.jpg', 'wb') as f:
#     #     f.write(image_data)
#     #     print("图片文件保存成功")
#     image_result = "dog!!"
#     print(f"this is check_image {image_name}")
#     # return render_template("read_image_result.html", image_result=image_result, datas=datas)
#     return redirect(f'/read_image?image_name={image_name}?image_result={image_result}')


@app.route("/error")
def error():
    return "没做"


@app.route('/line')
def line():
    table = request.args.get("table")
    create_line(table)
    print(f"table:{table}")
    return render_template('line.html')


@app.route('/liquid')
def liquid():
    return render_template('liquid.html')


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8082, debug=True)
