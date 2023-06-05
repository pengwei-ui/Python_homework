from flask import Flask, render_template, request, url_for
from werkzeug.utils import redirect

from Web import settings

app = Flask(__name__, template_folder="templates")
app.config.from_object(settings)

datas = {"a": "CNN", "b": "CSV", "c": "picture"}
Data = {
    1: {"name": "彭伟", "age": 24},
    2: {"name": "叒枅", "age": 25}
}


# 装饰器
#user = request.form.get('user') 获取post请求的参数
@app.route('/', methods=['GET', 'POST'])  # 路由
def login():  # 视图函数
    if request.method == 'GET':
        return render_template("data.html")
    user = request.form.get('user')
    pwd = request.form.get('pwd')
    error = "用户名或密码错误！"
    if user == "pw" and pwd == "1998":
        return redirect('/index')
    else:
        return render_template("data.html", error=error)

#endpoint为别名，可以使用url_for进行页面跳转
@app.route('/index', endpoint="idx")  # 路由
def index():
    data_list = Data
    return render_template("login.html", data_list=data_list)

#request.args.get("nid") 获取get请求的参数
@app.route("/edit", methods=["GET","POST"])
def edict():
    nid = request.args.get("nid")
    nid = int(nid)
    if request.method =="GET":
        info = Data[nid]
        return render_template("edit.html", info = info)

    user = request.form.get("user")
    age = request.form.get("age")
    Data[nid]['name'] = user
    Data[nid]['age'] = age
    return redirect(url_for("idx"))
#不写《int:》则默认为string类型
@app.route("/del/<int:nid>")
def delete(nid):
    del Data[nid]
    print(nid)
    return redirect("/index")

# <source> 声明变量名，该变量数据类型为str
# <int:num> 该变量数据类型为整形  <path:source> 该变量数据类型为路径
# return返回的数据为str\tuple\dict\response响应\WSGI，不能为int类型
# 路由中写了/，那么无论你是否写了/去请求，那么都会访问到数据，没写去请求就会重定向到有斜杠的网址
@app.route('/data/<source>/')  # 路由
def data(source):  # 视图函数
    return datas.get(source)


if __name__ == '__main__':
    # port进行设置端口号，如果host改成0.0.0.0那么外网可以访问
    # debug可以直接帮你探测页面内容是否改变，若改变，这页面刷新后会重新加载并显示，debug=True适用于开发（production）环节
    # app.run(host="127.0.0.1", port=8080, debug=True)
    app.run(host="127.0.0.1", port=8081)
