let mysql = require('mysql');

const connection = mysql.createConnection({
    host: 'localhost',              // 主机
    port: '3306',                   // 端口号
    user: 'root',                   // 用户名
    password: 'pengwei',                   // 密码
    database: 'db01'        // 数据库名
});
// 创建连接

connection.connect((error) => {
        if(error){
            console.log("连接失败" + error.stack)
            return
        }else{
            console.log("连接成功")
        }
    }
);

connection.end()