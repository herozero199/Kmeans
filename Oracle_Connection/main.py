import oracledb

connection = oracledb.connect(
    user="pvd",
    password="12345",
    dsn="localhost:1521/oracle"
)

cursor = connection.cursor()
with open("H:\DAT\HCSDL_DPT\Data\picture2.jpg", 'rb') as f:
    image = f.read()
cursor.execute("insert into test_hcsdl_dpt values (:image)", image=image)
connection.commit()

# with open('H:\\DAT\\HCSDL_DPT\\Data\\Rose\\rose_1.jpg', 'rb') as image_file:
#     image_data = image_file.read()
#
# sql = "insert into test_hcsdl_dpt (image) values (:image_data)"
# cursor.execute(sql, image_data=image_data)
# connection.commit()


# app = Flask(__name__)
#
# @app.route('/get-image')
# def get_image():
#     sql = "select image from test_hcsdl_dpt"
#     cursor.execute(sql)
#     result = cursor.fetchone()
#     # print(result)
#     blob_data = result[0]
#     # print(type(blob_data))
#     binary_data = base64.b64encode(blob_data.read())
#     # print(binary_data)
#     data = binary_data.decode("UTF-8")
#     # print(data)
#
#     # extract HSV histogram
#     stream = io.BytesIO(blob_data.read())
#     image = cv2.imdecode(np.frombuffer(stream.read(), np.uint8), cv2.IMREAD_COLOR)
#     cv2.imshow('image', image)
#     cv2.waitKey(0)
#
#     return '[{image: data},{},{}]'
#
# if __name__ == '__main__':
#     app.run(debug=True)