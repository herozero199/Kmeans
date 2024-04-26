import oracledb
connection = oracledb.connect(
    user="pvd",
    password="12345",
    dsn="localhost:1521/oracle"
)
cursor = connection.cursor()

centroids = []
get_centroids = "select * from centroids"
result = cursor.execute(get_centroids)
for row in result:
    cluster_id = row[0]
    feature_vector = [float(i) for i in row[1].read().split(';')]
    new_centroid = [cluster_id, feature_vector]
    centroids.append(new_centroid)
print(centroids)