
from flask import Flask, render_template, request, flash, send_file
from flaskext.mysql import MySQL
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from mpl_toolkits.mplot3d import Axes3D, axes3d
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
import django_excel as excel
import urllib





app= Flask(__name__)

app.secret_key = "mysecretkey"
#conexion con la base de datos
mysql=MySQL()
app.config['MYSQL_DATABASE_HOST']='localhost'
app.config['MYSQL_DATABASE_USER']='root'
app.config['MYSQL_DATABASE_PASSWORD']=''
app.config['MYSQL_DATABASE_DB']='signos'
mysql.init_app(app)

# routes
@app.route('/')
def index():
  
    return render_template('index.html')

@app.route('/Guardar', methods=['GET','POST'])
def Guardar():
    if request.method == 'POST' or request == 'GET':

     _name = request.form['name']
    selectuno=request.form.get('selectuno')
    _pregunta2=request.form.get('pregunta2')
    
    _Pregunta3=request.form.get('pregunta3')
    _Pregunta4=request.form.get('pregunta4')
    _Pregunta5=request.form.get('pregunta5')
    _Pregunta6=request.form.get('pregunta6')
    _Pregunta7=request.form.get('pregunta7')
    _Pregunta8=request.form.get('pregunta8')
    _Pregunta9=request.form.get('pregunta9')
    _Pregunta10=request.form.get('pregunta10')

    total = (int(selectuno)+int(_pregunta2)+int(_Pregunta3)+int(_Pregunta4)+int(_Pregunta5)+int(_Pregunta6)+int(_Pregunta7)+int(_Pregunta8)+int(_Pregunta9)+int(_Pregunta10));
    sql="INSERT INTO `r_informacion` ( `nombre`, `Pregunta1_res`, `Pregunta2_res`, `Pregunta3_res`, `Pregunta4_res`, `Pregunta5_res`, `Pregunta6_res`, `Pregunta7_res`, `Pregunta8_res`, `Pregunta9_res`, `Pregunta10_res`,`Total_Puntos`) VALUES ( %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"
    datos =(_name,selectuno,_pregunta2,_Pregunta3,_Pregunta4,_Pregunta5,_Pregunta6,_Pregunta7,_Pregunta8,_Pregunta9,_Pregunta10,total)
   # sql2 = "SELECT * FROM `r_informacion`;"
    conn=mysql.connect()
    cursor=conn.cursor()
    cursor.execute(sql, datos)
    conn.commit()
    flash("Archivo guardado")
    #cursor.execute(sql2)
   # respuestas =cursor.fetchall()
    return render_template('index.html')

@app.route('/verdatos')
def verdatos():
    sql2 = "SELECT * FROM `r_informacion`;"
    conn=mysql.connect()
    cursor=conn.cursor()
    cursor.execute(sql2)
    conn.commit() 
    respuestas =cursor.fetchall()
    return render_template('datos.html', respuestas=respuestas)   

@app.route('/grafica1')
def grafica():
    sql4 = "SELECT  Id_Persona,nombre,Pregunta1_res,Pregunta2_res,Pregunta3_res,Pregunta4_res,Pregunta5_res,Pregunta6_res,Pregunta7_res,Pregunta8_res,Pregunta9_res,Pregunta10_res,Total_Puntos FROM `r_informacion`;"
    conn=mysql.connect()
    cursor=conn.cursor()
    cursor.execute(sql4)
    #sql='SELECT * FROM preguntas'
    df = pd.read_sql_query(sql4,conn)
    print(df)
    df.to_csv("datos.csv", index=False)
    respuestas = cursor.fetchall()
    #cursor.close()

    #se aplica el algoritmo
    dataframe = pd.read_csv("datos.csv", engine='python')

    dataframe.describe()
    
    print(dataframe.groupby('Total_Puntos').size())
    dataframe.drop(['Total_Puntos'],1).hist()
    plt.show()
    return render_template('datos.html', respuestas = respuestas)  


@app.route('/grafica2')
def grafica2():
    sql4 = "SELECT  Id_Persona,nombre,Pregunta1_res,Pregunta2_res,Pregunta3_res,Pregunta4_res,Pregunta5_res,Pregunta6_res,Pregunta7_res,Pregunta8_res,Pregunta9_res,Pregunta10_res,Total_Puntos FROM `r_informacion`;"
    conn=mysql.connect()
    cursor=conn.cursor()
    cursor.execute(sql4)
    #sql='SELECT * FROM preguntas'
    df = pd.read_sql_query(sql4,conn)
    print(df)
    df.to_csv("datos.csv", index=False)
    respuestas = cursor.fetchall()
    #se aplica el algoritmo
    dataframe = pd.read_csv("datos.csv", engine='python')
    
    X = dataframe.iloc[:, [3,4,5,6,7,8,9,10,11]].values
     
    from sklearn.cluster import KMeans
    wcss = []
    for i in range(1, 12):
      kmeans = KMeans(n_clusters = i, init = 'k-means++')
      kmeans.fit(X)
      wcss.append(kmeans.inertia_)
     # # #grafica de codos
    plt.plot(range(1, 12), wcss)
    plt.title("Codos de jambu")
    plt.xlabel("Numero de clusters")
    plt.ylabel("wcss")
    plt.show()

    return render_template('datos.html', respuestas = respuestas)  

@app.route('/grafica3')
def grafica3():
    sql4 = "SELECT  Id_Persona,nombre,Pregunta1_res,Pregunta2_res,Pregunta3_res,Pregunta4_res,Pregunta5_res,Pregunta6_res,Pregunta7_res,Pregunta8_res,Pregunta9_res,Pregunta10_res,Total_Puntos FROM `r_informacion`;"
    conn=mysql.connect()
    cursor=conn.cursor()
    cursor.execute(sql4)
    #sql='SELECT * FROM preguntas'
    df = pd.read_sql_query(sql4,conn)
    print(df)
    df.to_csv("datos.csv", index=False)
    respuestas = cursor.fetchall()
    #cursor.close()
    dataframe = pd.read_csv("datos.csv", engine='python')
    sb.pairplot(dataframe.dropna(), hue='Total_Puntos',size=11,vars=["Pregunta1_res","Pregunta2_res","Pregunta3_res","Pregunta4_res","Pregunta5_res","Pregunta6_res","Pregunta7_res","Pregunta8_res","Pregunta9_res","Pregunta10_res"],kind='scatter')
    plt.show()
    return render_template('datos.html', respuestas = respuestas)  


@app.route('/grafica4')
def grafica4():
    sql4 = "SELECT  Id_Persona,nombre,Pregunta1_res,Pregunta2_res,Pregunta3_res,Pregunta4_res,Pregunta5_res,Pregunta6_res,Pregunta7_res,Pregunta8_res,Pregunta9_res,Pregunta10_res,Total_Puntos FROM `r_informacion`;"
    conn=mysql.connect()
    cursor=conn.cursor()
    cursor.execute(sql4)
    #sql='SELECT * FROM preguntas'
    df = pd.read_sql_query(sql4,conn)
    print(df)
    df.to_csv("datos.csv", index=False)
    respuestas = cursor.fetchall()
    #cursor.close()
    dataframe = pd.read_csv("datos.csv", engine='python')
    X = dataframe.iloc[:, [3,4,5,6,7,8,9,10,11]].values
    
# Creando el k-Means para los 4 grupos encontrados
    kmeans = KMeans(n_clusters = 12, init = 'k-means++', random_state = 42)
    y_kmeans = kmeans.fit_predict(X)
    colores=['red','green','blue','cyan','pink','orange','purple','grey','black','yellow','gold','silver']
# Visualizacion grafica de los clusters
    plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Aries')
    plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 180, c = 'green', label = 'Tauro')
    plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 240, c = 'blue', label = 'Geminis')
    plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 280, c = 'cyan', label = 'Cancer')
    plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 120, c = 'pink', label = 'Leo')
    plt.scatter(X[y_kmeans == 5, 0], X[y_kmeans == 5, 1], s = 200, c = 'orange', label = 'Virgo')
    plt.scatter(X[y_kmeans == 6, 0], X[y_kmeans == 6, 1], s = 260, c = 'purple', label = 'Libra')
    plt.scatter(X[y_kmeans == 7, 0], X[y_kmeans == 7, 1], s = 300, c = 'grey', label = 'Escorpio')
    plt.scatter(X[y_kmeans == 8, 0], X[y_kmeans == 8, 1], s = 160, c = 'black', label = 'Sagitario')
    plt.scatter(X[y_kmeans == 9, 0], X[y_kmeans == 9, 1], s = 140, c = 'yellow', label = 'Capricornio')
    plt.scatter(X[y_kmeans == 10, 0], X[y_kmeans == 10, 1], s = 220, c = 'gold', label = 'Acuario')
    plt.scatter(X[y_kmeans == 11, 0], X[y_kmeans == 11, 1], s = 320, c = 'silver', label = 'Piscis')
   
    

    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 400, marker='*', c = colores , label = 'Centroids')

    plt.title('Primer entrenamiento')
    plt.xlabel('')
    plt.ylabel('')
    plt.legend()
    plt.show()

    from sklearn.cluster import AgglomerativeClustering
    hc = AgglomerativeClustering(n_clusters = 12, 
                         affinity = 'euclidean', 
                         linkage = 'ward')

    y_hc = hc.fit_predict(X)

    plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 110, c = 'red', label = 'Aries')
    plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 180, c = 'green', label = 'Tauro')
    plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 240, c = 'blue', label = 'Geminis')
    plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 280, c = 'cyan', label = 'Cancer')
    plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 120, c = 'pink', label = 'Leo')
    plt.scatter(X[y_hc == 5, 0], X[y_hc == 5, 1], s = 200, c = 'orange', label = 'Virgo')
    plt.scatter(X[y_hc == 6, 0], X[y_hc == 6, 1], s = 260, c = 'purple', label = 'Libra')
    plt.scatter(X[y_hc == 7, 0], X[y_hc == 7, 1], s = 300, c = 'grey', label = 'Escorpio')
    plt.scatter(X[y_hc == 8, 0], X[y_hc == 8, 1], s = 160, c = 'black', label = 'Sagitario')
    plt.scatter(X[y_hc == 9, 0], X[y_hc == 9, 1], s = 140, c = 'yellow', label = 'Capricornio')
    plt.scatter(X[y_hc == 10, 0], X[y_hc == 10, 1], s = 220, c = 'gold', label = 'Acuario')
    plt.scatter(X[y_hc == 11, 0], X[y_hc == 11, 1], s = 320, c = 'silver', label = 'Piscis')

    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 400, marker='*', c = colores , label = 'Centroids')

    plt.title('Cluster segundo entrenamiento')
    plt.xlabel('')
    plt.ylabel('')
    plt.legend()
    plt.show()  
    return render_template('datos.html', respuestas = respuestas)  

@app.route('/grafica5')
def grafica5():
    sql4 = "SELECT  Id_Persona,nombre,Pregunta1_res,Pregunta2_res,Pregunta3_res,Pregunta4_res,Pregunta5_res,Pregunta6_res,Pregunta7_res,Pregunta8_res,Pregunta9_res,Pregunta10_res,Total_Puntos FROM `r_informacion`;"
    conn=mysql.connect()
    cursor=conn.cursor()
    cursor.execute(sql4)
    #sql='SELECT * FROM preguntas'
    df = pd.read_sql_query(sql4,conn)
    print(df)
    df.to_csv("datos.csv", index=False)
    respuestas = cursor.fetchall()
    #cursor.close()
    dataframe = pd.read_csv("datos.csv", engine='python')
    X = dataframe.iloc[:, [3,4,5,6,7,8,9,10,11]].values
    import scipy.cluster.hierarchy as sch
    dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))

    plt.title('Dendograma')
    plt.xlabel('')
    plt.ylabel('Distancias Euclidianas')
    plt.show()
    return render_template('datos.html', respuestas = respuestas)  
@app.route('/descargar')
def downloadFile ():
    path = "/Users/Jesus/Signos_Zodiacales/datos.csv"
    return send_file(path, as_attachment=True)
    
@app.route('/Analizar')
def AnalizarKmeas():
    sql4 = "SELECT  Id_Persona,Pregunta1_res,Pregunta2_res,Pregunta3_res,Pregunta4_res,Pregunta5_res,Pregunta6_res,Pregunta7_res,Pregunta8_res,Pregunta9_res,Pregunta10_res,Total_Puntos FROM `r_informacion`;"
    conn=mysql.connect()
    cursor=conn.cursor()
    cursor.execute(sql4)
    #sql='SELECT * FROM preguntas'
    df = pd.read_sql_query(sql4,conn)
    print(df)
    df.to_csv("datos.csv", index=False)
    data = cursor.fetchall()
    #cursor.close()

    #se aplica el algoritmo
    dataframe = pd.read_csv("datos.csv", engine='python')

    dataframe.describe()
    
    print(dataframe.groupby('Total_Puntos').size())
    dataframe.drop(['Total_Puntos'],1).hist()
    plt.show()
    sb.pairplot(dataframe.dropna(), hue='Total_Puntos',size=11,vars=["Pregunta1_res","Pregunta2_res","Pregunta3_res","Pregunta4_res","Pregunta5_res","Pregunta6_res","Pregunta7_res","Pregunta8_res","Pregunta9_res","Pregunta10_res"],kind='scatter')
    plt.show()
    X = np.array(dataframe[["Pregunta1_res","Pregunta2_res","Pregunta3_res","Pregunta4_res","Pregunta5_res","Pregunta6_res","Pregunta7_res","Pregunta8_res","Pregunta9_res","Pregunta10_res"]])
    y = np.array(dataframe['Total_Puntos'])
    X.shape
    
    from sklearn.cluster import KMeans
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++')
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

   
    
    kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
#     y_kmeans = kmeans.fit_predict(X)

#     colores=['red','green','blue','cyan','pink','orange','purple','grey','black','yellow','gold','silver']
#     # Visualizacion grafica de los clusters
    plt.scatter(X[:,0], X[:, 1], X[:, 2], s = 400, c = 'red', label = 'Aries')
    
    plt.show
    #sb.pairplot(dataframe.dropna(), hue='Total_Puntos',size=11,vars=["Pregunta1_res","Pregunta2_res","Pregunta3_res","Pregunta4_res","Pregunta5_res","Pregunta6_res","Pregunta7_res","Pregunta8_res","Pregunta9_res","Pregunta10_res"],kind='scatter')
   # X = signos.iloc[:, [11]].values
     
#     from sklearn.cluster import KMeans
#     wcss = []
#     for i in range(1, 12):
#      kmeans = KMeans(n_clusters = i, init = 'k-means++')
#      kmeans.fit(X)
#      wcss.append(kmeans.inertia_)

#     # # #grafica de codos
#     plt.plot(range(1, 12), wcss)
#     plt.title("Codos de jambu")
#     plt.xlabel("Numero de clusters")
#     plt.ylabel("wcss")
#     plt.show()


#     # Creando el k-Means para los 4 grupos encontrados
#     kmeans = KMeans(n_clusters = 12, init = 'k-means++', random_state = 42)
#     y_kmeans = kmeans.fit_predict(X)

#     colores=['red','green','blue','cyan','pink','orange','purple','grey','black','yellow','gold','silver']
#     # Visualizacion grafica de los clusters
#    # plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Aries')
#     plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 180, c = 'green', label = 'Tauro')
#     plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 240, c = 'blue', label = 'Geminis')
#     plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 280, c = 'cyan', label = 'Cancer')
#     plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 120, c = 'pink', label = 'Leo')
#     plt.scatter(X[y_kmeans == 5, 0], X[y_kmeans == 5, 1], s = 200, c = 'orange', label = 'Virgo')
#     plt.scatter(X[y_kmeans == 6, 0], X[y_kmeans == 6, 1], s = 260, c = 'purple', label = 'Libra')
#     plt.scatter(X[y_kmeans == 7, 0], X[y_kmeans == 7, 1], s = 300, c = 'grey', label = 'Escorpio')
#     plt.scatter(X[y_kmeans == 8, 0], X[y_kmeans == 8, 1], s = 160, c = 'black', label = 'Sagitario')
#     plt.scatter(X[y_kmeans == 9, 0], X[y_kmeans == 9, 1], s = 140, c = 'yellow', label = 'Capricornio')
#     plt.scatter(X[y_kmeans == 10, 0], X[y_kmeans == 10, 1], s = 220, c = 'gold', label = 'Acuario')
#     plt.scatter(X[y_kmeans == 11, 0], X[y_kmeans == 11, 1], s = 320, c = 'silver', label = 'Piscis')
   
    

#     plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 400, marker='*', c = colores , label = 'Centroids')

#     plt.title('Clusters of customers')
#     plt.xlabel('numero')
#     plt.ylabel('Puntaje(1-400)')
#     plt.legend()
#     plt.show()


#     # Creamos el dendograma para encontrar el número óptimo de clusters

#     import scipy.cluster.hierarchy as sch
#     dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))

#     plt.title('Dendograma')
#     plt.xlabel('Clientes')
#     plt.ylabel('Distancias Euclidianas')
#     plt.show()

#     from sklearn.cluster import AgglomerativeClustering
#     hc = AgglomerativeClustering(n_clusters = 12, 
#                         affinity = 'euclidean', 
#                         linkage = 'ward')

#     y_hc = hc.fit_predict(X)

#     plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 110, c = 'red', label = 'Aries')
#     plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 180, c = 'green', label = 'Tauro')
#     plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 240, c = 'blue', label = 'Geminis')
#     plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 280, c = 'cyan', label = 'Cancer')
#     plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 120, c = 'pink', label = 'Leo')
#     plt.scatter(X[y_hc == 5, 0], X[y_hc == 5, 1], s = 200, c = 'orange', label = 'Virgo')
#     plt.scatter(X[y_hc == 6, 0], X[y_hc == 6, 1], s = 260, c = 'purple', label = 'Libra')
#     plt.scatter(X[y_hc == 7, 0], X[y_hc == 7, 1], s = 300, c = 'grey', label = 'Escorpio')
#     plt.scatter(X[y_hc == 8, 0], X[y_hc == 8, 1], s = 160, c = 'black', label = 'Sagitario')
#     plt.scatter(X[y_hc == 9, 0], X[y_hc == 9, 1], s = 140, c = 'yellow', label = 'Capricornio')
#     plt.scatter(X[y_hc == 10, 0], X[y_hc == 10, 1], s = 220, c = 'gold', label = 'Acuario')
#     plt.scatter(X[y_hc == 11, 0], X[y_hc == 11, 1], s = 320, c = 'silver', label = 'Piscis')

#     plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 400, marker='*', c = colores , label = 'Centroids')

#     plt.title('Clusters of customers')
#     plt.xlabel('Annual Income (k$)')
#     plt.ylabel('Spending Score (1-100)')
#     plt.legend()
#     plt.show()    
    # # #modelo de clusters
    
    # clustering = KMeans(n_clusters = 10, max_itex= 300)
    # clustering.fit(signos_noms)

    # # # #agregando clasificacion al archivo origibal
    # signos['KMeans_clusters'] = clustering.labels_
    # signos.head()

    # #visualizacion de cluster formato
    # pca = PCA(n_components=2)
    # pca_signos = pca.fit_transform(signos_noms)
    # pca_signos_df = pd.DataFrame(dat = pca_signos,columna =['components_1,components_2'])
    # pca_nombres_signos = pd.contact([pca_signos_df, signos[['KMeans_clusters']]], axis=1)
    # pca_nombres_signos
    # #datos.shape()

    # fig = plt.figure(figsize = (6,6))
    # ax = fig.add_subplot(1,1,1)
    # ax.set_xlabel('components_1', fontsize=15)
    # ax.set_ylabel('componets_2', fontsize=15)
    # ax.set_title('componentes principals', fontsize = 15)

    # color_theme = np.array(["blue","green", "orange"])
    # ax.scatter(x=pca_nombres_signos.components_1, y = pca_nombres_signos.components_2,
    #     c= color_theme[pca_nombres_signos.Kmeans_clusters], s = 50)
  
    # plt.show()
    # indices=[]
    # muestras = pd.DataFrame(data.loc[indices], columns=data.keys()).reset_index(drop = True)
    # datos = data.drop(indices, axis=0)
    
    return render_template('k-means.html', preguntas = data)
    


# starting the app
if __name__ == "__main__":
    app.run(port=3000, debug=True)